# День 6. План реализации агента и его ревью

Цель — закрыть требования 6–15 из [day6-agent-requirements.md](day6-agent-requirements.md).
Файл писался в три захода: сперва набросок, затем ревью, которое нашло в нём
дыры, и затем итоговый план, по которому работа и велась. Все три оставлены —
из них видно, где рассуждение было неверным и почему.

## 1. Первый набросок и что в нём оказалось не так

Замысел был правильный: новый пакет `llmagent/` — коробка без единого `print`,
внутри input policy → транспорт → output policy → judge → учёт токенов, снаружи
`Agent(config)`. Дыр в нём нашлось четырнадцать, и одна принципиальная.

**Принципиальная.** Набросок предлагал `llmagent` поверх `llmchat.client` —
то есть коробка зависела бы от приложения, и «изолированный модуль» остался бы
словами. Проверка импортов показала, что `client`, `providers`, `tokens`,
`secrets`, `params`, `controls`, `strategies` и `session` **уже не тянут
`rich`**: его импортируют только `app.py`, `ui.py` и `switching.py`. Значит
зависимость надо не добавлять, а разворачивать — перенести rich-free модули
в `llmagent`, и пусть `llmchat` зависит от него, а не наоборот.

| # | Дыра в наброске | Чем закрыта |
|---|---|---|
| 1 | изоляция была бы фиктивной | перенос модулей + тест «`llmagent` не импортирует `rich` и `llmchat`» |
| 2 | замороженный конфиг несовместим с `/change_llm_params` и `/change_model` на лету | `Agent.reconfigure()` через `dataclasses.replace` |
| 3 | YandexGPT не спавнится: ему нужен folder id | поле `api_extra` |
| 4 | `DemoClient` спит 1,2–2,2 с случайно — тест на 100 агентов шёл бы минуты | `Transport.demo_delay` |
| 5 | переспрос по выходной политике не был описан | `max_attempts`, отдельный вид расхода, история не засоряется |
| 6 | судья — тоже расход, а учёт был только для основного вызова | `UsageMeter.by_kind` и `absorb` |
| 7 | бюджет был полем конфига без правила применения | предпроверка `UsageMeter.check` и `BudgetExceeded` |
| 8 | ошибка одного агента уронила бы пачку из ста | `AgentResult.error`, `spawn.ask_safely` |
| 9 | кэш клиентов из ста потоков | `threading.Lock` в `ClientRegistry` |
| 10 | `AgentConfig(provider="foo")` падал бы где-то в глубине транспорта | `config.resolve()` с перечнем доступных |
| 11 | инфраструктуры тестов не было вовсе | `pytest`, `requirements-dev.txt`, `./test.sh` |
| 12 | проверка ментора нигде не выполнялась | `./spawn.sh` — реально спавнит 100 агентов и печатает время |
| 13 | не было примеров конфигов | `configs/*.yaml` |
| 14 | шаг «перевести опыты на агента» был раздут до переписывания | опыты идут через `Agent.ask_messages`, презентация остаётся своя |

## 2. Итоговая структура

```
llmagent/                коробка: ни одного импорта rich и llmchat
  config.py              AgentConfig и части, DEFAULT_CONFIG, сериализация
  agent.py               фасад Agent: ask, answer_pending, retry_last, ask_messages
  policies.py            применение входной и выходной политик
  formats.py             форматы, схема, разбор и проверка ответа
  judge.py               промпт судьи и разбор вердикта
  usage.py               UsageMeter: токены, время, деньги, лимиты
  history.py             Conversation и Message
  registry.py            реквизиты без вопросов человеку, кэш клиентов
  result.py              AgentResult
  spawn.py               spawn, matrix, summarize
  errors.py              ConfigError, MissingCredentials, InputRejected, …
  params.py              параметры генерации   (перенос из llmchat)
  strategies.py          способы постановки задачи (перенос из llmchat)
  transport/             client, providers, tokens, secrets (перенос из llmchat)

llmchat/                 терминальный интерфейс — потребитель агента
  session.py             обёртка агента для экрана
  app.py, ui.py, switching.py

configs/                 примеры конфигов
tests/                   pytest, всё на заглушках
spawn_demo.py, spawn.sh  проверка ментора
test.sh                  прогон тестов
```

Зависимость идёт строго в одну сторону: `llmchat` → `llmagent`. Обратного
импорта нет, и это проверяется тестом, а не обещанием.

## 3. Дефолтный конфиг

Модель по умолчанию — ChatGPT `gpt-5.4-mini`: штатная модель провайдера
`openai`, самая дешёвая из пригодных (0,75 $ вход / 4,50 $ выход за 1M) и,
в отличие от `gpt-5.5` и `gpt-6-astra`, принимающая `temperature`.

```python
DEFAULT_CONFIG = AgentConfig(
    name="default",
    provider="openai",
    model="gpt-5.4-mini",
    api_key=None,                 # None — взять из OPENAI_API_KEY или ~/.openai-key
    api_extra=None,               # второй реквизит; нужен только YandexGPT
    system_prompt=SYSTEM_PROMPT,
    generation=GenerationParams(temperature=0.7),
    input=InputPolicy(max_chars=8000, forbid_empty=True),
    output=OutputPolicy(format="free"),
    judge=None,
    history=HistoryConfig(enabled=True, keep_last_answer=True),
    budget=Budget(),
    transport=Transport(demo=False),
)
```

Передача извне — четырьмя путями через один конструктор:

```python
Agent(DEFAULT_CONFIG)
Agent(DEFAULT_CONFIG.with_changes(model="gpt-5.4"))
Agent(AgentConfig.from_file("configs/books-json.yaml"))
Agent(AgentConfig.from_dict(payload))          # тело HTTP-запроса
```

Для чата — `./run.sh --config configs/demo.yaml`; команда `/config` печатает
действующий конфиг в том же виде, в каком его можно сохранить в файл.

## 4. Шаги и условия приёмки

| № | Шаг | Условие, при котором шаг считается сделанным |
|---|---|---|
| 1 | перенести rich-free модули в `llmagent/`, развернуть зависимость | `llmchat` импортирует `llmagent`, обратного импорта нет |
| 2 | `formats.py`: схема задаётся снаружи и сериализуема | схема `name/year/note` больше нигде не зашита; `Field.kind` — строка, а не `type` |
| 3 | `config.py`: `AgentConfig` и `DEFAULT_CONFIG` | `from_dict(to_dict(c)) == c`; ключ в выводе заменён на `***` |
| 4 | `usage.py`: `UsageMeter` с разбивкой по назначению | цифры `/stats` не изменились; переспрос и оценка видны отдельными строками |
| 5 | `registry.py`: реквизиты без `Console`, кэш клиентов под замком | агент создаётся из кода; сто агентов открывают не больше одного соединения на провайдера |
| 6 | `policies.py`: обе политики | входная отклоняет запрос до провайдера; выходная ловит формат, схему, обрыв и обрамление |
| 7 | `judge.py`: судья со своим конфигом | судить может другой провайдер; сломанный судья не рушит ответ |
| 8 | `agent.py`: фасад с конвейером и переспросом | негодный ответ и замечание к нему не попадают в историю диалога |
| 9 | тест изоляции | импорт `llmagent` не тянет `rich`; в исходниках нет `print` и `input` |
| 10 | `spawn.py` | 100 конфигов отвечают, ошибка одного не роняет пачку |
| 11 | чат переключается на агента | сквозной прогон `chat.py` доходит до выхода; `/change_model`, `/retry`, `/change_llm_params` работают |
| 12 | `compare.py` и `reasoning.py` — через `Agent.ask_messages` | оба скрипта проходят в демо-режиме; расход опытов попадает в счётчик агента |
| 13 | `configs/*.yaml` | каждый пример загружается и разрешается в существующую модель |
| 14 | `./spawn.sh` и `./test.sh` | проверка ментора выполняется одной командой и печатает время |
| 15 | документация | `docs/feature-agent.md`, разделы в `README.md` и `ARCHITECTURE.md` |

## 5. Решения по дороге

| Вопрос | Решение |
|---|---|
| откуда агент берёт ключ без человека | `config.api_key` → `OPENAI_API_KEY` → `~/.openai-key` → `MissingCredentials`. Спрашивает интерфейс и кладёт найденное в конфиг |
| синхронный SDK и параллельность | `asyncio.to_thread` поверх нынешнего клиента и семафор: ноль правок транспорта. `AsyncOpenAI` — если упрёмся в пропускную способность |
| ключ в сериализованном конфиге | `to_dict()` по умолчанию заменяет его на `***`; выгрузить целиком можно только явным `with_secrets=True` |
| деньги | всё проверяется на `DemoClient`; `./spawn.sh --live` спрашивает подтверждение и называет число запросов |
| `Message.accent` — цвет в доменной модели | оставлен: `ProviderInfo.accent` существовал и раньше, а вычищать его значило бы трогать каждую панель ради ровности |
| REST-микросервис | не делается: ментор назвал его **альтернативой** изолированному модулю, а не дополнением. `AgentConfig.from_dict` и `AgentResult.to_dict` — готовые концы для обработчика, когда он понадобится |
