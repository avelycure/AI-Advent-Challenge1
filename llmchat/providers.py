"""Описание поддерживаемых LLM-провайдеров и их моделей.

Все четыре провайдера принимают тело запроса в формате OpenAI, поэтому запросы
выполняет один SDK. Различия вынесены в описание провайдера:

* DeepSeek и OpenAI — ключ уходит в заголовок как есть;
* YandexGPT — модель задаётся URI ``gpt://<каталог>/<модель>/latest``,
  поэтому кроме ключа нужен идентификатор каталога;
* GigaChat — ключ авторизации сначала меняется на access-токен по OAuth,
  токен живёт 30 минут и обновляется автоматически.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Сколько токенов резервируем под ответ модели: на столько же выставляется
# max_tokens запроса, и на столько же уменьшается полезный объём контекста.
OUTPUT_RESERVE_CAP = 4096


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    context_window: int
    max_output: int
    # Поля тела запроса, которые нужны именно этой модели.
    extra_body: Dict[str, object] = field(default_factory=dict)
    # Цена за миллион токенов в долларах. None означает «цена неизвестна»:
    # программа покажет прочерк, а не выдуманное число.
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    cached_price: Optional[float] = None
    # Валюта прайса. Складывать рубли с долларами нельзя, поэтому итог по
    # сессии считается по каждой валюте отдельно.
    currency: str = "USD"
    # Оговорка к цене: второй тариф, срок действия, бесплатный лимит.
    price_note: str = ""

    @property
    def output_reserve(self) -> int:
        return min(self.max_output, OUTPUT_RESERVE_CAP)

    @property
    def priced(self) -> bool:
        return self.input_price is not None and self.output_price is not None


@dataclass(frozen=True)
class ExtraField:
    """Дополнительный реквизит, который нужен провайдеру помимо ключа."""

    title: str
    hint: str
    help_text: str
    # Файлы, откуда значение можно взять, чтобы не вводить его каждый раз.
    files: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OAuthInfo:
    """Обмен долгоживущего ключа авторизации на короткий access-токен."""

    url: str
    scope: str
    # Токен живёт 30 минут; обновляем заранее, чтобы не словить 401 в середине запроса.
    refresh_margin_seconds: int = 120


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    name: str
    base_url: Optional[str]
    api_key_env: str
    token_url: str
    accent: str
    key_hint: str
    models: List[ModelInfo]
    # key_title — для заголовка панели, key_phrase — для середины предложения,
    # чтобы не приходилось менять регистр на лету и не получать «api-ключ».
    key_title: str = "API-ключ"
    key_phrase: str = "API-ключ"
    # Локальные файлы с ключом, в порядке предпочтения. Программа их только
    # читает: сама она ключи никуда не записывает.
    key_files: List[str] = field(default_factory=list)
    # Адрес для проверки ключа. Нужен там, где список моделей открыт без
    # авторизации и потому ключ не проверяет: у OpenRouter это именно так.
    validate_url: Optional[str] = None
    extra_field: Optional[ExtraField] = None
    model_uri_template: Optional[str] = None
    oauth: Optional[OAuthInfo] = None
    # Поля тела запроса, которые понимает только этот провайдер.
    extra_body: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Тариф без оплаты: считать такие запросы по нулю честнее, чем прочерком.
    free: bool = False

    @property
    def default_model(self) -> ModelInfo:
        return self.models[0]

    def model_ref(self, model: ModelInfo, extra: Optional[str]) -> str:
        """Строка, которая уйдёт в поле ``model`` запроса."""
        if self.model_uri_template and extra:
            return self.model_uri_template.format(extra=extra, model=model.id)
        return model.id

    def extra_body_for(self, model_ref: str) -> Dict[str, object]:
        """Служебные поля запроса для конкретной модели поверх общих для провайдера."""
        body = dict(self.extra_body)
        for model in self.models:
            if model.id == model_ref or model_ref.endswith("/" + model.id):
                body.update(model.extra_body)
                break
        return body


def request_cost(provider: ProviderInfo, model: ModelInfo, prompt_tokens: int,
                 completion_tokens: int, cached_tokens: int = 0) -> Optional[float]:
    """Стоимость одного запроса в долларах. None — если цена модели неизвестна.

    Токены размышления отдельно не считаются: провайдер уже включил их в
    ``completion_tokens`` и тарифицирует как обычный выход.
    """
    if provider.free:
        return 0.0
    if not model.priced:
        return None
    cached = max(0, min(cached_tokens, prompt_tokens))
    cached_price = model.cached_price if model.cached_price is not None else model.input_price
    total = ((prompt_tokens - cached) * model.input_price
             + cached * cached_price
             + completion_tokens * model.output_price)
    return total / 1_000_000


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        key="openai",
        name="ChatGPT (OpenAI)",
        base_url=None,  # дефолтный https://api.openai.com/v1
        api_key_env="OPENAI_API_KEY",
        token_url="https://platform.openai.com/api-keys",
        accent="green",
        key_hint="ключ начинается с sk-",
        key_files=["~/.openai-key", "~/.config/llm-chat/openai.key"],
        models=[
            ModelInfo("gpt-5.4-mini", "GPT-5.4 mini — быстрая и дешёвая, вход до 272k",
                      400_000, 128_000,
                      input_price=0.75, output_price=4.50, cached_price=0.075),
            ModelInfo("gpt-5.4-nano", "GPT-5.4 nano — самая дешёвая, вход до 272k",
                      400_000, 128_000,
                      input_price=0.20, output_price=1.25, cached_price=0.02),
            ModelInfo("gpt-5.4", "GPT-5.4 — окно на 1,05 млн токенов", 1_050_000, 128_000,
                      input_price=2.50, output_price=15.00, cached_price=0.25),
            ModelInfo("gpt-5.5", "GPT-5.5 — сильнее 5.4; temperature не принимает",
                      1_050_000, 128_000,
                      input_price=5.00, output_price=30.00, cached_price=0.50),
            ModelInfo("gpt-6-astra", "GPT-6 Astra — новейшая; temperature не принимает",
                      1_050_000, 128_000,
                      input_price=10.00, output_price=50.00, cached_price=1.00),
        ],
    ),
    "deepseek": ProviderInfo(
        key="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        token_url="https://platform.deepseek.com/api_keys",
        accent="magenta",
        key_hint="ключ начинается с sk-",
        key_files=["~/.deepseek-key", "~/.llm-test-key", "~/.config/llm-chat/deepseek.key"],
        models=[
            # Цена намеренно не задана: в документации DeepSeek этих имён больше
            # нет — прайс published только для deepseek-v4-flash и -v4-pro,
            # и переносить их цену на старые псевдонимы было бы выдумкой.
            ModelInfo("deepseek-chat", "DeepSeek Chat — универсальная модель", 64_000, 8_192,
                      price_note="в документации этого имени больше нет, цена неизвестна"),
            ModelInfo("deepseek-reasoner", "DeepSeek Reasoner — с цепочкой рассуждений",
                      64_000, 8_192,
                      price_note="в документации этого имени больше нет, цена неизвестна"),
        ],
    ),
    "yandex": ProviderInfo(
        key="yandex",
        name="YandexGPT",
        base_url="https://llm.api.cloud.yandex.net/v1",
        api_key_env="YANDEX_API_KEY",
        token_url="https://yandex.cloud/ru/docs/iam/operations/api-key/create",
        accent="red",
        key_hint="API-ключ сервисного аккаунта, начинается с AQVN",
        key_title="API-ключ",
        key_phrase="API-ключ",
        key_files=["~/.yandex-key", "~/.config/llm-chat/yandex.key"],
        extra_field=ExtraField(
            title="Идентификатор каталога (folder ID)",
            hint="выглядит как b1g…, виден в консоли Yandex Cloud",
            help_text="Модель у Яндекса задаётся адресом gpt://<каталог>/<модель>/latest, "
                      "поэтому кроме ключа нужен идентификатор каталога.",
            files=["~/.yandex-folder", "~/.config/llm-chat/yandex.folder"],
        ),
        model_uri_template="gpt://{extra}/{model}/latest",
        models=[
            ModelInfo("yandexgpt-lite", "YandexGPT Lite — быстрая и дешёвая", 32_000, 2_000,
                      input_price=200, output_price=200, cached_price=200, currency="RUB",
                      price_note="0,2 ₽ за 1000 токенов, включая НДС"),
            ModelInfo("yandexgpt", "YandexGPT Pro — сильнее и дороже", 32_000, 2_000,
                      input_price=800, output_price=800, cached_price=800, currency="RUB",
                      price_note="0,8 ₽ за 1000 токенов (Pro 5.1, его и даёт /latest), "
                                 "включая НДС"),
        ],
        notes=["Ключ и каталог берутся в консоли Yandex Cloud: нужен сервисный аккаунт "
               "с ролью ai.languageModels.user."],
    ),
    "grok": ProviderInfo(
        key="grok",
        name="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        token_url="https://console.x.ai",
        accent="bright_cyan",
        key_hint="ключ начинается с xai-",
        key_files=["~/.grok-key", "~/.config/llm-chat/grok.key"],
        # Названия взяты из прав самого ключа (эндпоинт /v1/api-key), а не из
        # документации: у разных аккаунтов набор моделей отличается.
        models=[
            ModelInfo("grok-4.3", "Grok 4.3 — самая дешёвая из доступных", 131_072, 8_192,
                      input_price=1.25, output_price=2.50, cached_price=0.20,
                      price_note="при промпте от 200 000 токенов тариф вдвое выше"),
            ModelInfo("grok-4.5", "Grok 4.5 — сбалансированная", 131_072, 8_192,
                      input_price=2.00, output_price=6.00, cached_price=0.30,
                      price_note="при промпте от 200 000 токенов тариф вдвое выше"),
            ModelInfo("grok-4.6", "Grok 4.6 — самая свежая", 131_072, 8_192,
                      input_price=2.00, output_price=6.00, cached_price=0.50,
                      price_note="при промпте от 200 000 токенов тариф вдвое выше"),
        ],
        notes=["Бесплатного тарифа у xAI нет: без купленных кредитов API отвечает "
               "отказом на любой запрос, включая список моделей.",
               "Набор моделей зависит от аккаунта. Свой список видно в консоли "
               "console.x.ai или запросом к https://api.x.ai/v1/api-key."],
    ),
    "openrouter": ProviderInfo(
        key="openrouter",
        name="OpenRouter (бесплатные модели)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        token_url="https://openrouter.ai/keys",
        accent="bright_magenta",
        key_hint="ключ начинается с sk-or-v1-",
        key_files=["~/.openrouter-key", "~/.config/llm-chat/openrouter.key"],
        validate_url="https://openrouter.ai/api/v1/key",
        # Модели и размеры окон взяты из открытого каталога openrouter.ai/api/v1/models:
        # там указана цена, и у перечисленных ниже она равна нулю за ввод и за вывод.
        models=[
            ModelInfo("minimax/minimax-m3:free", "MiniMax M3 — окно на миллион токенов, отвечает быстро",
                      1_048_576, 943_718),
            ModelInfo("z-ai/glm-5.2:free", "GLM 5.2 от Z.ai — рассуждающая модель",
                      256_000, 230_400),
        ],
        notes=["Один ключ открывает модели разных разработчиков. Обе перечисленные "
               "бесплатны: в каталоге провайдера у них нулевая цена. Ключ выдаётся "
               "без привязки карты.",
               "Бесплатные модели делят общую очередь и временами отвечают отказом "
               "«перегружена». По проверке MiniMax отвечает стабильнее и быстрее, "
               "GLM — рассуждающая модель, ей нужен max_tokens побольше."],
        free=True,
    ),
    "groq": ProviderInfo(
        key="groq",
        name="Groq (бесплатный тариф)",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        token_url="https://console.groq.com/keys",
        accent="bright_yellow",
        key_hint="ключ начинается с gsk_",
        key_files=["~/.groq-key", "~/.config/llm-chat/groq.key"],
        # Размеры окон взяты из ответа самого /v1/models, а не из документации:
        # у qwen3.8 там 131 042, а не круглые 131 072.
        # Все модели Groq рассуждают перед ответом, и рассуждение тратит тот же
        # max_tokens, что и сам ответ. Без ограничения короткий запрос возвращает
        # пустой текст: лимит уходит на размышление. Qwen позволяет отключить его
        # совсем, GPT-OSS принимает только low, medium и high, а системы compound
        # не принимают поле вовсе.
        models=[
            ModelInfo("qwen/qwen3.8-27b", "Qwen 3.8 27B — самая свежая", 131_042, 16_384,
                      extra_body={"reasoning_effort": "none"}),
            ModelInfo("qwen/qwen3.6-27b", "Qwen 3.6 27B — предыдущая версия", 131_072, 16_384,
                      extra_body={"reasoning_effort": "none"}),
            ModelInfo("openai/gpt-oss-120b", "GPT-OSS 120B — самая крупная", 131_072, 65_536,
                      extra_body={"reasoning_effort": "low"}),
            ModelInfo("openai/gpt-oss-20b", "GPT-OSS 20B — самая быстрая, около 1000 токенов в секунду",
                      131_072, 65_536, extra_body={"reasoning_effort": "low"}),
            ModelInfo("groq/compound", "Compound — модель с веб-поиском и запуском кода", 131_072, 8_192),
            ModelInfo("groq/compound-mini", "Compound Mini — облегчённая", 131_072, 8_192),
        ],
        notes=["Бесплатный тариф выдаётся без привязки карты. По заголовкам ответа "
               "лимиты такие: 1000 запросов в сутки и 8000 токенов в минуту.",
               "Лимит в 8000 токенов в минуту тратится и на запрос, и на ответ, "
               "поэтому длинный диалог упрётся в него раньше, чем в размер окна.",
               "Llama 3.1 и 3.3 из документации обычному ключу не выдаются: Groq "
               "перевёл их в тариф Enterprise."],
        free=True,
    ),
    "gemini": ProviderInfo(
        key="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        token_url="https://aistudio.google.com/apikey",
        accent="blue",
        key_hint="ключ из Google AI Studio, начинается с AIza",
        key_files=["~/.gemini-key", "~/.config/llm-chat/gemini.key"],
        models=[
            ModelInfo("gemini-3.7-flash", "Gemini 3.7 Flash — самая свежая", 1_048_576, 65_536,
                      input_price=0.75, output_price=3.75,
                      price_note="акционная цена до 31.12.2026; с 1 января 2027 — "
                                 "$1.50 вход и $7.50 выход"),
            ModelInfo("gemini-3.5-flash", "Gemini 3.5 Flash", 1_048_576, 65_536,
                      input_price=1.50, output_price=9.00,
                      price_note="платный тариф; у ключа может быть бесплатный лимит"),
            ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash — проверенная временем",
                      1_048_576, 65_536,
                      input_price=0.30, output_price=2.50,
                      price_note="цена для текста; аудио на входе дороже — $1.00"),
        ],
        notes=["У Google AI Studio есть бесплатный тариф с ограничением по числу "
               "запросов в сутки; карта для получения ключа не нужна.",
               "Список моделей по OpenAI-совместимому пути не отдаётся, поэтому "
               "проверка ключа откладывается до первого запроса."],
    ),
    "gigachat": ProviderInfo(
        key="gigachat",
        name="GigaChat (Сбер)",
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key_env="GIGACHAT_CREDENTIALS",
        token_url="https://developers.sber.ru/studio/workspaces",
        accent="bright_green",
        key_hint="ключ авторизации из личного кабинета, длинная строка Base64",
        key_title="Ключ авторизации",
        key_phrase="ключ авторизации",
        key_files=["~/.gigachat-key", "~/.config/llm-chat/gigachat.key"],
        oauth=OAuthInfo(
            url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            scope="GIGACHAT_API_PERS",  # тариф для физических лиц
        ),
        models=[
            ModelInfo("GigaChat-2", "GigaChat 2 Lite — быстрая, входит в бесплатный лимит",
                      128_000, 4_096,
                      input_price=65, output_price=65, currency="RUB",
                      price_note="0,065 ₽ за 1000 токенов — выведено из цены пакета; "
                                 "отдельного прайса за токен Сбер не публикует"),
            ModelInfo("GigaChat-2-Pro", "GigaChat 2 Pro — для сложных задач", 128_000, 4_096,
                      input_price=500, output_price=500, currency="RUB",
                      price_note="0,5 ₽ за 1000 токенов — выведено из цены пакета"),
            ModelInfo("GigaChat-2-Max", "GigaChat 2 Max — самая мощная", 128_000, 4_096,
                      input_price=650, output_price=650, currency="RUB",
                      price_note="0,65 ₽ за 1000 токенов — выведено из цены пакета"),
        ],
        notes=["Ключ авторизации меняется на access-токен по OAuth; токен живёт 30 минут "
               "и обновляется программой автоматически.",
               "Сервер Сбера использует сертификат НУЦ Минцифры. Если увидите ошибку "
               "проверки сертификата, установите его командой: "
               "curl -k https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt "
               ">> $(python -m certifi)"],
    ),
}

PROVIDER_ORDER: List[str] = ["deepseek", "openai", "yandex", "gigachat",
                             "grok", "openrouter", "groq", "gemini"]
