"""Сессии: своя память у каждого запуска и вложенные вызовы под-агентов.

Проверяется то, ради чего всё и делалось: две сессии не пересекаются памятью,
а под-агент — отдельный процесс со своей сессией, чей итог возвращается в
вызвавшую сессию вместе со своим расходом.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from llmagent import Agent, AgentConfig, Transport
from llmagent.usage import MAIN, SUB
from llmchat.session import Session
from llmchat import subagent

from conftest import ScriptedClient, child_env

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAST_DEMO = Transport(demo=True, demo_delay=0.0)


def one_shot(question: str, config: pathlib.Path, home: pathlib.Path) -> dict:
    """Запуск в режиме одного вопроса — тот самый, которым идёт под-агент."""
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(config), "--ask", question, "--json"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env(home))
    assert finished.returncode == 0, finished.stderr[-2000:]
    # В stdout обязан быть только JSON: его разбирает вызывающий, и любая
    # посторонняя строка сломала бы разбор.
    return json.loads(finished.stdout)


@pytest.fixture
def fast_config(tmp_path) -> pathlib.Path:
    path = tmp_path / "fast.yaml"
    AgentConfig(name="fast", transport=FAST_DEMO).to_file(str(path))
    return path


# --------------------------------------------------------------------------
# Своя память у каждой сессии
# --------------------------------------------------------------------------

def test_each_agent_gets_its_own_session_id():
    config = AgentConfig(transport=FAST_DEMO)
    assert Agent(config).session_id != Agent(config).session_id


def test_two_sessions_in_one_process_do_not_share_memory():
    """Сценарий «Влад»: левая сессия знает имя, правая — нет."""
    left = Agent(AgentConfig(transport=FAST_DEMO),
                 client=ScriptedClient(["Приятно познакомиться!", "Тебя зовут Влад."]))
    right = Agent(AgentConfig(transport=FAST_DEMO),
                  client=ScriptedClient(["Не знаю, ты не сообщал."]))

    left.ask("Привет, меня зовут Влад")
    left.ask("Как меня зовут?")
    right.ask("Как меня зовут?")

    left_text = " ".join(m.content for m in left.conversation.messages)
    right_text = " ".join(m.content for m in right.conversation.messages)
    assert "Влад" in left_text and "Влад" not in right_text
    assert len(left.conversation) == 4 and len(right.conversation) == 2
    assert left.usage.total_tokens != right.usage.total_tokens


def test_memory_grows_inside_one_session():
    client = ScriptedClient(["раз", "два", "три"])
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=client)
    for question in ("первый", "второй", "третий"):
        agent.ask(question)
    sent = [len(call) for call in client.calls]
    assert sent == sorted(sent) and sent[0] < sent[-1]


def test_separate_processes_start_with_empty_memory(fast_config, tmp_path):
    """Каждый запуск в терминале — своя сессия: истории предыдущего не видно."""
    first = one_shot("Привет, меня зовут Влад", fast_config, tmp_path)
    second = one_shot("Как меня зовут?", fast_config, tmp_path)

    assert first["session_id"] != second["session_id"]
    # Одинаковая длина запроса означает, что второй процесс не унаследовал
    # переписку первого: иначе отправленных токенов стало бы заметно больше.
    assert second["prompt_tokens"] < first["prompt_tokens"] + 20
    assert second["usage"]["requests"] == 1


def test_one_shot_prints_nothing_but_json(fast_config, tmp_path):
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--ask", "Привет", "--json"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env(tmp_path))
    payload = json.loads(finished.stdout)          # упадёт, если вывод засорён
    assert finished.stdout.strip().startswith("{")
    assert payload["ok"] and payload["text"]


def test_one_shot_without_json_prints_plain_text(fast_config, tmp_path):
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--ask", "Привет"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env(tmp_path))
    assert finished.returncode == 0
    assert "Вы спросили" in finished.stdout
    assert "{" not in finished.stdout.splitlines()[0]


def test_one_shot_never_asks_anything(tmp_path):
    """Дочерний процесс не должен ждать ввода: спрашивать ему некого.

    Единственный тест с боевым конфигом, поэтому домашний каталог подменён
    пустым: агент не должен найти ни ключа, ни повода отправить запрос.
    """
    home = tmp_path / "empty-home"
    home.mkdir()
    path = tmp_path / "live.yaml"
    AgentConfig(name="live").to_file(str(path))     # боевой режим, ключа нет
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(path), "--ask", "Привет", "--json"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env(home),
        input="", timeout=60)
    payload = json.loads(finished.stdout)
    assert finished.returncode == 3
    assert payload["ok"] is False and "не найден" in payload["error"]


# --------------------------------------------------------------------------
# Вложенный вызов
# --------------------------------------------------------------------------

def test_subagent_runs_in_its_own_process_and_session():
    parent = AgentConfig(name="parent", transport=Transport(demo=True, demo_delay=0.0))
    delegation = subagent.run("frugal", "Что такое хороший код", parent)

    assert delegation.ok, delegation.error
    assert delegation.session_id and delegation.text
    assert delegation.model == "gpt-5.4-nano"      # конфиг под-агента, не родителя
    assert delegation.total_tokens > 0


def test_subagent_result_reaches_the_calling_session():
    parent = Agent(AgentConfig(name="parent", transport=Transport(demo=True, demo_delay=0.0)),
                   client=ScriptedClient("ответ родителя"))
    delegation = subagent.run("frugal", "Что такое хороший код", parent.config)
    assert delegation.ok, delegation.error

    parent.record_delegation(delegation.name, delegation.question, delegation.text)
    parent.absorb_delegation(delegation.prompt_tokens, delegation.completion_tokens,
                             delegation.seconds, delegation.cost, delegation.currency)

    note = parent.conversation.messages[-1]
    assert note.note and note.role == "user" and note.model == "frugal"
    assert delegation.text.strip()[:20] in note.content
    # Расход под-агента виден отдельной строкой: платил тот же кошелёк.
    assert parent.usage.by_kind[SUB].requests == 1
    assert parent.usage.by_kind[MAIN].requests == 0
    assert parent.usage.total_tokens == delegation.total_tokens


def test_subagent_memory_never_touches_the_parent():
    parent = Agent(AgentConfig(name="parent", transport=Transport(demo=True, demo_delay=0.0)),
                   client=ScriptedClient("ответ родителя"))
    parent.ask("Привет, меня зовут Влад")
    before = [m.content for m in parent.conversation.messages]

    first = subagent.run("frugal", "Как меня зовут?", parent.config)
    second = subagent.run("frugal", "Как меня зовут?", parent.config)

    # Два под-агента подряд — две разные сессии, ни одна не видела «Влада».
    assert first.session_id != second.session_id
    assert "Влад" not in first.text and "Влад" not in second.text
    assert [m.content for m in parent.conversation.messages] == before


def test_unknown_config_is_reported_not_raised():
    parent = AgentConfig(transport=Transport(demo=True, demo_delay=0.0))
    delegation = subagent.run("нет-такого", "вопрос", parent)
    assert not delegation.ok
    assert "нет конфига" in delegation.error and "frugal" in delegation.error


def test_broken_output_policy_is_not_a_failed_call():
    """Под-агент отработал, но формы не соблюл — это разные вещи."""
    parent = AgentConfig(transport=Transport(demo=True, demo_delay=0.0))
    delegation = subagent.run("books-json", "Книги Талеба", parent)
    assert delegation.ok            # вызов удался
    assert not delegation.valid     # но заглушка отвечает прозой, а нужен JSON
    assert delegation.text


def test_credentials_go_through_the_environment_not_the_command_line():
    """Ключ в argv виден в выводе ps любому процессу пользователя."""
    parent = AgentConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-parent-secret")
    child = AgentConfig(provider="openai", model="gpt-5.4-nano")
    environment = subagent.child_environment(parent, child)
    assert environment["OPENAI_API_KEY"] == "sk-parent-secret"

    # Другому провайдеру родительский ключ не передаётся — он ему не подойдёт.
    other = AgentConfig(provider="deepseek", model="deepseek-chat")
    assert subagent.child_environment(parent, other).get("OPENAI_API_KEY") != "sk-parent-secret"


def test_catalog_lists_the_shipped_configs():
    names = subagent.available()
    assert {"demo", "default", "books-json", "reviewer", "frugal"} <= set(names)


# --------------------------------------------------------------------------
# Правки конфига из командной строки
# --------------------------------------------------------------------------

def run_cli(*arguments, home: pathlib.Path = None, expect: int = 0):
    """Настоящая программа целиком. Домашний каталог подменён: ни живого API,
    ни настоящего ~/.llm-agent тесты касаться не должны."""
    environment = child_env(home)
    finished = subprocess.run([sys.executable, "chat.py", *arguments],
                              cwd=str(ROOT), capture_output=True, text=True,
                              input="", env=environment, timeout=120)
    assert finished.returncode == expect, (finished.returncode, finished.stdout[-1500:],
                                           finished.stderr[-1500:])
    return finished


def test_flags_override_the_config(fast_config, tmp_path):
    shown = run_cli("--config", str(fast_config), "--model", "gpt-5.4-nano",
                    "--temperature", "0.15", "--max-tokens", "64",
                    "--format", "json", "--max-cost", "0.02", "--no-history",
                    "--show-config", home=tmp_path).stdout
    for expected in ("gpt-5.4-nano", "0.15", "64", "json", "0.02"):
        assert expected in shown
    assert "False" in shown            # history.enabled


def test_set_beats_a_named_flag(fast_config, tmp_path):
    """Порядок слоёв обещан в справке, значит его надо закрепить."""
    shown = run_cli("--config", str(fast_config), "--temperature", "0.2",
                    "--set", "generation.temperature=0.9",
                    "--show-config", home=tmp_path).stdout
    assert "0.9" in shown and "0.2" not in shown


def test_config_can_be_given_as_json(tmp_path):
    shown = run_cli("--config", '{"model": "gpt-5.4-nano", "budget": {"max_cost": 0.03}}',
                    "--demo", "--show-config", home=tmp_path).stdout
    assert "gpt-5.4-nano" in shown and "0.03" in shown


def test_broken_json_config_is_reported(tmp_path):
    finished = run_cli("--config", '{"model": ', "--demo", "--show-config",
                       home=tmp_path, expect=2)
    assert "не разобран" in finished.stdout


def test_appended_prompt_is_visible_in_one_row(fast_config, tmp_path):
    """Многострочное значение не должно рвать строку поля пополам."""
    shown = run_cli("--config", str(fast_config),
                    "--append-system-prompt", "ХВОСТ-МЕТКА",
                    "--show-config", home=tmp_path).stdout
    rows = [line for line in shown.splitlines() if "system_prompt" in line]
    assert len(rows) == 1 and "ХВОСТ-МЕТКА" in rows[0]


def test_show_config_never_prints_the_key(tmp_path):
    shown = run_cli("--demo", "--set", "api_key=sk-очень-секретный",
                    "--show-config", home=tmp_path).stdout
    assert "очень-секретный" not in shown
    assert "***" in shown


def test_unknown_field_is_reported_with_alternatives(tmp_path):
    finished = run_cli("--demo", "--set", "temperatura=0.2", "--show-config",
                       home=tmp_path, expect=2)
    assert "temperature" in finished.stdout


def test_one_shot_applies_the_overrides(fast_config, tmp_path):
    finished = run_cli("--config", str(fast_config), "--ask", "Привет", "--json",
                       "--model", "gpt-5.4-nano", "--set", "name=разовый",
                       home=tmp_path)
    payload = json.loads(finished.stdout)
    assert payload["model"] == "gpt-5.4-nano"
    assert payload["agent"] == "разовый"


def test_subagent_receives_the_tweaks():
    """Правка должна доехать до дочернего процесса и там примениться.

    Сверяем по модели и подписи: заглушка не смотрит на предел длины и не
    смотрит на температуру, а вот чем именно её позвали — видно в ответе.
    """
    parent = AgentConfig(name="parent", transport=Transport(demo=True, demo_delay=0.0))
    plain = subagent.run("frugal", "вопрос", parent)
    tuned = subagent.run("frugal", "вопрос", parent, ["model=gpt-5.4", "name=подправленный"])
    assert plain.ok and tuned.ok, (plain.error, tuned.error)
    assert plain.model == "gpt-5.4-nano"       # как задано в configs/frugal.yaml
    assert tuned.model == "gpt-5.4"            # как задано правкой


def test_subagent_tweak_mistake_is_caught_before_the_process_starts():
    parent = AgentConfig(transport=Transport(demo=True, demo_delay=0.0))
    delegation = subagent.run("frugal", "вопрос", parent, ["темпе=0.1"])
    assert not delegation.ok and "нет поля" in delegation.error


# --------------------------------------------------------------------------
# Сессии на диске
# --------------------------------------------------------------------------

def chat(*arguments, home: pathlib.Path, script: str = "/exit\n", expect: int = 0):
    finished = subprocess.run([sys.executable, "chat.py", *arguments],
                              cwd=str(ROOT), capture_output=True, text=True,
                              input=script, env=child_env(home), timeout=180)
    assert finished.returncode == expect, (finished.returncode, finished.stderr[-1500:])
    return finished.stdout


def test_dialogue_is_saved_and_can_be_continued(fast_config, tmp_path):
    chat("--config", str(fast_config), home=tmp_path,
         script="Привет, меня зовут Влад\n/exit\n")

    listing = chat("--sessions", home=tmp_path)
    assert "Сохранённые сессии" in listing

    returned = chat("--continue", home=tmp_path, script="/history\n/exit\n")
    assert "Возвращаюсь в сессию" in returned
    assert "Влад" in returned


def test_one_shot_is_not_saved(fast_config, tmp_path):
    one_shot("Привет", fast_config, tmp_path)
    assert "Сохранённых сессий нет" in chat("--sessions", home=tmp_path)


def test_no_save_keeps_the_disk_clean(fast_config, tmp_path):
    chat("--config", str(fast_config), "--no-save", home=tmp_path,
         script="Привет\n/exit\n")
    assert "Сохранённых сессий нет" in chat("--sessions", home=tmp_path)


def test_continue_without_sessions_says_so(tmp_path):
    finished = subprocess.run([sys.executable, "chat.py", "--continue", "--demo"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              input="", env=child_env(tmp_path), timeout=60)
    assert finished.returncode == 2
    assert "возвращаться некуда" in finished.stdout


def test_resume_names_the_recent_sessions_when_id_is_wrong(fast_config, tmp_path):
    chat("--config", str(fast_config), home=tmp_path, script="Привет\n/exit\n")
    finished = subprocess.run([sys.executable, "chat.py", "--resume", "нетакой"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              input="", env=child_env(tmp_path), timeout=60)
    assert finished.returncode == 2
    assert "нет сессии" in finished.stdout


def test_taken_session_id_is_refused_instead_of_overwriting(fast_config, tmp_path):
    chat("--config", str(fast_config), "--session-id", "мояс", home=tmp_path,
         script="Привет, меня зовут Влад\n/exit\n")
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--session-id", "мояс"],
        cwd=str(ROOT), capture_output=True, text=True, input="",
        env=child_env(tmp_path), timeout=60)
    assert finished.returncode == 2
    assert "--resume" in finished.stdout
    # Прежний разговор должен остаться целым.
    assert "Влад" in chat("--resume", "мояс", home=tmp_path, script="/history\n/exit\n")


def test_removing_one_session_by_id(fast_config, tmp_path):
    chat("--config", str(fast_config), "--session-id", "перв", home=tmp_path,
         script="Привет\n/exit\n")
    chat("--config", str(fast_config), "--session-id", "втор", home=tmp_path,
         script="Привет\n/exit\n")

    assert "удалена" in chat("--rm-session", "перв", home=tmp_path)
    listing = chat("--sessions", home=tmp_path)
    assert "втор" in listing and "перв" not in listing


def test_removing_a_missing_session_is_an_error(tmp_path):
    finished = subprocess.run([sys.executable, "chat.py", "--rm-session", "нетакой"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              input="", env=child_env(tmp_path), timeout=60)
    assert finished.returncode == 1
    assert "Нет сессии" in finished.stdout


def test_removing_all_sessions(fast_config, tmp_path):
    """Обещано, что переписка лежит на диске до тех пор, пока её не удалят."""
    chat("--config", str(fast_config), home=tmp_path, script="Привет\n/exit\n")
    assert "Удалено" in chat("--rm-session", "all", home=tmp_path)
    assert "Сохранённых сессий нет" in chat("--sessions", home=tmp_path)
    assert list((tmp_path / ".llm-agent" / "sessions").glob("*.json")) == []


def test_session_files_never_touch_the_real_home(fast_config, tmp_path):
    """Тесты не должны оставлять следов в настоящем ~/.llm-agent."""
    chat("--config", str(fast_config), home=tmp_path, script="Привет\n/exit\n")
    saved = list((tmp_path / ".llm-agent" / "sessions").glob("*.json"))
    assert len(saved) == 1


def test_switching_model_inside_one_provider_does_not_ask_for_the_key(fast_config, tmp_path):
    """Ключ из конфига программе уже известен, и переспрашивать его нельзя.

    Пока хранилище реквизитов о нём не знало, /change_model съедал следующую
    строку ввода как ответ на вопрос о ключе, и следующая команда пропадала.
    """
    shown = chat("--config", str(fast_config), home=tmp_path,
                 script="Привет\n/change_model 5\n/change_llm_params temperature=0.15\n/exit\n")
    assert "Модель переключена" in shown
    assert "Применено" in shown           # команда после переключения не пропала
    assert "Нужен API-ключ" not in shown


def test_config_changing_commands_are_saved_at_once(fast_config, tmp_path):
    """Переключить модель и выйти, не задав вопроса, — обычное дело."""
    chat("--config", str(fast_config), home=tmp_path,
         script="Привет\n/change_model 5\n/exit\n")
    saved = json.loads(next((tmp_path / ".llm-agent" / "sessions").glob("*.json"))
                       .read_text(encoding="utf-8"))
    assert saved["config"]["model"] == "gpt-5.4"


def test_returned_session_keeps_the_switched_model(fast_config, tmp_path):
    chat("--config", str(fast_config), home=tmp_path,
         script="Привет\n/change_model 5\n/exit\n")
    assert "gpt-5.4 " in chat("--continue", home=tmp_path) + " "


# --------------------------------------------------------------------------
# Найденное ревью. По тесту на каждый дефект.
# --------------------------------------------------------------------------

def test_key_prompt_gets_the_credential_store(tmp_path, monkeypatch):
    """Путь с расспросом ключа: два разных хранилища легко перепутать местами.

    Раз перепутали — и запуск падал с AttributeError сразу после того, как
    выяснялось, что ключа нет. Ни один тест этого не замечал: все передавали
    --config с заглушкой, где ключ не спрашивают вовсе. Поэтому проверка идёт
    по самой развилке, а не через канал ввода — там ввод пароля неустойчив.
    """
    from llmagent import Credentials, SessionStore
    from llmchat import app
    from llmchat.ui import make_console

    # Домашний каталог пустой: настоящий ключ найтись не должен.
    monkeypatch.setenv("HOME", str(tmp_path))

    class FakeKeys:
        def __init__(self):
            self.asked = []

        def has(self, provider):
            return False

        def remember(self, provider, credentials):
            pass

        def ready_for(self, console, provider, step=None):
            self.asked.append(provider.key)
            return Credentials("sk-проба", None, "тест")

    keys = FakeKeys()
    args = app.parse_args(["--config", "default"])
    agent = app.prepare_agent(make_console(), args,
                              SessionStore(str(tmp_path / "sessions")), keys,
                              None, app.read_config("default"))
    assert keys.asked == ["openai"], "ключ спросили не у того хранилища"
    assert agent.config.api_key == "sk-проба"


def test_missing_key_with_a_config_asks_instead_of_crashing(tmp_path):
    """Боевой конфиг без ключа: программа просит его, а не падает."""
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", "default"],
        cwd=str(ROOT), capture_output=True, text=True, input="",
        env=child_env(tmp_path), timeout=60)
    assert "Нужен реквизит" in finished.stdout
    assert "AttributeError" not in finished.stdout + finished.stderr


def test_unusable_session_id_is_refused_before_the_first_answer(fast_config, tmp_path):
    """Иначе ошибка всплыла бы при записи — уже после полученного ответа."""
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--session-id", "!!!"],
        cwd=str(ROOT), capture_output=True, text=True, input="Привет\n/exit\n",
        env=child_env(tmp_path), timeout=60)
    assert finished.returncode == 2
    assert "негодный --session-id" in finished.stdout
    assert "Traceback" not in finished.stderr


def test_saving_failure_does_not_lose_the_answer(fast_config, tmp_path, monkeypatch):
    from llmagent import AgentError, SessionStore

    def refuse(*args, **kwargs):
        raise AgentError("диск отказал")

    store = SessionStore(str(tmp_path / "sessions"))
    monkeypatch.setattr(store, "save", refuse)
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=ScriptedClient("ответ"))
    session = Session(agent)
    # Ровно то, что делает цикл диалога: ответ уже получен, запись сорвалась.
    session.add_user("вопрос")
    result = agent.answer_pending()
    assert result.ok
    with pytest.raises(AgentError):
        store.save(agent)
    assert len(agent.conversation) == 2      # ответ на месте


def test_template_with_a_colon_survives(tmp_path):
    """Пример из README: без объявленного типа поля YAML съедал его в словарь."""
    shown = run_cli("--demo", "--set", "template=Переведи на английский: {input}",
                    "--show-config", home=tmp_path).stdout
    assert "Переведи на английский: {input}" in shown


def test_one_shot_honours_continue(fast_config, tmp_path):
    """Раньше --continue в режиме --ask молча пропадал вместе с конфигом сессии."""
    chat("--config", str(fast_config), home=tmp_path,
         script="Привет, меня зовут Влад\n/exit\n")

    fresh = json.loads(run_cli("--config", str(fast_config), "--ask", "Как меня зовут?",
                               "--json", home=tmp_path).stdout)
    continued = json.loads(run_cli("--continue", "--ask", "Как меня зовут?", "--json",
                                   home=tmp_path).stdout)
    # Переписка ушла в запрос — значит память подхвачена.
    assert continued["prompt_tokens"] > fresh["prompt_tokens"]
    # И заглушка осталась заглушкой: конфиг сессии не потерялся.
    assert continued["ok"] and continued["cost"] is not None


def test_returning_to_a_demo_session_never_reaches_the_network(fast_config, tmp_path):
    """Демонстрационный режим записан в конфиге сессии, и повторять --demo не надо."""
    chat("--config", str(fast_config), home=tmp_path, script="Привет\n/exit\n")
    shown = chat("--continue", home=tmp_path, script="/change_model 5\n/exit\n")
    assert "Модель переключена" in shown
    assert "Нужен API-ключ" not in shown and "API-ключ ·" not in shown


def test_new_session_starts_the_count_afresh():
    """Расход прежнего разговора остался в его записи и в новую сессию не идёт."""
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=ScriptedClient("ответ"))
    agent.ask("первый")
    spent = agent.usage.total_tokens
    assert spent > 0

    first = agent.session_id
    agent.new_session()
    assert agent.session_id != first
    assert agent.usage.total_tokens == 0
    assert agent.usage.requests == 0


def test_failed_form_is_flagged_in_the_parent_memory():
    """Человеку про несоблюдённую форму говорит панель, а модели — оговорка."""
    parent = Agent(AgentConfig(name="parent", transport=FAST_DEMO),
                   client=ScriptedClient("ответ родителя"))
    delegation = subagent.run("books-json", "Книги Талеба", parent.config)
    assert delegation.ok and not delegation.valid

    from llmchat.subagent import caveat_of
    parent.record_delegation(delegation.name, delegation.question, delegation.text,
                             caveat_of(delegation))
    note = parent.conversation.messages[-1].content
    assert "Оговорка" in note and "не прошёл проверку формы" in note


def test_good_form_gets_no_caveat():
    from llmchat.subagent import caveat_of
    from llmchat.subagent import Delegation

    assert caveat_of(Delegation("x", "q", valid=True)) == ""


def test_launcher_works_through_a_symlink(tmp_path):
    """Скрипт запуска задуман для PATH, значит обязан работать через ссылку.

    Пока он искал свой каталог по ссылке, а не по настоящему файлу, запуск
    через симlink создавал окружение рядом с симлинком и падал на поиске
    requirements.txt в ~/.local/bin.
    """
    # Каталог ссылки и домашний каталог — разные: в домашний система кладёт
    # своё, и по нему нельзя судить, намусорил ли скрипт рядом с ссылкой.
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    bin_dir.mkdir()
    home.mkdir()
    link = bin_dir / "agent"
    link.symlink_to(ROOT / "agent")

    finished = subprocess.run([str(link), "--config", "demo", "--set",
                               "transport.demo_delay=0", "--ask", "Привет"],
                              cwd=str(home), capture_output=True, text=True,
                              env=child_env(home), timeout=120)
    assert finished.returncode == 0, finished.stderr[-800:]
    assert "Вы спросили" in finished.stdout
    # Окружение должно подняться в репозитории, а не рядом с ссылкой.
    assert [item.name for item in bin_dir.iterdir()] == ["agent"]
