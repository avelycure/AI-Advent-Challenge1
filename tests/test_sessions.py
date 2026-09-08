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
from llmchat import subagent

from conftest import ScriptedClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAST_DEMO = Transport(demo=True, demo_delay=0.0)


def child_env(home: pathlib.Path = None) -> dict:
    """Окружение дочернего процесса, из которого нельзя попасть в живой API.

    Окружение задаётся с нуля, поэтому переменные вида ``*_API_KEY`` в него не
    попадают. Но ``HOME`` без явного значения Python восстанавливает из
    системной записи пользователя, и тогда агент находит настоящий ключ в
    ``~/.openai-key`` и уходит в платный запрос. Домашний каталог поэтому
    подменяется пустым: без него тест не может потратить ни цента.
    """
    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "TERM": "dumb"}
    if home is not None:
        environment["HOME"] = str(home)
    return environment


def one_shot(question: str, config: pathlib.Path) -> dict:
    """Запуск в режиме одного вопроса — тот самый, которым идёт под-агент."""
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(config), "--ask", question, "--json"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env())
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


def test_separate_processes_start_with_empty_memory(fast_config):
    """Каждый запуск в терминале — своя сессия: истории предыдущего не видно."""
    first = one_shot("Привет, меня зовут Влад", fast_config)
    second = one_shot("Как меня зовут?", fast_config)

    assert first["session_id"] != second["session_id"]
    # Одинаковая длина запроса означает, что второй процесс не унаследовал
    # переписку первого: иначе отправленных токенов стало бы заметно больше.
    assert second["prompt_tokens"] < first["prompt_tokens"] + 20
    assert second["usage"]["requests"] == 1


def test_one_shot_prints_nothing_but_json(fast_config):
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--ask", "Привет", "--json"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env())
    payload = json.loads(finished.stdout)          # упадёт, если вывод засорён
    assert finished.stdout.strip().startswith("{")
    assert payload["ok"] and payload["text"]


def test_one_shot_without_json_prints_plain_text(fast_config):
    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(fast_config), "--ask", "Привет"],
        cwd=str(ROOT), capture_output=True, text=True, env=child_env())
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
