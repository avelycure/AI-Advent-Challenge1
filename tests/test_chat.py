"""Терминальный чат как потребитель агента.

Сам чат переписан почти целиком, поэтому проверяется главное: цифры на экране
берутся из агента и совпадают с его счётчиками, а программа целиком доходит от
запуска до выхода, не сломавшись по дороге.
"""
from __future__ import annotations

import pathlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    GenerationParams,
    MCPConfig,
    ToolPolicy,
    Transport,
)
from llmchat.session import DEFAULT_TOPIC, Session
from llmchat.app import handle_invariant

from conftest import ScriptedClient, child_env

ROOT = pathlib.Path(__file__).resolve().parent.parent


def session_with(client) -> Session:
    config = AgentConfig(name="chat", transport=Transport(demo=True, demo_delay=0.0))
    return Session(Agent(config, client=client))


def test_screen_numbers_come_from_the_agent():
    session = session_with(ScriptedClient("ответ"))
    session.add_user("вопрос")
    session.agent.answer_pending()

    agent = session.agent
    assert session.requests == agent.usage.requests
    assert session.total_tokens == agent.usage.total_tokens
    assert session.total_costs == agent.usage.costs
    assert session.context_used() == agent.context_used()
    assert session.free_tokens() == agent.free_tokens()
    assert session.exchanges == 1
    assert session.model.id == "gpt-5.4-mini"


def test_changing_params_on_screen_reconfigures_the_agent():
    session = session_with(ScriptedClient("ответ"))
    session.params = GenerationParams(temperature=0.2, max_tokens=64)
    assert session.agent.config.generation.temperature == 0.2
    assert session.output_reserve == 64


def test_invariants_can_be_managed_from_the_dialogue():
    session = session_with(ScriptedClient("ответ"))

    handle_invariant(
        session,
        "add стек stack.python Сервер только на Python | Командный стандарт")
    rule = session.invariants[0]
    assert (rule.id, rule.category, rule.rule, rule.reason) == (
        "stack.python", "stack", "Сервер только на Python", "Командный стандарт")

    handle_invariant(session, "remove stack.python")
    assert session.invariants == ()


def test_switching_model_keeps_history_and_spending():
    from llmagent.transport import PROVIDERS

    session = session_with(ScriptedClient("ответ"))
    session.add_user("вопрос")
    session.agent.answer_pending()
    spent = session.total_tokens

    session.switch_to(PROVIDERS["openai"], PROVIDERS["openai"].models[2])
    assert session.model.id == "gpt-5.4"
    assert len(session.messages) == 2
    assert session.total_tokens == spent


def test_new_dialogue_clears_history_and_topic_but_not_spending():
    session = session_with(ScriptedClient("ответ"))
    session.add_user("вопрос")
    session.agent.answer_pending()
    session.topic = "Что-то"
    session.reset()
    assert session.messages == [] and session.topic == DEFAULT_TOPIC
    assert session.total_tokens > 0


@pytest.mark.parametrize("script,expect", [
    ("Привет\n/stats\n/config\n/exit\n", "Конфиг агента"),
    ("/help\n/exit\n", "/mcp"),
    ("/mcp\n/exit\n", "MCP отключён"),
    ("/invariant add stack stack.python Сервер только на Python\n"
     "/invariant\n/exit\n", "stack.python"),
])
def test_chat_runs_from_start_to_exit(tmp_path, script, expect):
    """Сквозной прогон настоящей программы: заглушка, ноль трат, ноль вопросов."""
    config = tmp_path / "fast-demo.yaml"
    AgentConfig(name="fast-demo",
                transport=Transport(demo=True, demo_delay=0.0)).to_file(str(config))

    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(config)],
        cwd=str(ROOT), input=script, capture_output=True, text=True,
        env=child_env(tmp_path))
    assert finished.returncode == 0, finished.stderr[-2000:]
    assert expect in finished.stdout
    assert "Traceback" not in finished.stderr


@pytest.mark.parametrize(("online", "expected"), [
    (True, "github_repository_stats"),
    (False, "недоступен"),
])
def test_mcp_command_reports_live_status_without_calling_llm(
        tmp_path, online, expected):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the MCP bridge")

    payload = (
        {
            "ok": True,
            "server": {"name": "test-mcp", "version": "1.0.0"},
            "tools": [{
                "name": "github_repository_stats",
                "description": "test",
                "inputSchema": {"type": "object", "properties": {}},
            }],
        }
        if online else
        {"ok": False, "error": "connection refused sentinel-secret"}
    )
    client = tmp_path / "fake-mcp-client.js"
    client.write_text(
        "process.stdout.write({}); process.exitCode = {};".format(
            json.dumps(json.dumps(payload)), 0 if online else 1),
        encoding="utf-8",
    )
    config = tmp_path / "mcp-demo.yaml"
    AgentConfig(
        name="mcp-demo",
        transport=Transport(demo=True, demo_delay=0.0),
        tools=ToolPolicy(enabled=("github_repository_stats",)),
        mcp=MCPConfig(enabled=True),
    ).to_file(str(config))
    environment = child_env(tmp_path)
    environment["PATH"] = os.path.dirname(node) + os.pathsep + environment["PATH"]
    environment["MCP_CLIENT_SCRIPT"] = str(client)

    finished = subprocess.run(
        [sys.executable, "chat.py", "--config", str(config), "--demo"],
        cwd=str(ROOT),
        input="/mcp\n/exit\n",
        capture_output=True,
        text=True,
        env=environment,
    )

    assert finished.returncode == 0, finished.stderr[-2000:]
    assert expected in finished.stdout
    assert "запросов к API: 0" in finished.stdout
    assert "sentinel-secret" not in finished.stdout
    assert "Traceback" not in finished.stderr
