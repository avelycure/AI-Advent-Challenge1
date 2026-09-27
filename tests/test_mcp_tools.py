"""MCP-инструмент проходит полный цикл и используется агентом."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    AgentError,
    ConfigError,
    MCPConfig,
    Toolbox,
    ToolPolicy,
    Transport,
)
from llmagent.transport import Completion, ToolCall, count_message_tokens, count_text_tokens
from llmchat import mcp_tools
from llmchat.subagent import toolbox_for

TOOL_NAME = "github_repository_stats"
FAST_DEMO = Transport(demo=True, demo_delay=0.0)


def listed_tool():
    return {
        "name": TOOL_NAME,
        "description": "Collect aggregate GitHub repository statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "visibility": {
                    "type": "string",
                    "enum": ["all", "public", "private"],
                    "description": "Which visibility to include.",
                },
                "top": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Size of the top list.",
                },
            },
        },
    }


def stats():
    return {
        "repositories": {"total": 7, "public": 5, "private": 2},
        "totals": {"stars": 11, "forks": 3, "openItems": 4, "sizeKb": 100},
        "languages": {"Python": 4, "JavaScript": 3},
        "topRepositories": [],
    }


def fake_bridge(arguments, **kwargs):
    if "--list" in arguments:
        return {
            "ok": True,
            "server": {"name": "github-statistics-server", "version": "1.0.0"},
            "tools": [listed_tool()],
        }
    return {"ok": True, "tool": TOOL_NAME, "result": stats(), "content": []}


def test_mcp_schema_becomes_agent_tool_spec(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_invoke", fake_bridge)

    spec = mcp_tools.specs_for([TOOL_NAME])[0]

    assert spec.name == TOOL_NAME
    assert spec.parameters["properties"]["visibility"]["enum"] == [
        "all", "public", "private"]
    outcome = spec.handler({"visibility": "all", "top": 3})
    assert outcome.ok
    assert json.loads(outcome.text)["repositories"]["total"] == 7


def test_bridge_does_not_forward_github_token(monkeypatch, tmp_path):
    client = tmp_path / "client.js"
    client.write_text("// test bridge", encoding="utf-8")
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(
            stdout='{"ok":true,"tools":[]}', stderr="", returncode=0)

    monkeypatch.setenv("GITHUB_TOKEN", "must-stay-on-server")
    monkeypatch.setenv(mcp_tools.CLIENT_ENV, str(client))
    monkeypatch.setattr(mcp_tools.subprocess, "run", run)

    assert mcp_tools._invoke(["--list"])["ok"] is True
    assert "GITHUB_TOKEN" not in seen


def test_mcp_status_lists_server_and_tools(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_invoke", fake_bridge)

    status = mcp_tools.connection_status(enabled=True)

    assert status.state == "connected"
    assert status.name == "github-statistics-server"
    assert status.version == "1.0.0"
    assert [tool.name for tool in status.tools] == [TOOL_NAME]


def test_mcp_status_reports_disabled_without_calling_bridge(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("disabled MCP must not start a client")

    monkeypatch.setattr(mcp_tools, "_invoke", should_not_run)

    status = mcp_tools.connection_status(enabled=False)

    assert status.state == "disabled"
    assert status.tools == ()


def test_mcp_status_reports_connection_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AgentError("fetch failed")

    monkeypatch.setattr(mcp_tools, "_invoke", unavailable)

    status = mcp_tools.connection_status(enabled=True)

    assert status.state == "unavailable"
    assert status.error == "Не удалось подключиться к MCP-серверу"


def test_mcp_status_hides_credentials_and_query(monkeypatch):
    monkeypatch.setenv(
        mcp_tools.URL_ENV,
        "https://user:password@example.test/mcp?token=secret#private",
    )
    monkeypatch.setattr(mcp_tools, "_invoke", fake_bridge)

    status = mcp_tools.connection_status(enabled=True)

    assert status.url == "https://example.test"
    assert "password" not in status.url
    assert "secret" not in status.url


def test_mcp_status_removes_terminal_direction_controls(monkeypatch):
    def dangerous_server(arguments, **kwargs):
        return {
            "ok": True,
            "server": {"name": "safe\u202ereversed", "version": "1\u2066.0"},
            "tools": [listed_tool()],
        }

    monkeypatch.setattr(mcp_tools, "_invoke", dangerous_server)

    status = mcp_tools.connection_status(enabled=True)

    assert "\u202e" not in status.name
    assert "\u2066" not in status.version


def test_tool_arguments_are_sent_through_stdin(monkeypatch, tmp_path):
    client = tmp_path / "client.js"
    client.write_text("// fake", encoding="utf-8")
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen["stdin"] = kwargs.get("input")
        return SimpleNamespace(
            stdout=json.dumps({"ok": True, "result": {"done": True}}),
            stderr="",
            returncode=0,
        )

    monkeypatch.setenv(mcp_tools.CLIENT_ENV, str(client))
    monkeypatch.setattr(mcp_tools.subprocess, "run", run)
    spec = mcp_tools.placeholder_specs([TOOL_NAME])[0]

    outcome = spec.handler({"secret": "not-in-process-list"})

    assert outcome.ok is True
    assert "not-in-process-list" not in " ".join(seen["command"])
    assert json.loads(seen["stdin"]) == {"secret": "not-in-process-list"}


def test_offline_mcp_keeps_configured_tool_available(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AgentError("fetch failed")

    monkeypatch.setattr(mcp_tools, "_invoke", unavailable)
    config = AgentConfig(
        transport=FAST_DEMO,
        tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2),
        mcp=MCPConfig(enabled=True),
    )

    agent = Agent(config, toolbox=toolbox_for(config), client=object())

    assert agent.config.tools.enabled == (TOOL_NAME,)


def test_startup_discovery_uses_short_timeout(monkeypatch):
    seen = []

    def unavailable(arguments, timeout):
        seen.append(timeout)
        raise AgentError("timeout")

    monkeypatch.setattr(mcp_tools, "_invoke", unavailable)
    config = AgentConfig(
        transport=FAST_DEMO,
        tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2),
        mcp=MCPConfig(enabled=True),
    )

    Agent(config, toolbox=toolbox_for(config), client=object())

    assert seen == [mcp_tools.STATUS_TIMEOUT_SECONDS]


def test_bridge_error_does_not_expose_configured_url(monkeypatch, tmp_path):
    client = tmp_path / "client.js"
    client.write_text("// fake", encoding="utf-8")
    secret_url = "http://user:password@example.test/private-token?key=query-secret"

    def run(*args, **kwargs):
        return SimpleNamespace(
            stdout=json.dumps({"ok": False, "error": "failed at " + secret_url}),
            stderr="",
            returncode=1,
        )

    monkeypatch.setenv(mcp_tools.CLIENT_ENV, str(client))
    monkeypatch.setenv(mcp_tools.URL_ENV, secret_url)
    monkeypatch.setattr(mcp_tools.subprocess, "run", run)

    toolbox = Toolbox(mcp_tools.placeholder_specs([TOOL_NAME]))
    outcome = toolbox.run(TOOL_NAME, "{}")

    assert outcome.ok is False
    assert "password" not in outcome.text
    assert "private-token" not in outcome.text
    assert "query-secret" not in outcome.text


def test_agent_calls_mcp_tool_and_uses_its_result(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_invoke", fake_bridge)

    class ResultUsingClient:
        def validate_key(self):
            return None

        def complete(self, model_ref, messages, max_tokens, temperature=0.7,
                     top_p=None, stop=None, response_format=None, tools=None):
            has_tool_result = any(
                message.get("role") == "tool" for message in messages)
            if tools and not has_tool_result:
                assert tools[0]["function"]["name"] == TOOL_NAME
                return Completion(
                    "", count_message_tokens(messages), 5, elapsed=0.001,
                    tool_calls=[ToolCall("call-1", TOOL_NAME,
                                         '{"visibility":"all","top":3}')],
                )

            tool_message = next(
                message for message in reversed(messages)
                if message.get("role") == "tool")
            result = json.loads(tool_message["content"])
            answer = "Получено {} репозиториев и {} звёзд.".format(
                result["repositories"]["total"], result["totals"]["stars"])
            return Completion(
                answer, count_message_tokens(messages), count_text_tokens(answer),
                elapsed=0.001)

    config = AgentConfig(
        transport=FAST_DEMO,
        tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2),
        mcp=MCPConfig(enabled=True),
    )
    agent = Agent(config, client=ResultUsingClient(), toolbox=toolbox_for(config))

    result = agent.ask("Сколько у меня репозиториев?")

    assert result.text == "Получено 7 репозиториев и 11 звёзд."
    assert result.tool_calls[0]["name"] == TOOL_NAME
    assert result.tool_calls[0]["ok"] is True
    assert len(result.tool_calls) == 1


def test_mcp_catalog_supports_enabling_tool_after_agent_creation(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_invoke", fake_bridge)
    config = AgentConfig(transport=FAST_DEMO, mcp=MCPConfig(enabled=True))
    agent = Agent(config, toolbox=toolbox_for(config), client=object())

    updated = agent.reconfigure(
        tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2))

    assert updated.tools.enabled == (TOOL_NAME,)


def test_mcp_connection_is_startup_only():
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=object())

    with pytest.raises(ConfigError, match="при создании агента"):
        agent.reconfigure(mcp=MCPConfig(enabled=True))

    assert agent.config.mcp.enabled is False
