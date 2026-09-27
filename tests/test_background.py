"""Day 18: фоновый агент делегирует schedule и storage MCP-серверу."""
from __future__ import annotations

import io
import json
import subprocess
import threading
import time

import pytest

from llmagent import (
    AgentConfig,
    AgentError,
    MCPConfig,
    ToolOutcome,
    ToolPolicy,
    Transport,
)
from llmchat import mcp_tools
from llmchat.background import (
    AgentToolInvoker,
    BackgroundOptions,
    BackgroundRunner,
    DEFAULT_TASK_ID,
    DirectToolInvoker,
    SCHEDULE_TOOL,
    SUMMARY_TOOL,
    build_background_agent,
    run_worker,
    summary_line,
)


def listed_tool(name):
    return {
        "name": name,
        "description": name,
        "inputSchema": {"type": "object", "properties": {}},
    }


def config():
    return AgentConfig(
        transport=Transport(demo=True, demo_delay=0.0),
        tools=ToolPolicy(enabled=(SCHEDULE_TOOL, SUMMARY_TOOL), max_calls=1),
        mcp=MCPConfig(enabled=True),
    )


def test_demo_once_calls_stable_schedule_then_uses_report(monkeypatch):
    calls = []
    report = {
        "task": {"id": DEFAULT_TASK_ID},
        "latest": {
            "completedAt": "2026-09-28T12:00:00Z",
            "stats": {
                "repositories": {"total": 7, "public": 5, "private": 2},
                "totals": {"stars": 11, "forks": 3, "openItems": 4},
            },
        },
        "history": [],
        "delta": None,
    }

    def bridge(arguments, stdin=None, **kwargs):
        if "--list" in arguments:
            return {
                "ok": True,
                "server": {"name": "test", "version": "1"},
                "tools": [listed_tool(SCHEDULE_TOOL), listed_tool(SUMMARY_TOOL)],
            }
        name = arguments[arguments.index("--tool") + 1]
        payload = json.loads(stdin)
        calls.append((name, payload))
        result = (
            {"task": {
                "id": payload["taskId"],
                "enabled": True,
                "intervalMinutes": payload["intervalMinutes"],
                "nextRunAt": "2026-09-29T12:00:00Z",
            }}
            if name == SCHEDULE_TOOL else report
        )
        return {"ok": True, "tool": name, "result": result, "content": []}

    monkeypatch.setattr(mcp_tools, "_invoke", bridge)
    options = BackgroundOptions()
    runner = BackgroundRunner(
        AgentToolInvoker(build_background_agent(config(), demo=True)),
        options,
        run_immediately=True,
    )
    output = io.StringIO()

    code = run_worker(runner, threading.Event(), once=True, stream=output)

    assert code == 0
    assert [name for name, _ in calls] == [SCHEDULE_TOOL, SUMMARY_TOOL]
    assert calls[0][1]["taskId"] == DEFAULT_TASK_ID
    assert calls[0][1]["enabled"] is True
    assert calls[0][1]["runImmediately"] is True
    assert "7 репозиториев" in output.getvalue()
    assert "11 звёзд" in output.getvalue()


def test_daemon_upserts_once_and_only_polls_reports_afterwards():
    stop = threading.Event()

    class Runner:
        options = BackgroundOptions(poll_seconds=0.001)
        schedule_ready = False
        schedules = 0
        reports = 0

        def ensure_schedule(self, cancel_event=None):
            del cancel_event
            self.schedules += 1
            self.schedule_ready = True
            return {"task": {"id": DEFAULT_TASK_ID, "intervalMinutes": 1440}}

        def read_summary(self, cancel_event=None):
            del cancel_event
            self.reports += 1
            if self.reports == 2:
                stop.set()
            return {"latest": None}

    runner = Runner()

    assert run_worker(runner, stop, once=False, stream=io.StringIO()) == 0
    assert runner.schedules == 1
    assert runner.reports == 2


def test_direct_daemon_does_not_request_immediate_run(monkeypatch):
    calls = []

    def call_tool(name, arguments, **kwargs):
        calls.append((name, arguments, kwargs))
        result = (
            {"task": {"id": arguments["taskId"], "intervalMinutes": 1440}}
            if name == SCHEDULE_TOOL else {"latest": None, "history": []}
        )
        return ToolOutcome(
            json.dumps(result), detail={"structured_content": result})

    monkeypatch.setattr(mcp_tools, "call_tool", call_tool)
    runner = BackgroundRunner(DirectToolInvoker(), BackgroundOptions())

    assert run_worker(
        runner, threading.Event(), once=True, stream=io.StringIO()) == 0
    assert calls[0][0] == SCHEDULE_TOOL
    assert calls[0][1]["runImmediately"] is False
    assert calls[0][2]["timeout"] == mcp_tools.STATUS_TIMEOUT_SECONDS
    assert isinstance(calls[0][2]["cancel_event"], threading.Event)


def test_retries_use_bounded_exponential_backoff():
    class StopAfterWait:
        def __init__(self):
            self.delays = []

        def is_set(self):
            return len(self.delays) >= 3

        def wait(self, delay):
            self.delays.append(delay)

    class Runner:
        options = BackgroundOptions(poll_seconds=1)
        schedule_ready = True

        def read_summary(self, cancel_event=None):
            del cancel_event
            raise AgentError("temporary MCP failure")

    stop = StopAfterWait()

    assert run_worker(
        Runner(), stop, once=False, stream=io.StringIO()) == 0
    assert stop.delays == [2, 4, 8]


def test_stop_after_schedule_skips_summary():
    stop = threading.Event()

    class Runner:
        options = BackgroundOptions()
        schedule_ready = False

        def ensure_schedule(self, cancel_event=None):
            del cancel_event
            self.schedule_ready = True
            stop.set()
            return {"task": {"id": DEFAULT_TASK_ID, "intervalMinutes": 1440}}

        def read_summary(self, cancel_event=None):
            del cancel_event
            raise AssertionError("summary must not start after stop")

    assert run_worker(
        Runner(), stop, once=False, stream=io.StringIO()) == 0


def test_summary_formats_delta_from_mcp_without_local_storage():
    payload = {
        "latest": {
            "stats": {
                "repositories": {"total": 3, "public": 2, "private": 1},
                "totals": {"stars": 9, "forks": 2, "openItems": 1},
            },
        },
        "delta": {
            "repositories": {"total": 1},
            "totals": {"stars": 2},
        },
    }

    text = summary_line(payload, BackgroundOptions())

    assert "3 репозиториев" in text
    assert "репозитории +1" in text
    assert "звёзды +2" in text


def test_once_fails_visibly_when_immediate_collection_failed():
    class Runner:
        options = BackgroundOptions()
        schedule_ready = False

        def ensure_schedule(self, cancel_event=None):
            del cancel_event
            self.schedule_ready = True
            return {"task": {"id": DEFAULT_TASK_ID, "intervalMinutes": 1440}}

        def read_summary(self, cancel_event=None):
            del cancel_event
            return {
                "latest": None,
                "history": [{"ok": False, "error": "token rejected"}],
                "delta": None,
            }

    output = io.StringIO()

    assert run_worker(
        Runner(), threading.Event(), once=True, stream=output) == 1
    assert "Последний сбор завершился ошибкой" in output.getvalue()
    assert "token rejected" not in output.getvalue()


@pytest.mark.parametrize("changes, message", [
    ({"task_id": "bad id"}, "task id"),
    ({"interval_minutes": 4}, "schedule interval"),
    ({"poll_seconds": 0}, "poll interval"),
])
def test_background_options_reject_unsafe_values(changes, message):
    values = BackgroundOptions().__dict__.copy()
    values.update(changes)

    with pytest.raises(AgentError, match=message):
        BackgroundOptions(**values).validate()


def test_config_must_enable_mcp_scheduler_tools():
    broken = AgentConfig(
        transport=Transport(demo=True, demo_delay=0.0),
        tools=ToolPolicy(enabled=(SUMMARY_TOOL,)),
        mcp=MCPConfig(enabled=True),
    )

    with pytest.raises(AgentError, match=SCHEDULE_TOOL):
        build_background_agent(broken, demo=True)


def test_direct_daemon_cancels_blocking_mcp_process_quickly(
        monkeypatch, tmp_path):
    client = tmp_path / "client.js"
    client.write_text("// blocking fake", encoding="utf-8")
    process_started = threading.Event()
    process_terminated = threading.Event()

    class BlockingProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            del args, kwargs
            process_started.set()

        def communicate(self, input=None, timeout=None):
            del input
            if process_terminated.is_set():
                self.returncode = -15
                return "", ""
            raise subprocess.TimeoutExpired("node", timeout)

        def terminate(self):
            process_terminated.set()

        def kill(self):
            process_terminated.set()

    monkeypatch.setenv(mcp_tools.CLIENT_ENV, str(client))
    monkeypatch.setattr(mcp_tools.subprocess, "Popen", BlockingProcess)
    runner = BackgroundRunner(DirectToolInvoker(timeout=60), BackgroundOptions())
    runner.schedule_ready = True
    stop = threading.Event()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(run_worker(
            runner, stop, once=False, stream=io.StringIO())))

    started = time.monotonic()
    worker.start()
    assert process_started.wait(0.5)
    stop.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert time.monotonic() - started < 1.0
    assert process_terminated.is_set()
    assert result == [0]
