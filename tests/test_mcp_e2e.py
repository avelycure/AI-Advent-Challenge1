"""Настоящий transport E2E без доступа к реальному GitHub."""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llmagent import Agent, AgentConfig, MCPConfig, ToolPolicy, Transport
from llmchat import mcp_tools
from llmchat.subagent import toolbox_for
from mcp_demo import StatisticsUsingClient, TOOL_NAME


class FakeGitHubHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — имя задаёт BaseHTTPRequestHandler
        repositories = [
            {
                "full_name": "demo/public",
                "private": False,
                "fork": False,
                "archived": False,
                "language": "Python",
                "stargazers_count": 8,
                "forks_count": 2,
                "open_issues_count": 1,
                "size": 50,
                "html_url": "https://github.com/demo/public",
                "updated_at": "2026-09-28T00:00:00Z",
            },
            {
                "full_name": "demo/private",
                "private": True,
                "fork": False,
                "archived": False,
                "language": "JavaScript",
                "stargazers_count": 0,
                "forks_count": 0,
                "open_issues_count": 0,
                "size": 25,
                "html_url": "https://github.com/demo/private",
                "updated_at": "2026-09-27T00:00:00Z",
            },
        ]
        payload = json.dumps(repositories).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-RateLimit-Limit", "5000")
        self.send_header("X-RateLimit-Remaining", "4999")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_, *args):
        return None


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_port(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail("MCP server stopped before accepting connections")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail("MCP server did not start in 10 seconds")


def test_agent_uses_result_from_real_mcp_http_process(monkeypatch):
    client_path = pathlib.Path(
        os.environ.get(mcp_tools.CLIENT_ENV, str(mcp_tools.DEFAULT_CLIENT))
    ).resolve()
    server_path = client_path.with_name("server.js")
    project = client_path.parent.parent
    if not server_path.is_file() or not (project / "node_modules").is_dir():
        if os.environ.get("MCP_E2E_REQUIRED") == "1":
            pytest.fail("required sibling MCP project is not installed")
        pytest.skip("sibling MCP project is not installed")

    github = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHubHandler)
    github_thread = threading.Thread(target=github.serve_forever, daemon=True)
    github_thread.start()
    mcp_port = free_port()
    environment = dict(os.environ)
    environment.update({
        "GITHUB_TOKEN": "e2e-test-token",
        "GITHUB_API_URL": "http://127.0.0.1:{}".format(github.server_port),
        "MCP_PORT": str(mcp_port),
    })
    process = subprocess.Popen(
        ["node", str(server_path)],
        cwd=str(project),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_port(process, mcp_port)
        monkeypatch.setenv(mcp_tools.CLIENT_ENV, str(client_path))
        monkeypatch.setenv(
            mcp_tools.URL_ENV,
            "http://127.0.0.1:{}/mcp".format(mcp_port),
        )
        config = AgentConfig(
            transport=Transport(demo=True, demo_delay=0.0),
            tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2),
            mcp=MCPConfig(enabled=True),
        )
        agent = Agent(
            config,
            client=StatisticsUsingClient(),
            toolbox=toolbox_for(config),
        )

        result = agent.ask("Собери статистику GitHub.")

        assert "2 репозиториев" in result.text
        assert "1 публичных и 1 приватных" in result.text
        assert "Всего звёзд: 8" in result.text
        assert [(call["name"], call["ok"]) for call in result.tool_calls] == [
            (TOOL_NAME, True),
        ]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        github.shutdown()
        github.server_close()
