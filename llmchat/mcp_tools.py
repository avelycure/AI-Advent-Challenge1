"""MCP-инструменты как обычные ToolSpec агента.

Агент поддерживает Python 3.8+, а официальный Python MCP SDK требует 3.10+.
Поэтому тонкий мост запускает официальный Node MCP-клиент отдельным процессом.
Секрет GitHub мосту не передаётся: он остаётся только у MCP-сервера.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from llmagent import AgentError, ToolOutcome, ToolSpec

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CLIENT = ROOT.parent / "AI-Advent-MCP" / "mcp1" / "src" / "client.js"
DEFAULT_URL = "http://127.0.0.1:3000/mcp"
CLIENT_ENV = "MCP_CLIENT_SCRIPT"
URL_ENV = "MCP_URL"
TIMEOUT_SECONDS = 60.0
STATUS_TIMEOUT_SECONDS = 5.0
MAX_OUTPUT_CHARS = 1_000_000
MAX_TOOLS = 100
MAX_LABEL_CHARS = 120
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class MCPToolInfo:
    name: str


@dataclass(frozen=True)
class MCPConnectionStatus:
    url: str
    state: str
    name: str = "MCP server"
    version: str = ""
    tools: tuple = ()
    error: str = ""


def _safe_url(raw: str) -> str:
    """Origin для экрана: path, userinfo и query могут содержать секреты."""
    try:
        parsed = urlsplit(raw)
        if not parsed.scheme or not parsed.hostname:
            return "некорректный адрес"
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = "[{}]".format(hostname)
        netloc = hostname
        if parsed.port is not None:
            netloc += ":{}".format(parsed.port)
        return urlunsplit((parsed.scheme, netloc, "", "", ""))
    except (TypeError, ValueError):
        return "некорректный адрес"


def _safe_label(value: object, limit: int = MAX_LABEL_CHARS) -> str:
    """Однострочная terminal-safe подпись из недоверенных MCP metadata."""
    cleaned = "".join(
        character
        if ord(character) >= 32
        and not 127 <= ord(character) <= 159
        and unicodedata.category(character) != "Cf"
        else " "
        for character in str(value)
    )
    compact = " ".join(cleaned.split())
    return compact if len(compact) <= limit else compact[:limit - 1] + "…"


def _safe_connection_error(error: AgentError) -> str:
    message = str(error).lower()
    if "не найден mcp-клиент" in message:
        return "MCP-клиент не найден"
    if "не ответил" in message or "timeout" in message:
        return "MCP-сервер не ответил вовремя"
    if "не вернул json" in message or "неверного типа" in message:
        return "MCP-сервер вернул некорректный ответ"
    return "Не удалось подключиться к MCP-серверу"


def _client_path() -> pathlib.Path:
    configured = os.environ.get(CLIENT_ENV)
    path = pathlib.Path(configured).expanduser() if configured else DEFAULT_CLIENT
    return path.resolve()


def _stop_process(process: subprocess.Popen) -> None:
    """Остановить transport-процесс, не оставляя его после отмены/timeout."""
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.communicate(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.communicate()


def _run_cancellable(command: Sequence[str], stdin: Optional[str], timeout: float,
                     environment: Dict[str, str], cwd: str,
                     cancel_event) -> subprocess.CompletedProcess:
    """Запустить bridge с кооперативной отменой для daemon shutdown."""
    if cancel_event.is_set():
        raise AgentError("MCP-вызов отменён")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            cwd=cwd,
        )
    except OSError as exc:
        raise AgentError("не удалось запустить MCP-клиент: {}".format(exc)) from exc

    started = time.monotonic()
    pending_input = stdin
    while True:
        if cancel_event.is_set():
            _stop_process(process)
            raise AgentError("MCP-вызов отменён")
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            _stop_process(process)
            raise AgentError(
                "MCP-сервер не ответил за {} секунд".format(int(timeout)))
        try:
            stdout, stderr = process.communicate(
                input=pending_input,
                timeout=min(0.1, remaining),
            )
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            # communicate() можно безопасно продолжить после TimeoutExpired,
            # но stdin разрешено передать только при первом вызове.
            pending_input = None


def _invoke(arguments: Sequence[str], timeout: float = TIMEOUT_SECONDS,
            stdin: Optional[str] = None, cancel_event=None) -> Dict[str, object]:
    """Вызвать JSON CLI-мост, не передавая секреты в argv."""
    client = _client_path()
    if not client.is_file():
        raise AgentError(
            "не найден MCP-клиент {}. Укажите путь в {}".format(client, CLIENT_ENV))

    # The transport-only process gets a deliberately tiny environment. API
    # credentials belong either to the MCP server or to the parent agent.
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        URL_ENV: os.environ.get(URL_ENV, DEFAULT_URL),
    }
    command = [
        "node", "--max-old-space-size=64", str(client), *arguments, "--json"]
    if cancel_event is not None:
        finished = _run_cancellable(
            command, stdin, timeout, environment, str(client.parent.parent),
            cancel_event)
    else:
        try:
            finished = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
                cwd=str(client.parent.parent),
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(
                "MCP-сервер не ответил за {} секунд".format(int(timeout))) from exc
        except OSError as exc:
            raise AgentError(
                "не удалось запустить MCP-клиент: {}".format(exc)) from exc

    if len(finished.stdout) > MAX_OUTPUT_CHARS:
        raise AgentError("MCP-клиент вернул слишком большой ответ")

    try:
        payload = json.loads(finished.stdout)
    except ValueError as exc:
        detail = (finished.stderr or finished.stdout or "пустой вывод").strip()
        raise AgentError("MCP-клиент не вернул JSON: {}".format(detail[:200])) from exc

    if not isinstance(payload, dict):
        raise AgentError("MCP-клиент вернул JSON неверного типа")
    if finished.returncode != 0 or not payload.get("ok"):
        error = payload.get("error")
        if not error:
            blocks = payload.get("content") or []
            error = next((
                block.get("text") for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ), None)
        message = str(error or "MCP-вызов завершился ошибкой")
        configured_url = os.environ.get(URL_ENV, DEFAULT_URL)
        message = message.replace(configured_url, _safe_url(configured_url))
        raise AgentError(message)
    return payload


def connection_status(enabled: bool) -> MCPConnectionStatus:
    """Свежий статус настроенного MCP без исключений и раскрытия секретов."""
    url = _safe_url(os.environ.get(URL_ENV, DEFAULT_URL))
    if not enabled:
        return MCPConnectionStatus(url=url, state="disabled")

    try:
        payload = _invoke(["--list"], timeout=STATUS_TIMEOUT_SECONDS)
    except AgentError as exc:
        return MCPConnectionStatus(
            url=url,
            state="unavailable",
            error=_safe_connection_error(exc),
        )

    server = payload.get("server")
    if not isinstance(server, dict):
        server = {}
    listed = payload.get("tools")
    if not isinstance(listed, list):
        return MCPConnectionStatus(
            url=url,
            state="unavailable",
            error="MCP-сервер вернул некорректный список инструментов",
        )
    if len(listed) > MAX_TOOLS:
        return MCPConnectionStatus(
            url=url,
            state="unavailable",
            error="MCP-сервер опубликовал слишком много инструментов",
        )
    tools = tuple(
        MCPToolInfo(name=_safe_label(tool.get("name")))
        for tool in listed
        if isinstance(tool, dict)
        and TOOL_NAME_PATTERN.fullmatch(str(tool.get("name") or ""))
    )
    return MCPConnectionStatus(
        url=url,
        state="connected",
        name=_safe_label(server.get("name") or "MCP server", 80),
        version=_safe_label(server.get("version") or "", 40),
        tools=tools,
    )


def placeholder_specs(names: Iterable[str]) -> List[ToolSpec]:
    """Инструменты конфига, когда каталог недоступен при запуске.

    Обработчик остаётся настоящим: если сервер восстановится, вызов заработает
    без перезапуска. До этого ошибка вернётся модели как ToolOutcome(ok=False).
    """
    return [
        ToolSpec(
            name=name,
            description="MCP tool; server was unavailable during startup.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            handler=_handler(name),
        )
        for name in dict.fromkeys(names)
    ]


def call_tool(name: str, arguments: Dict[str, object],
              timeout: float = TIMEOUT_SECONDS, cancel_event=None) -> ToolOutcome:
    """Вызвать MCP напрямую, когда решение уже принял не LLM, а supervisor."""
    if not TOOL_NAME_PATTERN.fullmatch(name):
        raise AgentError("некорректное имя MCP-инструмента")
    if not isinstance(arguments, dict):
        raise AgentError("аргументы MCP-инструмента должны быть объектом")
    payload = _invoke([
        "--tool", name,
        "--arguments-stdin",
    ], timeout=timeout, stdin=json.dumps(arguments, ensure_ascii=False),
        cancel_event=cancel_event)
    result = payload.get("result")
    if result is None:
        blocks = payload.get("content") or []
        text = "\n".join(
            str(block.get("text", "")) for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        text = json.dumps(result, ensure_ascii=False, sort_keys=True)

    return ToolOutcome(
        text or "MCP-инструмент вернул пустой результат.",
        detail={"mcp_url": _safe_url(os.environ.get(URL_ENV, DEFAULT_URL)),
                "structured_content": result},
    )


def _handler(name: str):
    def call(arguments: Dict[str, object]) -> ToolOutcome:
        return call_tool(name, arguments)

    return call


def specs_for(names: Optional[Iterable[str]] = None,
              timeout: float = TIMEOUT_SECONDS) -> List[ToolSpec]:
    """Получить схемы MCP и превратить каталог сервера в ToolSpec."""
    wanted = None if names is None else list(dict.fromkeys(names))
    if wanted == []:
        return []

    payload = _invoke(["--list"], timeout=timeout)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise AgentError("MCP-сервер вернул неверный список инструментов")
    if len(tools) > MAX_TOOLS:
        raise AgentError("MCP-сервер опубликовал слишком много инструментов")

    by_name = {
        str(tool.get("name")): tool
        for tool in tools
        if isinstance(tool, dict)
        and TOOL_NAME_PATTERN.fullmatch(str(tool.get("name") or ""))
    }
    if wanted is None:
        wanted = list(by_name)
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise AgentError(
            "MCP-сервер не публикует {}; доступны: {}".format(
                ", ".join(missing), ", ".join(sorted(by_name)) or "ни одного"))

    specs = []
    for name in wanted:
        tool = by_name[name]
        parameters = tool.get("inputSchema") or {
            "type": "object", "properties": {}}
        specs.append(ToolSpec(
            name=name,
            description=str(tool.get("description") or name),
            parameters=parameters,
            handler=_handler(name),
        ))
    return specs
