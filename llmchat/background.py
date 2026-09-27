"""Фоновый GitHub-агент поверх расписания, которым владеет MCP-сервер.

Этот модуль намеренно не вычисляет ``nextRunAt`` и ничего не сохраняет на
диск. Агент один раз на старте настраивает задачу через MCP, а затем только
читает готовую сводку. Расписание, запуски GitHub API и JSON-хранилище остаются
в одном месте — внутри MCP-сервера.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO

from llmagent import Agent, AgentConfig, AgentError, HistoryConfig, LLMError, Transport
from llmagent.transport import Completion, ToolCall, count_message_tokens, count_text_tokens

from . import mcp_tools
from .subagent import toolbox_for


SCHEDULE_TOOL = "github_stats_schedule_upsert"
SUMMARY_TOOL = "github_stats_summary"
DEFAULT_TASK_ID = "github-daily-report"
DEFAULT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples" / "github-background.yaml"
)
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_RETRY_SECONDS = 300.0


@dataclass(frozen=True)
class BackgroundOptions:
    task_id: str = DEFAULT_TASK_ID
    interval_minutes: int = 1440
    poll_seconds: float = 60.0
    history_limit: int = 10
    visibility: str = "all"
    affiliations: tuple = ("owner", "collaborator", "organization_member")
    include_forks: bool = True
    include_archived: bool = True
    top: int = 5

    def validate(self) -> None:
        if not TASK_ID_PATTERN.fullmatch(self.task_id):
            raise AgentError(
                "task id: нужны 1–64 латинских букв, цифр или символов _.-")
        if not 5 <= self.interval_minutes <= 10080:
            raise AgentError("schedule interval: допустимо от 5 до 10080 минут")
        if self.poll_seconds <= 0:
            raise AgentError("poll interval: должен быть больше нуля секунд")
        if not 1 <= self.history_limit <= 50:
            raise AgentError("history limit: допустимо от 1 до 50")
        if not 1 <= self.top <= 20:
            raise AgentError("top: допустимо от 1 до 20")

    def schedule_arguments(self, run_immediately: bool) -> Dict[str, object]:
        return {
            "taskId": self.task_id,
            "intervalMinutes": self.interval_minutes,
            "enabled": True,
            "runImmediately": run_immediately,
            "visibility": self.visibility,
            "affiliations": list(self.affiliations),
            "includeForks": self.include_forks,
            "includeArchived": self.include_archived,
            "top": self.top,
        }

    def summary_arguments(self) -> Dict[str, object]:
        return {"taskId": self.task_id, "historyLimit": self.history_limit}


class BackgroundDemoClient:
    """Локальная scripted LLM для безопасной и повторяемой видеодемонстрации.

    MCP и GitHub остаются настоящими. Заглушка заменяет только платную LLM и
    всегда вызывает ровно тот инструмент, который указан worker-ом в вопросе.
    """

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref, messages, max_tokens, temperature=0.7,
                 top_p=None, stop=None, response_format=None, tools=None):
        tool_message = next((
            message for message in reversed(messages)
            if message.get("role") == "tool"
        ), None)
        if tool_message is not None:
            answer = "MCP-инструмент выполнен; результат принят фоновым агентом."
            return Completion(
                answer,
                count_message_tokens(messages),
                count_text_tokens(answer),
                elapsed=0.001,
            )

        request = next(
            (str(message.get("content") or "") for message in reversed(messages)
             if message.get("role") == "user"),
            "",
        )
        marker = "MCP_CALL "
        if not request.startswith(marker):
            answer = "Не получена команда MCP_CALL."
            return Completion(
                answer,
                count_message_tokens(messages),
                count_text_tokens(answer),
                elapsed=0.001,
            )
        try:
            command = json.loads(request.splitlines()[0][len(marker):])
            name = str(command["name"])
            arguments = json.dumps(command["arguments"], ensure_ascii=False)
        except (KeyError, TypeError, ValueError):
            answer = "Команда MCP_CALL не разобрана."
            return Completion(
                answer,
                count_message_tokens(messages),
                count_text_tokens(answer),
                elapsed=0.001,
            )
        available = {
            item["function"]["name"] for item in (tools or [])
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        if name not in available:
            answer = "Инструмент {} не опубликован MCP-сервером.".format(name)
            return Completion(
                answer,
                count_message_tokens(messages),
                count_text_tokens(answer),
                elapsed=0.001,
            )
        return Completion(
            "",
            count_message_tokens(messages),
            1,
            elapsed=0.001,
            tool_calls=[ToolCall("background-{}".format(name), name, arguments)],
        )


def worker_config(config: AgentConfig, demo: bool = False) -> AgentConfig:
    """Проверить конфиг и сделать каждый poll независимым от старой истории."""
    required = {SCHEDULE_TOOL, SUMMARY_TOOL}
    missing = required - set(config.tools.enabled)
    if missing:
        raise AgentError(
            "background config не включает MCP-инструменты: {}".format(
                ", ".join(sorted(missing))))
    if not config.mcp.enabled:
        raise AgentError("background config должен включать mcp.enabled")
    transport = Transport(demo=True, demo_delay=0.0) if demo else config.transport
    return config.with_changes(
        history=HistoryConfig(enabled=False),
        transport=transport,
    )


def build_background_agent(config: AgentConfig, demo: bool = False) -> Agent:
    ready = worker_config(config, demo=demo)
    client = BackgroundDemoClient() if demo else None
    return Agent(ready, client=client, toolbox=toolbox_for(ready))


def _prompt(name: str, arguments: Dict[str, object]) -> str:
    command = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "MCP_CALL {}\n"
        "Вызови ровно указанный MCP-инструмент с указанными аргументами. "
        "Не подменяй его ответ предположениями."
    ).format(command)


def _structured_result(result, expected: str) -> Dict[str, object]:
    calls = [call for call in result.tool_calls if call.get("name") == expected]
    if not calls:
        raise AgentError("агент не вызвал обязательный инструмент {}".format(expected))
    call = calls[-1]
    if not call.get("ok"):
        raise AgentError(str(call.get("text") or "MCP-инструмент завершился ошибкой"))
    detail = call.get("detail")
    payload = detail.get("structured_content") if isinstance(detail, dict) else None
    if not isinstance(payload, dict):
        try:
            payload = json.loads(str(call.get("text") or ""))
        except ValueError as exc:
            raise AgentError(
                "инструмент {} не вернул structured result".format(expected)) from exc
    if not isinstance(payload, dict):
        raise AgentError("инструмент {} вернул результат неверного типа".format(expected))
    return payload


def _outcome_result(outcome, expected: str) -> Dict[str, object]:
    if not outcome.ok:
        raise AgentError(outcome.text)
    payload = outcome.detail.get("structured_content")
    if not isinstance(payload, dict):
        try:
            payload = json.loads(outcome.text)
        except ValueError as exc:
            raise AgentError(
                "инструмент {} не вернул structured result".format(expected)) from exc
    if not isinstance(payload, dict):
        raise AgentError("инструмент {} вернул результат неверного типа".format(expected))
    return payload


class AgentToolInvoker:
    """Доказательный once-flow: решение вызвать tool проходит через Agent."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def invoke(self, name: str, arguments: Dict[str, object],
               cancel_event=None) -> Dict[str, object]:
        del cancel_event
        return _structured_result(self.agent.ask(_prompt(name, arguments)), name)


class DirectToolInvoker:
    """Экономный 24/7-flow: supervisor вызывает известный MCP tool без LLM."""

    def __init__(self, timeout: float = mcp_tools.STATUS_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout

    def invoke(self, name: str, arguments: Dict[str, object],
               cancel_event=None) -> Dict[str, object]:
        outcome = mcp_tools.call_tool(
            name, arguments, timeout=self.timeout, cancel_event=cancel_event)
        return _outcome_result(outcome, name)


class BackgroundRunner:
    """Тонкий supervisor: upsert один раз, затем чтение отчётов MCP."""

    def __init__(self, invoker, options: BackgroundOptions,
                 run_immediately: bool = False) -> None:
        options.validate()
        self.invoker = invoker
        self.options = options
        self.run_immediately = run_immediately
        self.schedule_ready = False

    def ensure_schedule(self, cancel_event=None) -> Dict[str, object]:
        payload = self.invoker.invoke(
            SCHEDULE_TOOL,
            self.options.schedule_arguments(self.run_immediately),
            cancel_event=cancel_event,
        )
        self.schedule_ready = True
        return payload

    def read_summary(self, cancel_event=None) -> Dict[str, object]:
        return self.invoker.invoke(
            SUMMARY_TOOL, self.options.summary_arguments(),
            cancel_event=cancel_event)


def schedule_line(payload: Dict[str, object], options: BackgroundOptions) -> str:
    task = payload.get("task")
    if not isinstance(task, dict):
        task = {}
    return (
        "Расписание {task}: каждые {interval} мин; включено; "
        "следующий запуск {next_run}."
    ).format(
        task=task.get("id") or options.task_id,
        interval=task.get("intervalMinutes") or options.interval_minutes,
        next_run=task.get("nextRunAt") or "определяет MCP-сервер",
    )


def summary_line(payload: Dict[str, object], options: BackgroundOptions) -> str:
    history = payload.get("history")
    newest = history[0] if isinstance(history, list) and history else None
    failed = isinstance(newest, dict) and newest.get("ok") is False
    warning = "Последний сбор завершился ошибкой. " if failed else ""
    latest = payload.get("latest")
    if not isinstance(latest, dict):
        return warning + "Для задачи {} готовых отчётов пока нет.".format(
            options.task_id)
    stats = latest.get("stats") if isinstance(latest.get("stats"), dict) else latest
    repositories = stats.get("repositories") if isinstance(stats, dict) else None
    totals = stats.get("totals") if isinstance(stats, dict) else None
    if not isinstance(repositories, dict) or not isinstance(totals, dict):
        return warning + "Для задачи {} получен отчёт без агрегатов.".format(
            options.task_id)
    text = (
        "GitHub: {total} репозиториев ({public} публичных, {private} приватных), "
        "{stars} звёзд, {forks} форков, {open_items} открытых issue/PR."
    ).format(
        total=repositories.get("total", 0),
        public=repositories.get("public", 0),
        private=repositories.get("private", 0),
        stars=totals.get("stars", 0),
        forks=totals.get("forks", 0),
        open_items=totals.get("openItems", 0),
    )
    delta = payload.get("delta")
    if isinstance(delta, dict):
        repository_delta = delta.get("repositories")
        total_delta = delta.get("totals")
        changes = []
        if isinstance(repository_delta, dict) and repository_delta.get("total"):
            changes.append("репозитории {:+}".format(repository_delta["total"]))
        if isinstance(total_delta, dict) and total_delta.get("stars"):
            changes.append("звёзды {:+}".format(total_delta["stars"]))
        if changes:
            text += " Изменения: {}.".format(", ".join(changes))
    return warning + text


def latest_run_failed(payload: Dict[str, object]) -> bool:
    history = payload.get("history")
    newest = history[0] if isinstance(history, list) and history else None
    return isinstance(newest, dict) and newest.get("ok") is False


def emit(stream: TextIO, event: str, message: str,
         payload: Optional[Dict[str, object]] = None, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps({
            "event": event,
            "message": message,
            "data": payload,
        }, ensure_ascii=False, sort_keys=True), file=stream, flush=True)
    else:
        print("[{}] {}".format(event, message), file=stream, flush=True)


def run_worker(runner: BackgroundRunner, stop: threading.Event, once: bool,
               stream: TextIO = sys.stdout, json_output: bool = False) -> int:
    """Работать до сигнала; ошибки daemon-режима переживаются на следующем poll."""
    failures = 0
    while not stop.is_set():
        try:
            if not runner.schedule_ready:
                schedule = runner.ensure_schedule(cancel_event=stop)
                emit(stream, "schedule", schedule_line(schedule, runner.options),
                     schedule, json_output)
            if stop.is_set():
                break
            summary = runner.read_summary(cancel_event=stop)
            emit(stream, "report", summary_line(summary, runner.options),
                 summary, json_output)
            if once and latest_run_failed(summary):
                return 1
            failures = 0
        except (AgentError, LLMError, ValueError) as exc:
            failures += 1
            emit(stream, "error", str(exc), json_output=json_output)
            if once:
                return 1
        if once:
            return 0
        delay = runner.options.poll_seconds
        if failures:
            delay = min(delay * (2 ** min(failures, 4)), MAX_RETRY_SECONDS)
        stop.wait(delay)
    emit(stream, "stopped", "Фоновый агент остановлен.", json_output=json_output)
    return 0


def install_signal_handlers(stop: threading.Event) -> None:
    def request_stop(signum, frame):
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Фоновый GitHub-агент: расписание и отчёты хранит MCP-сервер, "
            "этот процесс лишь настраивает задачу и наблюдает результат."))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML/JSON конфиг агента")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID,
                        help="стабильный id задачи: повторный запуск обновит её")
    parser.add_argument("--schedule-minutes", type=int, default=1440,
                        help="частота GitHub-задачи в MCP, 5..10080 минут")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="как часто worker читает готовый отчёт; это не scheduler")
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--visibility", choices=("all", "public", "private"),
                        default="all")
    parser.add_argument("--affiliation", action="append",
                        choices=("owner", "collaborator", "organization_member"),
                        dest="affiliations")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--exclude-forks", action="store_true")
    parser.add_argument("--exclude-archived", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="один upsert + один report и выход; удобно для видео")
    parser.add_argument("--demo", action="store_true",
                        help="локальная scripted LLM; MCP и GitHub остаются настоящими")
    parser.add_argument("--json", action="store_true", help="писать JSON Lines")
    return parser.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> BackgroundOptions:
    return BackgroundOptions(
        task_id=args.task_id,
        interval_minutes=args.schedule_minutes,
        poll_seconds=args.poll_seconds,
        history_limit=args.history_limit,
        visibility=args.visibility,
        affiliations=tuple(args.affiliations or (
            "owner", "collaborator", "organization_member")),
        include_forks=not args.exclude_forks,
        include_archived=not args.exclude_archived,
        top=args.top,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        options = options_from_args(args)
        options.validate()
        config = AgentConfig.from_file(args.config)
        ready = worker_config(config, demo=args.demo)
        if args.demo and not args.once:
            raise AgentError("--demo предназначен для одного цикла: добавьте --once")
        if args.demo:
            invoker = AgentToolInvoker(build_background_agent(ready, demo=True))
        else:
            timeout = (
                mcp_tools.TIMEOUT_SECONDS if args.once
                else mcp_tools.STATUS_TIMEOUT_SECONDS
            )
            invoker = DirectToolInvoker(timeout=timeout)
        runner = BackgroundRunner(
            invoker,
            options,
            run_immediately=args.once,
        )
    except AgentError as exc:
        print("background agent: {}".format(exc), file=sys.stderr)
        return 2

    stop = threading.Event()
    install_signal_handlers(stop)
    return run_worker(runner, stop, args.once, json_output=args.json)


__all__ = [
    "BackgroundDemoClient",
    "BackgroundOptions",
    "BackgroundRunner",
    "AgentToolInvoker",
    "DirectToolInvoker",
    "DEFAULT_TASK_ID",
    "SCHEDULE_TOOL",
    "SUMMARY_TOOL",
    "build_background_agent",
    "install_signal_handlers",
    "latest_run_failed",
    "main",
    "options_from_args",
    "parse_args",
    "run_worker",
    "schedule_line",
    "summary_line",
    "worker_config",
]
