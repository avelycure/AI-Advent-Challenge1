"""Вызов под-агента отдельным процессом.

Под-агент запускается той же командой, что и сам чат, — в режиме одного
вопроса с выводом в JSON. Так изоляция получается не на обещании, а на
устройстве: у дочернего процесса своя память, свои счётчики и свой конфиг,
а падение под-агента родителя не касается.

Наружу отсюда торчит ровно две вещи: ``run`` запускает под-агента и
возвращает разобранный итог, ``panel`` показывает этот итог человеку.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmagent import AgentConfig, ConfigError, overrides
from llmagent.registry import EXTRA_ENV

from .ui import fmt, format_cost, format_seconds

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
ENTRY_POINT = ROOT / "chat.py"
TIMEOUT_SECONDS = 300.0


@dataclass
class Delegation:
    """Итог работы под-агента, каким его увидел родитель."""

    name: str
    question: str
    text: str = ""
    error: Optional[str] = None
    session_id: str = ""
    provider: str = ""
    model: str = ""
    attempts: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost: Optional[float] = None
    currency: str = "USD"
    checks: List[Dict[str, object]] = field(default_factory=list)
    # Прошёл ли ответ выходную политику под-агента. Это не то же, что удался
    # ли вызов: под-агент мог отработать штатно и вернуть ответ не той формы.
    valid: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------
# Поиск конфига
# --------------------------------------------------------------------------

def available() -> List[str]:
    """Короткие имена конфигов из configs/ — то, что можно указать команде."""
    return sorted(path.stem for path in CONFIGS_DIR.glob("*.y*ml"))


def resolve_config(name: str) -> pathlib.Path:
    """Короткое имя из configs/ либо путь к файлу как он задан."""
    for candidate in (CONFIGS_DIR / (name + ".yaml"), CONFIGS_DIR / (name + ".yml"),
                      pathlib.Path(name).expanduser()):
        if candidate.is_file():
            return candidate
    raise ConfigError("нет конфига «{}». Доступны: {}".format(
        name, ", ".join(available()) or "ни одного"))


# --------------------------------------------------------------------------
# Запуск
# --------------------------------------------------------------------------

def child_environment(parent: AgentConfig, child: AgentConfig) -> Dict[str, str]:
    """Окружение дочернего процесса.

    Реквизиты передаются переменными окружения, а не аргументами командной
    строки: argv виден в выводе ``ps`` любому процессу пользователя, а ключ
    провайдера там появляться не должен. Передаём только если провайдер тот
    же — иначе под-агент найдёт свой ключ сам.
    """
    environment = dict(os.environ)
    if parent.provider != child.provider:
        return environment
    if parent.api_key:
        environment[parent.provider_info.api_key_env] = parent.api_key
    if parent.api_extra:
        environment[EXTRA_ENV] = parent.api_extra
    return environment


# Одиночные два дефиса — там, где до них и после них пробел или край строки.
# Внутри вопроса такая последовательность попадаться может, и разделителем
# считается только первая: «/agent frugal -- вопрос с -- внутри» не должен
# терять хвост вопроса.
SEPARATOR = re.compile(r"(?:^|\s)--(?:\s|$)")


def split_request(argument: str) -> Tuple[str, List[str], str]:
    """Разобрать «имя правка=значение … -- вопрос».

    Без ``--`` всё после имени остаётся вопросом: привычное поведение важнее
    возможности не набирать два знака.
    """
    name, _, tail = argument.strip().partition(" ")
    marker = SEPARATOR.search(tail)
    if marker is None:
        return name, [], tail.strip()

    tweaks = tail[:marker.start()].split()
    if any("=" not in item for item in tweaks):
        # Похоже, что «--» просто попались в тексте вопроса: правки выглядят
        # как поле=значение, и если хоть одна не выглядит — это не правки.
        return name, [], tail.strip()
    return name, tweaks, tail[marker.end():].strip()


def run(name: str, question: str, parent: AgentConfig,
        tweaks: Optional[List[str]] = None) -> Delegation:
    """Запустить под-агента и вернуть его итог.

    Ни одно исключение наружу не летит: сорванный под-агент — это результат
    с заполненным ``error``, и родительская сессия продолжает работать.
    """
    try:
        path = resolve_config(name)
        child = AgentConfig.from_file(str(path))
        # Правки проверяем здесь, а не в дочернем процессе: понятную ошибку
        # про опечатку в поле надо показать сразу, а не через запуск и разбор.
        if tweaks:
            child = overrides.apply(child, tweaks)
    except ConfigError as exc:
        return Delegation(name, question, error=str(exc))

    command = [sys.executable, str(ENTRY_POINT), "--config", str(path),
               "--ask", question, "--json"]
    for tweak in tweaks or []:
        command += ["--set", tweak]
    # Родитель в демонстрационном режиме — значит и под-агент тоже: иначе
    # проверочный прогон внезапно ушёл бы в сеть и потратил деньги.
    if parent.transport.demo and not child.transport.demo:
        command.append("--demo")

    try:
        finished = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=TIMEOUT_SECONDS,
                                  env=child_environment(parent, child))
    except subprocess.TimeoutExpired:
        return Delegation(name, question,
                          error="под-агент не ответил за {:.0f} с".format(TIMEOUT_SECONDS))
    except OSError as exc:
        return Delegation(name, question, error="не удалось запустить: {}".format(exc))

    return parse(name, question, finished)


def parse(name: str, question: str, finished: "subprocess.CompletedProcess") -> Delegation:
    """Разобрать вывод дочернего процесса. Он обязан быть одним объектом JSON."""
    try:
        payload = json.loads(finished.stdout)
    except ValueError:
        complaint = (finished.stderr or finished.stdout or "пустой вывод").strip()
        return Delegation(name, question,
                          error="под-агент не вернул JSON: {}".format(complaint[:200]))

    # Признак неудачи — заполненное поле error, а не ok: ok говорит о форме
    # ответа, и под-агент, не соблюдший формат, всё равно отработал.
    if payload.get("error"):
        return Delegation(name, question, error=str(payload["error"]),
                          session_id=payload.get("session_id", ""))

    return Delegation(
        name=name, question=question, text=payload.get("text", ""),
        valid=bool(payload.get("ok", True)),
        session_id=payload.get("session_id", ""),
        provider=payload.get("provider", ""), model=payload.get("model", ""),
        attempts=int(payload.get("attempts", 1)),
        prompt_tokens=int(payload.get("prompt_tokens", 0)),
        completion_tokens=int(payload.get("completion_tokens", 0)),
        seconds=float(payload.get("seconds", 0.0)),
        cost=payload.get("cost"), currency=payload.get("currency", "USD"),
        checks=list(payload.get("checks", [])),
    )


# --------------------------------------------------------------------------
# Показ
# --------------------------------------------------------------------------

def panel(delegation: Delegation) -> RenderableType:
    if not delegation.ok:
        return Panel(
            Text(delegation.error or "", style="red"),
            title="⤷ Под-агент [bold]{}[/] не справился".format(delegation.name),
            title_align="left", border_style="red", box=box.ROUNDED, padding=(0, 1))

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("Сессия", "{} · своя память, отдельный процесс".format(
        delegation.session_id or "—"))
    facts.add_row("Модель", "{} · {}".format(delegation.provider, delegation.model))
    if delegation.attempts > 1:
        facts.add_row("Попыток", str(delegation.attempts))
    failed = [c for c in delegation.checks if not c.get("passed")]
    if failed:
        facts.add_row("Форма", "[yellow]не сошлась: {}[/]".format(
            ", ".join(str(c.get("name")) for c in failed[:2])))
    facts.add_row("Расход", "{} · {} · {}".format(
        format_seconds(delegation.seconds),
        "{} ток.".format(fmt(delegation.total_tokens)),
        format_cost(delegation.cost, delegation.currency)))

    body = Text(delegation.text.strip() or "(пусто)")
    hint = Text.from_markup(
        "\n[dim]Ответ принят в память этой сессии — на него можно ссылаться "
        "следующим вопросом.[/]")
    accent = "green" if delegation.valid else "yellow"
    return Panel(Group(facts, Text(""), body, hint),
                 title="⤷ Под-агент [bold]{}[/]".format(delegation.name),
                 title_align="left", border_style=accent, box=box.ROUNDED, padding=(0, 1))


def catalog_panel() -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False, expand=True)
    table.add_column("Конфиг", style="bold", no_wrap=True)
    table.add_column("Модель", no_wrap=True)
    table.add_column("Чем отличается")

    for name in available():
        try:
            config = AgentConfig.from_file(str(resolve_config(name)))
        except ConfigError:
            continue
        traits = []
        if config.output.format != "free":
            traits.append("формат " + config.output.format)
        if config.judge is not None:
            traits.append("судья")
        if config.budget.max_cost is not None:
            traits.append("бюджет")
        if not config.history.enabled:
            traits.append("без истории")
        if config.transport.demo:
            traits.append("заглушка")
        table.add_row(name, config.model, ", ".join(traits) or "значения по умолчанию")

    hint = Text.from_markup(
        "\n[cyan]/agent books-json Книги Нассима Талеба[/][dim] — запустить под-агента[/]\n"
        "[cyan]/agent frugal temperature=0.1 max_tokens=80 -- Что такое хороший код[/]\n"
        "[dim]До[/] [bold]--[/][dim] правки конфига, после — вопрос.\n"
        "Под-агент идёт отдельным процессом: своя сессия, своя память, "
        "свои счётчики. Его ответ вернётся сюда и попадёт в память этой сессии.[/]")
    return Panel(Group(table, hint), title="⤷ Под-агенты", title_align="left",
                 border_style="cyan", box=box.ROUNDED, padding=(0, 1))
