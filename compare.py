#!/usr/bin/env python3
"""Сравнение ответа модели без ограничений и с ограничениями.

Один и тот же вопрос отправляется дважды. Разница только в управлении ответом:
во втором запросе добавлены описание формата, ограничение длины и условие
завершения. Результаты показываются рядом, а соблюдение формата проверяется
разбором ответа, а не на глаз.

Запуск:

    ./compare.sh                       вопрос и формат по умолчанию
    ./compare.sh --format yaml
    ./compare.sh --prompt "..." --format md
    ./compare.sh --demo                без обращения к сети
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmchat.app import setup
from llmchat.client import LLMError
from llmchat.controls import (
    CONSTRAINED_MAX_TOKENS,
    FORMATS,
    ITEM_COUNT,
    MARKER_PATTERN,
    STOP_MARKER,
    Controls,
    Validation,
    validate,
)
from llmchat.ui import make_console, plural

DEFAULT_PROMPT = ("Назови пять языков программирования, год появления каждого "
                  "и чем он примечателен.")
FREE_TEMPERATURE = 0.7
CONTROLLED_TEMPERATURE = 0.2


@dataclass
class Run:
    """Результат одного из двух запросов."""
    title: str
    accent: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    seconds: float
    max_tokens: int
    stop: Optional[List[str]]
    response_format: Optional[dict]
    validation: Validation
    repaired: bool = False

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())


def banner(console: Console) -> None:
    console.clear()
    title = Text("🎛  УПРАВЛЕНИЕ ОТВЕТОМ МОДЕЛИ", style="bold white")
    subtitle = Text(
        "один и тот же запрос отправляется дважды: свободно и с ограничениями\n"
        "формат ответа · ограничение длины · условие завершения",
        style="dim")
    console.print(Panel(Group(Align.center(title), Text(), Align.center(subtitle)),
                        box=box.DOUBLE, border_style="bright_blue", padding=(1, 4)))
    console.print()


def ask_model(client, session_model_ref, messages, max_tokens, temperature,
              stop=None, response_format=None):
    started = time.time()
    completion = client.complete(session_model_ref, messages, max_tokens=max_tokens,
                                 temperature=temperature, stop=stop,
                                 response_format=response_format)
    return completion, time.time() - started


def run_free(console, client, model_ref, prompt, max_tokens, fmt) -> Run:
    with console.status("[bold]запрос без ограничений…[/]", spinner="dots"):
        completion, seconds = ask_model(
            client, model_ref, [{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=FREE_TEMPERATURE)
    return Run("Без ограничений", "yellow", completion.text, completion.prompt_tokens,
               completion.completion_tokens, completion.finish_reason, seconds,
               max_tokens, None, None,
               validate(completion.text, fmt, completion.finish_reason))


def run_controlled(console, client, model_ref, prompt, controls: Controls, repair: bool) -> Run:
    messages = [{"role": "system", "content": controls.instruction},
                {"role": "user", "content": prompt}]
    with console.status("[bold]запрос с ограничениями…[/]", spinner="dots"):
        completion, seconds = ask_model(
            client, model_ref, messages, max_tokens=controls.max_tokens,
            temperature=CONTROLLED_TEMPERATURE, stop=controls.stop,
            response_format=controls.response_format)
    result = validate(completion.text, controls.fmt, completion.finish_reason)
    repaired = False

    # «Нельзя отклоняться от формата»: одна попытка исправления с указанием ошибок.
    if repair and not result.ok:
        failures = result.failure_summary()
        messages += [
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content":
                "Ответ нарушает требования: {}. Исправь именно это и пришли только "
                "данные в заданном формате, ничего больше.".format(failures)},
        ]
        with console.status("[bold]формат нарушен, повторяю с указанием ошибок…[/]",
                            spinner="dots"):
            fixed, extra_seconds = ask_model(
                client, model_ref, messages, max_tokens=controls.max_tokens,
                temperature=CONTROLLED_TEMPERATURE, stop=controls.stop,
                response_format=controls.response_format)
        fixed_result = validate(fixed.text, controls.fmt, fixed.finish_reason)
        if fixed_result.passed_count > result.passed_count:
            completion, result, repaired = fixed, fixed_result, True
            seconds += extra_seconds

    return Run("С ограничениями", "green", completion.text, completion.prompt_tokens,
               completion.completion_tokens, completion.finish_reason, seconds,
               controls.max_tokens, controls.stop, controls.response_format,
               result, repaired)


# --------------------------------------------------------------------------
# Отрисовка сравнения
# --------------------------------------------------------------------------

def answer_panel(run: Run, width: Optional[int], max_lines: int = 12) -> RenderableType:
    lines = run.text.splitlines()
    shown = lines[:max_lines]
    body = Text("\n".join(shown))
    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        body.append("\n… ещё {} {}".format(hidden, plural(hidden, ("строка", "строки", "строк"))),
                    style="dim italic")
    mark = "✓ формат соблюдён" if run.validation.ok else "✗ формат нарушен"
    return Panel(body, title="[bold {}]{}[/]".format(run.accent, run.title),
                 subtitle="[{}]{}[/]".format("green" if run.validation.ok else "red", mark),
                 subtitle_align="right", border_style=run.accent, box=box.ROUNDED,
                 padding=(0, 1), width=width)


def metrics_table(free: Run, controlled: Run) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Показатель", style="dim")
    table.add_column("Без ограничений", style="yellow", justify="right")
    table.add_column("С ограничениями", style="green", justify="right")

    def row(name, left, right):
        table.add_row(name, str(left), str(right))

    row("Символов в ответе", len(free.text), len(controlled.text))
    row("Строк", free.lines, controlled.lines)
    row("Токенов в ответе", free.completion_tokens, controlled.completion_tokens)
    row("Лимит max_tokens", free.max_tokens, controlled.max_tokens)
    row("Стоп-последовательность", "—", ", ".join(controlled.stop or []) or "—")
    row("response_format", "—", (controlled.response_format or {}).get("type", "—"))
    row("Причина завершения", free.finish_reason, controlled.finish_reason)
    row("Маркер остановки в сыром ответе", "—",
        "остался" if MARKER_PATTERN.search(controlled.text) else "обрезан провайдером")
    row("Время ответа", "{:.1f} с".format(free.seconds), "{:.1f} с".format(controlled.seconds))
    row("Проверок пройдено",
        "{} из {}".format(free.validation.passed_count, len(free.validation.checks)),
        "{} из {}".format(controlled.validation.passed_count, len(controlled.validation.checks)))
    return Panel(table, title="Сравнение", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED)


def checks_table(free: Run, controlled: Run) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Проверка", style="dim")
    table.add_column("Без огр.", justify="center")
    table.add_column("С огр.", justify="center")
    table.add_column("Замечание", style="dim")

    def mark(check):
        if check.passed:
            return "[green]✓[/]"
        return "[red]✗[/]" if check.critical else "[yellow]⚠[/]"

    for left, right in zip(free.validation.checks, controlled.validation.checks):
        detail = right.detail if not right.passed else (left.detail if not left.passed else "")
        table.add_row(left.name, mark(left), mark(right), detail[:44])
    return Panel(table, title="Проверка формата разбором ответа", title_align="left",
                 subtitle="[dim]✗ — данные непригодны · ⚠ — снимается при разборе[/]",
                 subtitle_align="right", border_style="magenta", box=box.ROUNDED)


def controls_panel(controls: Controls, prompt: str) -> RenderableType:
    body = Text()
    body.append("Вопрос одинаковый в обоих запросах:\n", style="bold")
    body.append("  {}\n\n".format(prompt))
    body.append("Во втором запросе добавлено:\n", style="bold")
    body.append("  1. Формат — ", style="")
    body.append("{}\n".format(controls.fmt.title), style="bold green")
    body.append("     {}\n".format(controls.fmt.shape.replace("\n", "  ")), style="dim")
    body.append("  2. Ограничение длины — ", style="")
    body.append("max_tokens={}, ровно {} элементов, note ≤ 60 символов\n".format(
        controls.max_tokens, ITEM_COUNT), style="bold green")
    body.append("  3. Условие завершения — ", style="")
    body.append("стоп-последовательность {}\n".format(STOP_MARKER), style="bold green")
    if controls.response_format:
        body.append("  4. Параметр провайдера — ", style="")
        body.append("response_format={}\n".format(controls.response_format["type"]),
                    style="bold green")
    return Panel(body, title="Условия опыта", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def verdict(free: Run, controlled: Run) -> RenderableType:
    text = Text()
    if controlled.validation.ok and not free.validation.ok:
        text.append("Управляемый ответ прошёл все проверки, свободный — нет. ", style="bold green")
        text.append("Ответ пригоден для машинной обработки без разбора текста.", style="dim")
    elif controlled.validation.ok:
        text.append("Оба ответа прошли проверки. ", style="bold yellow")
        text.append("Свободный ответ совпал со схемой случайно — на это нельзя полагаться.",
                    style="dim")
    else:
        text.append("Управляемый ответ не прошёл все проверки. ", style="bold red")
        text.append("Смотрите замечания выше.", style="dim")
    if controlled.validation.remarks:
        text.append("\nЗамечание к оформлению: {}. Снято при разборе, на данные не влияет.".format(
            ", ".join(controlled.validation.remarks)), style="yellow")
    if controlled.repaired:
        text.append("\nПотребовалась одна повторная попытка с указанием ошибок.",
                    style="yellow")
    saved = free.completion_tokens - controlled.completion_tokens
    if saved > 0:
        text.append("\nОтвет короче на {} {} ({:.0f}%).".format(
            saved, plural(saved, ("токен", "токена", "токенов")),
            100.0 * saved / max(1, free.completion_tokens)), style="dim")
    return Panel(text, title="Вывод", title_align="left", border_style="green", box=box.ROUNDED)


def answers_block(console: Console, free: Run, controlled: Run) -> RenderableType:
    if console.width >= 110:
        width = (console.width - 3) // 2
        return Columns([answer_panel(free, width), answer_panel(controlled, width)],
                       equal=True, expand=True)
    return Group(answer_panel(free, None), answer_panel(controlled, None))


def render(console: Console, controls: Controls, prompt: str, free: Run, controlled: Run,
           step: bool = False) -> None:
    """Показать сравнение. В пошаговом режиме — двумя экранами, чтобы всё влезло."""
    first = [controls_panel(controls, prompt), answers_block(console, free, controlled)]
    second = [metrics_table(free, controlled), checks_table(free, controlled),
              verdict(free, controlled)]

    if not step:
        console.clear()
        for block in first + second:
            console.print(block)
        return

    for screen in (first, second):
        console.clear()
        for block in screen:
            console.print(block)
        # После второго экрана тоже ждём: иначе программа завершится мгновенно
        # и прочитать таблицу проверок будет невозможно.
        prompt = ("\n[dim]Enter — показать сравнение и проверки[/] " if screen is first
                  else "\n[dim]Enter — выход[/] ")
        try:
            console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение свободного и управляемого ответа")
    parser.add_argument("--format", choices=sorted(FORMATS), default="json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=CONSTRAINED_MAX_TOKENS)
    parser.add_argument("--no-repair", action="store_true",
                        help="не переспрашивать модель при нарушении формата")
    parser.add_argument("--step", action="store_true",
                        help="показывать результат двумя экранами, по нажатию Enter")
    parser.add_argument("--demo", action="store_true", help="без обращения к сети")
    args = parser.parse_args(argv)

    console = make_console()
    try:
        provider, model, model_ref, client = setup(console, args.demo, banner=banner)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    controls = Controls(fmt=FORMATS[args.format], max_tokens=args.max_tokens)
    console.clear()
    console.print(controls_panel(controls, args.prompt))

    try:
        free = run_free(console, client, model_ref, args.prompt,
                        model.output_reserve, controls.fmt)
        controlled = run_controlled(console, client, model_ref, args.prompt,
                                    controls, repair=not args.no_repair)
    except LLMError as exc:
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                            border_style="red", box=box.ROUNDED))
        return 1

    render(console, controls, args.prompt, free, controlled, step=args.step)
    return 0 if controlled.validation.ok else 2


if __name__ == "__main__":
    sys.exit(main())
