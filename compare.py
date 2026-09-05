#!/usr/bin/env python3
"""Пять способов управлять ответом модели, по одному за раз.

Один и тот же вопрос уходит в модель пять раз, каждый раз отдельным запросом без
истории. От шага к шагу добавляется ровно один рычаг, а не сумма предыдущих —
иначе непонятно, что именно подействовало. По каждому шагу видно отправленный
запрос целиком, ответ целиком, измерения и проверку разбором.

Запуск:

    ./compare.sh              подряд, одним экраном
    ./compare.sh --step       по шагам, между шагами ждёт Enter
    ./compare.sh --demo       без обращения к сети, для репетиции сценария
"""
from __future__ import annotations

import argparse
import html.parser
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmchat.app import setup
from llmchat.client import JSON_HINT_NOTE, JSON_INSTRUCTION, LLMError
from llmchat.controls import Check
from llmchat.ui import make_console, plural

QUESTION = "Приведи список наиболее успешных книг Нассима Талеба."

# Одна на все пять шагов: разная температура сделала бы сравнение нечестным.
TEMPERATURE = 0.0

HTML_INSTRUCTION = (
    "Ответ — фрагмент HTML: таблица <table> со столбцами «название», «год», «о чём». "
    "Никакого текста до и после разметки. Не обрамляй ответ тройными кавычками."
)

LIST_REQUEST = (
    "Ответь пронумерованным списком: номер, название, год и одно предложение "
    "о чём книга. Без вступления и без выводов."
)

# Стоп-строки привязаны к началу строки: без переноса «4.» поймало бы и год «2004.»
# внутри предложения. Вариантов три, потому что нумерацию модель оформляет
# по-разному — «4.», «4)» или «**4.». DeepSeek принимает до шестнадцати строк.
STOP_ON_FOURTH = ["\n4.", "\n4)", "\n**4"]

# Договорный маркер. Латиница и знаки процента выбраны потому, что в живом тексте
# такая последовательность не встречается и устойчиво разбивается на токены:
# кириллический «###КОНЕЦ###» модель разбивала переносом строки, и стоп-строка
# переставала срабатывать (см. llmchat/controls.py).
AGREED_MARKER = "%%END%%"
MARKER_REQUEST = (
    "Приведи список наиболее успешных книг Нассима Талеба. "
    "Ответь пронумерованным списком: номер, название, год и одно предложение "
    "о чём книга. После каждой книги выведи {}. Без вступления и без выводов."
).format(AGREED_MARKER)

SHORT_MAX_TOKENS = 120


@dataclass
class Request:
    user: str
    max_tokens: int
    system: Optional[str] = None
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, str]] = None
    # Строка, которую в запрос добавила сама программа, а не сценарий.
    added_system: Optional[str] = None


@dataclass
class Reply:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())

    @property
    def numbered_items(self) -> int:
        count = 0
        for line in self.text.splitlines():
            head = line.strip().lstrip("*").strip()[:2]
            if len(head) == 2 and head[0].isdigit() and head[1] in ".)":
                count += 1
        return count


@dataclass
class Shot:
    label: str
    request: Request
    reply: Reply


@dataclass
class StepResult:
    number: int
    title: str
    lever: str
    shots: List[Shot]
    checks: List[Check] = field(default_factory=list)
    outcome: str = ""
    detail: str = ""
    # Какой из запросов шага представляет его в итоговой таблице.
    headline: int = 0

    @property
    def main(self) -> Shot:
        return self.shots[self.headline]


def banner(console: Console) -> None:
    console.clear()
    title = Text("🎛  УПРАВЛЕНИЕ ОТВЕТОМ МОДЕЛИ", style="bold white")
    subtitle = Text(
        "один вопрос, пять запросов, в каждом ровно один рычаг\n"
        "ничего · формат в промпте · формат через API · условие завершения · длина",
        style="dim")
    console.print(Panel(Group(Align.center(title), Text(), Align.center(subtitle)),
                        box=box.DOUBLE, border_style="bright_blue", padding=(1, 4)))
    console.print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Пять способов управлять ответом модели")
    parser.add_argument("--step", action="store_true",
                        help="показывать по шагам, между шагами ждать Enter")
    parser.add_argument("--ask-keys", action="store_true",
                        help="не подхватывать сохранённые реквизиты, спросить их заново")
    parser.add_argument("--demo", action="store_true", help="без обращения к сети")
    args = parser.parse_args(argv)

    console = make_console()
    try:
        provider, model, model_ref, client = setup(console, args.demo, banner=banner,
                                                   ask_keys=args.ask_keys)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    console.clear()
    console.print(plan_panel(model.output_reserve))
    wait(console, args.step, "Enter — начать опыт")

    results = run_all(console, client, model_ref, model.output_reserve, args.step)
    if not results:
        return 1

    show_summary(console, results)
    wait(console, args.step, "Enter — выход")
    return 0


def run_all(console: Console, client, model_ref: str, reserve: int,
            step: bool) -> List[StepResult]:
    """Пройти пять шагов, показывая каждый сразу после выполнения."""
    steps: List[Callable[[Console, object, str, int], StepResult]] = [
        step_nothing, step_format_in_prompt, step_format_by_api,
        step_completion_condition, step_length_limit,
    ]
    results: List[StepResult] = []
    for run_step in steps:
        try:
            result = run_step(console, client, model_ref, reserve)
        except LLMError as exc:
            console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                                border_style="red", box=box.ROUNDED))
            return results
        results.append(result)
        show_step(console, result, step)
    return results


# --------------------------------------------------------------------------
# Шаги опыта
# --------------------------------------------------------------------------

def step_nothing(console: Console, client, model_ref: str, reserve: int) -> StepResult:
    request = Request(user=QUESTION, max_tokens=reserve)
    reply = ask(console, client, model_ref, request, "запрос без единого ограничения…")
    checks = [Check("Модель закончила сама, а не по лимиту",
                    reply.finish_reason == "stop",
                    "finish_reason = {}".format(reply.finish_reason))]
    return StepResult(1, "Ничего", "рычагов нет — это точка отсчёта",
                      [Shot("Голый вопрос", request, reply)], checks,
                      "{} символов свободного текста".format(len(reply.text)))


def step_format_in_prompt(console: Console, client, model_ref: str, reserve: int) -> StepResult:
    request = Request(user=QUESTION, max_tokens=reserve, system=HTML_INSTRUCTION)
    reply = ask(console, client, model_ref, request, "формат задан инструкцией в промпте…")
    return StepResult(2, "Формат в промпте", "инструкция: фрагмент HTML с таблицей",
                      [Shot("Инструкция плюс тот же вопрос", request, reply)],
                      check_html(reply.text),
                      "структура задана, держится на послушности модели")


def step_format_by_api(console: Console, client, model_ref: str, reserve: int) -> StepResult:
    request = Request(user=QUESTION, max_tokens=reserve,
                      response_format={"type": "json_object"})
    reply = ask(console, client, model_ref, request, "формат задан параметром запроса…")
    if JSON_HINT_NOTE in reply.notes:
        request.added_system = JSON_INSTRUCTION
    checks, keys = check_json(reply.text)
    return StepResult(3, "Формат через API", "response_format = json_object",
                      [Shot("Тот же вопрос, схему не описываем", request, reply)], checks,
                      "синтаксис гарантирован, схему модель выбрала сама",
                      detail="имена полей придумала модель — {}".format(keys))


def step_completion_condition(console: Console, client, model_ref: str,
                              reserve: int) -> StepResult:
    listing = "{} {}".format(QUESTION, LIST_REQUEST)
    free = Request(user=listing, max_tokens=reserve)
    stopped = Request(user=listing, max_tokens=reserve, stop=STOP_ON_FOURTH)
    agreed = Request(user=MARKER_REQUEST, max_tokens=reserve, stop=[AGREED_MARKER])

    shots = [
        Shot("4а — без условия завершения", free,
             ask(console, client, model_ref, free, "список без условия завершения…")),
        Shot("4б — stop на четвёртом пункте", stopped,
             ask(console, client, model_ref, stopped, "тот же промпт, добавлен stop…")),
        Shot("4в — договорный маркер {}".format(AGREED_MARKER), agreed,
             ask(console, client, model_ref, agreed, "маркер, о котором договорились…")),
    ]
    return StepResult(4, "Условие завершения", "stop — обрыв на стороне провайдера",
                      shots, check_stop(shots), describe_stop(shots),
                      detail=describe_marker(shots), headline=1)


def step_length_limit(console: Console, client, model_ref: str, reserve: int) -> StepResult:
    request = Request(user=QUESTION, max_tokens=SHORT_MAX_TOKENS)
    reply = ask(console, client, model_ref, request, "тот же вопрос, но лимит длины…")
    return StepResult(5, "Ограничение длины", "max_tokens = {}".format(SHORT_MAX_TOKENS),
                      [Shot("Голый вопрос с лимитом", request, reply)],
                      check_length(reply),
                      "ответ не короче, а обрублен на полуслове")


def ask(console: Console, client, model_ref: str, request: Request, status: str) -> Reply:
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.user})

    started = time.time()
    with console.status("[bold]{}[/]".format(status), spinner="dots"):
        try:
            completion = client.complete(
                model_ref, messages, max_tokens=request.max_tokens,
                temperature=TEMPERATURE, stop=request.stop,
                response_format=request.response_format)
        except LLMError as exc:
            # Пустой ответ — тоже результат опыта: стоп-строка могла срезать всё.
            return Reply(error=str(exc), seconds=time.time() - started)
    return Reply(completion.text, completion.prompt_tokens, completion.completion_tokens,
                 completion.finish_reason, time.time() - started,
                 list(completion.notes), list(completion.dropped_params))


# --------------------------------------------------------------------------
# Проверки разбором
# --------------------------------------------------------------------------

def check_html(text: str) -> List[Check]:
    raw = text.strip()
    parsed = HtmlShape()
    try:
        parsed.feed(raw)
        parsed.close()
        broken = None
    except Exception as exc:  # noqa: BLE001 — сообщение парсера идёт в отчёт
        broken = str(exc)
    return [
        Check("Разбирается html.parser без ошибок", broken is None, broken or ""),
        Check("Есть тег <table>", parsed.has_table,
              "таблиц: {}".format(parsed.tables)),
        Check("В таблице есть строки", parsed.rows >= 2,
              "строк <tr>: {}".format(parsed.rows)),
        Check("Нет текста вне разметки", not parsed.outside,
              "лишнее: {}".format(parsed.outside[:60]) if parsed.outside else ""),
        Check("Нет обрамления тройными кавычками", "```" not in raw,
              "найдено ```" if "```" in raw else ""),
    ]


def check_json(text: str):
    raw = text.strip()
    try:
        data = json.loads(raw)
        parsed, problem = True, ""
    except ValueError as exc:
        data, parsed, problem = None, False, str(exc)

    keys = "—"
    element_keys = ""
    if isinstance(data, dict) and data:
        keys = ", ".join(list(data)[:4])
        first = data[list(data)[0]]
        if isinstance(first, list) and first and isinstance(first[0], dict):
            element_keys = ", ".join(list(first[0])[:6])
    shape = keys if not element_keys else "{} → {}".format(keys, element_keys)

    checks = [
        Check("json.loads разбирает ответ", parsed, problem),
        Check("Схему никто не задавал — имена полей придумала модель", True,
              "верхний уровень: {}".format(keys)),
    ]
    if element_keys:
        checks.append(Check("Поля элемента списка", True, element_keys))
    return checks, shape


def check_stop(shots: List[Shot]) -> List[Check]:
    free, stopped, agreed = shots[0].reply, shots[1].reply, shots[2].reply
    marker_left = any(part in stopped.text for part in STOP_ON_FOURTH)
    checks = [
        Check("Промпт 4а и 4б совпадает до символа",
              shots[0].request.user == shots[1].request.user,
              "менялся только параметр stop"),
        Check("4б завершился по стоп-строке", stopped.finish_reason == "stop",
              "finish_reason = {}".format(stopped.finish_reason)),
        Check("Стоп-строки нет в ответе — её срезал провайдер", not marker_left),
        Check("Пунктов стало меньше", stopped.numbered_items < free.numbered_items,
              "{} против {}".format(stopped.numbered_items, free.numbered_items)),
        Check("Токенов в ответе меньше", stopped.completion_tokens < free.completion_tokens,
              "{} против {} — в {:.1f} раза".format(
                  stopped.completion_tokens, free.completion_tokens,
                  free.completion_tokens / max(1, stopped.completion_tokens))),
    ]
    if agreed.ok:
        checks.append(Check("Договорный маркер {} до пользователя не дошёл".format(AGREED_MARKER),
                            AGREED_MARKER not in agreed.text,
                            "ответ оборван на первой же книге"))
    return checks


def check_length(reply: Reply) -> List[Check]:
    tail = reply.text.rstrip()[-1:] if reply.text.strip() else ""
    return [
        Check("Завершение по лимиту, а не по воле модели",
              reply.finish_reason == "length",
              "finish_reason = {}".format(reply.finish_reason)),
        Check("Ответ оборван на полуслове", tail not in ".!?»)…",
              "последний символ: {!r}".format(tail)),
    ]


class HtmlShape(html.parser.HTMLParser):
    """Считает теги и текст вне разметки: этого хватает для проверки шага 2."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables = 0
        self.rows = 0
        self.depth = 0
        self.outside = ""

    @property
    def has_table(self) -> bool:
        return self.tables > 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
        if tag == "tr":
            self.rows += 1
        self.depth += 1

    def handle_endtag(self, tag):
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        if self.depth == 0 and data.strip():
            self.outside += data.strip() + " "


# --------------------------------------------------------------------------
# Показ
# --------------------------------------------------------------------------

def show_step(console: Console, result: StepResult, step: bool) -> None:
    for shot in result.shots:
        console.clear()
        console.print(step_header(result, shot.label))
        console.print(request_panel(shot.request))
        for page, panel in enumerate(answer_panels(console, shot)):
            if page:
                wait(console, step, "Enter — продолжение ответа")
                console.clear()
                console.print(step_header(result, shot.label))
            console.print(panel)
        wait(console, step, "Enter — измерения и проверки")

    console.clear()
    console.print(step_header(result, "измерения и проверки"))
    console.print(measures_table(result))
    console.print(checks_panel(result))
    wait(console, step, "Enter — следующий шаг")


def show_summary(console: Console, results: List[StepResult]) -> None:
    console.clear()
    console.print(summary_table(results))
    console.print(verdict_panel(results))


def plan_panel(reserve: int) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Шаг", style="bold", no_wrap=True)
    table.add_column("Рычаг")
    table.add_column("Как задан", style="dim")
    for number, lever, how in (
            ("1", "ничего", "только вопрос, max_tokens = резерв модели ({})".format(reserve)),
            ("2", "формат в промпте", "инструкция: фрагмент HTML с таблицей"),
            ("3", "формат через API", "response_format = json_object, схему не описываем"),
            ("4", "условие завершения", "stop — тот же промпт с параметром и без него"),
            ("5", "ограничение длины", "max_tokens = {}".format(SHORT_MAX_TOKENS))):
        table.add_row(number, lever, how)

    body = Group(
        Text.from_markup("Вопрос один и тот же во всех шагах, каждый уходит "
                         "[bold]отдельным запросом без истории[/]:"),
        Text("  {}".format(QUESTION), style="bold cyan"),
        Text(),
        Text.from_markup("Каждый шаг — тот же вопрос плюс [bold]ровно один[/] рычаг, "
                         "а не сумма предыдущих. Температура везде {}.".format(TEMPERATURE)),
        Text(),
        table)
    return Panel(body, title="Условия опыта", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def step_header(result: StepResult, subtitle: str) -> RenderableType:
    text = Text()
    text.append("Шаг {} из 5 · ".format(result.number), style="dim")
    text.append(result.title, style="bold white")
    text.append("  —  {}".format(result.lever), style="cyan")
    return Panel(text, subtitle="[dim]{}[/]".format(subtitle), subtitle_align="right",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def request_panel(request: Request) -> RenderableType:
    body = Text()
    if request.added_system:
        body.append("system (добавлено программой):\n", style="bold yellow")
        body.append("  {}\n".format(request.added_system), style="yellow")
    if request.system:
        body.append("system:\n", style="bold")
        body.append("  {}\n".format(request.system))
    body.append("user:\n", style="bold")
    body.append("  {}\n".format(request.user))
    body.append("\nПараметры: ", style="bold")
    body.append("max_tokens={}  stop={}  response_format={}".format(
        request.max_tokens,
        json.dumps(request.stop, ensure_ascii=False) if request.stop else "—",
        (request.response_format or {}).get("type", "—")), style="cyan")
    return Panel(body, title="Что отправлено", title_align="left",
                 border_style="yellow", box=box.ROUNDED, padding=(0, 1))


def answer_panels(console: Console, shot: Shot) -> List[RenderableType]:
    reply = shot.reply
    if not reply.ok:
        return [Panel(Text(reply.error, style="red"),
                      title="Модель не ответила — это тоже результат опыта",
                      title_align="left", border_style="red", box=box.ROUNDED)]

    lines = reply.text.splitlines() or [""]
    per_page = max(8, console.size.height - 22)
    pages = [lines[at:at + per_page] for at in range(0, len(lines), per_page)]
    panels = []
    for index, page in enumerate(pages, start=1):
        suffix = "" if len(pages) == 1 else "  ({} из {})".format(index, len(pages))
        panels.append(Panel(Text("\n".join(page)),
                            title="Что вернулось{}".format(suffix), title_align="left",
                            subtitle="[dim]{} {} · {} {}[/]".format(
                                len(reply.text),
                                plural(len(reply.text), ("символ", "символа", "символов")),
                                reply.lines,
                                plural(reply.lines, ("строка", "строки", "строк"))),
                            subtitle_align="right",
                            border_style="green", box=box.ROUNDED, padding=(0, 1)))
    return panels


def measures_table(result: StepResult) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Показатель", style="dim")
    for shot in result.shots:
        table.add_column(shot.label, justify="right")

    def row(name, value):
        table.add_row(name, *[value(shot.reply) for shot in result.shots])

    row("Символов", lambda r: str(len(r.text)))
    row("Строк", lambda r: str(r.lines))
    row("Токенов в ответе", lambda r: str(r.completion_tokens))
    row("Токенов в запросе", lambda r: str(r.prompt_tokens))
    row("Причина завершения", lambda r: r.finish_reason or "—")
    row("Время", lambda r: "{:.1f} с".format(r.seconds))
    notes = [note for shot in result.shots for note in shot.reply.notes]
    if notes:
        table.add_row("Запрос подправлен", *[
            "да" if shot.reply.notes else "—" for shot in result.shots])
    return Panel(table, title="Измерения", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED)


def checks_panel(result: StepResult) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False, show_header=False)
    table.add_column("", justify="center", width=3)
    table.add_column("Проверка")
    table.add_column("Подробности", style="dim")
    for check in result.checks:
        table.add_row("[green]✓[/]" if check.passed else "[red]✗[/]",
                      check.name, check.detail)
    passed = sum(1 for check in result.checks if check.passed)
    return Panel(table, title="Проверка разбором", title_align="left",
                 subtitle="[dim]пройдено {} из {}[/]".format(passed, len(result.checks)),
                 subtitle_align="right", border_style="magenta", box=box.ROUNDED)


def summary_table(results: List[StepResult]) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Шаг", style="bold", no_wrap=True)
    table.add_column("Рычаг", no_wrap=True)
    table.add_column("Симв.", justify="right")
    table.add_column("Токенов", justify="right")
    table.add_column("Завершение", justify="right")
    table.add_column("Проверки", justify="center")
    table.add_column("Что дал рычаг", style="dim")
    for result in results:
        reply = result.main.reply
        passed = sum(1 for check in result.checks if check.passed)
        table.add_row(
            str(result.number), result.title, str(len(reply.text)),
            str(reply.completion_tokens), reply.finish_reason or "—",
            "{}/{}".format(passed, len(result.checks)), result.outcome)
    return Panel(table, title="Пять рычагов рядом", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED)


def verdict_panel(results: List[StepResult]) -> RenderableType:
    by_number = {result.number: result for result in results}
    text = Text()
    text.append("Инструкция в промпте", style="bold")
    text.append(" — единственный рычаг, который задаёт структуру, "
                "но держится на послушности модели.\n", style="dim")

    text.append("response_format=json_object", style="bold")
    third = by_number.get(3)
    text.append(" гарантирует синтаксис, а не схему: {}. Описание схемы "
                "из промпта никуда не девается.\n".format(
                    third.detail if third else "имена полей выбирает модель"), style="dim")

    text.append("stop", style="bold")
    fourth = by_number.get(4)
    text.append(" обрывает генерацию на стороне провайдера при том же промпте: {}. "
                "Но сработает только если модель напечатает эту строку, "
                "а сама строка в ответ не попадёт.{}\n".format(
                    fourth.outcome if fourth else "—",
                    " " + fourth.detail if fourth and fourth.detail else ""), style="dim")

    text.append("max_tokens", style="bold")
    fifth = by_number.get(5)
    tail = ""
    if fifth and fifth.main.reply.text:
        tail = " Хвост ответа: «…{}».".format(
            " ".join(fifth.main.reply.text.split())[-28:])
    text.append(" не сокращает ответ, а режет его на полуслове — это страховка "
                "от расходов, а не способ получить короткий ответ.{}\n".format(tail),
                style="dim")
    return Panel(text, title="Вывод по числам этого прогона", title_align="left",
                 border_style="green", box=box.ROUNDED, padding=(0, 1))


def describe_stop(shots: List[Shot]) -> str:
    free, stopped = shots[0].reply, shots[1].reply
    if not stopped.ok:
        return "стоп-строка срезала весь ответ"
    ratio = free.completion_tokens / max(1, stopped.completion_tokens)
    items = stopped.numbered_items
    return "{} {} вместо {}, токенов меньше в {:.1f} раза".format(
        items, plural(items, ("пункт", "пункта", "пунктов")),
        free.numbered_items, ratio)


def describe_marker(shots: List[Shot]) -> str:
    agreed = shots[2].reply
    if not agreed.ok:
        return "Договорный маркер срезал ответ целиком."
    return ("С договорным маркером {} ответ обрывается на первой же книге — "
            "{} токенов, и самого маркера в тексте нет.").format(
                AGREED_MARKER, agreed.completion_tokens)


def wait(console: Console, step: bool, prompt: str) -> None:
    if not step:
        return
    try:
        console.input("\n[dim]{}[/] ".format(prompt))
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(130)


if __name__ == "__main__":
    sys.exit(main())
