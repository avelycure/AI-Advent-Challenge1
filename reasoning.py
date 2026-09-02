#!/usr/bin/env python3
"""Одна задача, четыре способа спросить — и сравнение результатов.

Задача решается через API четырьмя способами: прямым вопросом, с просьбой
решать пошагово, по промпту, который модель написала себе сама, и от лица
группы экспертов. Сравнивается и верность итога, и качество объяснения.

Верность считает код: у задачи есть заранее вычисленный ответ. Качество
оценивает **другая модель** по трём признакам — своя оценка своему же ответу
была бы слабым доводом.

Запуск:

    ./reasoning.sh                                   ответы и оценка одной моделью
    ./reasoning.sh --judge gigachat                  оценивает другой провайдер
    ./reasoning.sh --runs 3                          несколько прогонов
    ./reasoning.sh --demo                            без обращения к сети
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmchat.app import setup
from llmchat.client import LLMError, make_client
from llmchat.providers import PROVIDERS
from llmchat.secrets import find_sources
from llmchat.strategies import (
    DIGIT_TASK,
    EXTRACT_PROMPT,
    JUDGE_CRITERIA,
    JUDGE_PROMPT,
    STRATEGIES,
    Strategy,
    Task,
    extract_answer,
    parse_extracted,
    parse_scores,
)
from llmchat.ui import fmt, make_console, plural

ANSWER_MAX_TOKENS = 2600
JUDGE_MAX_TOKENS = 120
TEMPERATURE = 0.3


@dataclass
class Attempt:
    strategy: Strategy
    text: str
    answer: Optional[int]
    tokens: int
    seconds: float
    calls: int
    artifact: Optional[str] = None
    scores: Dict[str, int] = field(default_factory=dict)
    # Ответ упёрся в лимит длины: сравнивать его с дошедшими до конца нечестно,
    # поэтому обрыв показывается отдельно.
    truncated: bool = False
    judge_note: str = ""
    extracted_by: str = "разбором текста"

    @property
    def correct(self) -> bool:
        return self.answer is not None and self.answer == DIGIT_TASK.answer

    @property
    def quality(self) -> Optional[float]:
        if len(self.scores) < len(JUDGE_CRITERIA):
            return None
        return sum(self.scores.values()) / len(self.scores)


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def run_strategy(console: Console, client, model_ref: str, task: Task,
                 strategy: Strategy) -> Attempt:
    calls = 0
    tokens = 0
    artifact = None
    started = time.time()

    def ask(prompt: str) -> str:
        """Вспомогательный запрос — нужен способу с промптом от самой модели."""
        nonlocal calls, tokens, artifact
        completion = client.complete(model_ref, [{"role": "user", "content": prompt}],
                                     max_tokens=600, temperature=TEMPERATURE)
        calls += 1
        tokens += completion.prompt_tokens + completion.completion_tokens
        artifact = completion.text
        return completion.text

    with console.status("[bold]{}…[/]".format(strategy.title), spinner="dots"):
        messages = strategy.build(task, ask)
        completion = client.complete(model_ref, messages,
                                     max_tokens=ANSWER_MAX_TOKENS, temperature=TEMPERATURE)
    calls += 1
    tokens += completion.prompt_tokens + completion.completion_tokens

    return Attempt(strategy, completion.text, extract_answer(completion.text),
                   tokens, time.time() - started, calls, artifact,
                   truncated=(completion.finish_reason == "length"))


def extract_final(console: Console, client, model_ref: str, attempt: Attempt) -> None:
    """Достать итоговое число из ответа отдельным запросом.

    Разбор регулярками по свободному тексту принципиально хрупок: модели
    заканчивают ответ чем угодно, и последнее число в тексте часто оказывается
    промежуточным. Поэтому итог достаёт модель, а регулярка остаётся запасным
    путём на случай отказа.
    """
    prompt = EXTRACT_PROMPT.format(text=attempt.text[:4000])
    try:
        with console.status("[dim]извлекаю итог: {}…[/]".format(attempt.strategy.title),
                            spinner="dots"):
            completion = client.complete(model_ref, [{"role": "user", "content": prompt}],
                                         max_tokens=24, temperature=0.0)
        # Ответ модели-извлекателя главнее регулярки, в том числе когда она
        # говорит «итога нет»: пустое поле честнее случайного числа из текста.
        attempt.answer = parse_extracted(completion.text)
        attempt.extracted_by = "моделью"
    except LLMError as exc:
        attempt.extracted_by = "разбором текста (извлекатель недоступен)"
        attempt.judge_note = str(exc)[:50]


def judge(console: Console, client, model_ref: str, task: Task,
          attempt: Attempt) -> None:
    prompt = JUDGE_PROMPT.format(question=task.question, answer=task.answer,
                                 text=attempt.text[:4000])
    # Одна повторная попытка: оценщик иногда отвечает не в том виде,
    # и терять из-за этого целую строку сравнения не хочется.
    for attempt_number in (1, 2):
        try:
            with console.status("[dim]оцениваю: {}…[/]".format(attempt.strategy.title),
                                spinner="dots"):
                completion = client.complete(
                    model_ref, [{"role": "user", "content": prompt}],
                    max_tokens=JUDGE_MAX_TOKENS + 80 * attempt_number, temperature=0.0)
        except LLMError as exc:
            attempt.judge_note = str(exc)[:60]
            return
        attempt.scores = parse_scores(completion.text)
        if len(attempt.scores) == len(JUDGE_CRITERIA):
            return
        attempt.judge_note = "оценщик ответил не по форме"
    # Оценка — украшение: без неё опыт остаётся годным.


def build_judge(console: Console, key: str, demo: bool):
    """Собрать клиент-оценщик у другого провайдера, взяв ключ из файла."""
    provider = PROVIDERS.get(key)
    if provider is None:
        console.print("[red]Неизвестный провайдер для оценки: {}[/]".format(key))
        return None, None
    sources = find_sources(provider.api_key_env, provider.key_files)
    if not sources and not demo:
        console.print("[yellow]Нет сохранённого ключа для {} — оценка будет своей же "
                      "моделью.[/]".format(provider.name))
        return None, None
    secret = sources[0].value if sources else "demo"
    extra = None
    if provider.extra_field is not None:
        found = find_sources(None, provider.extra_field.files)
        if not found:
            console.print("[yellow]Нет каталога для {} — оценка будет своей же "
                          "моделью.[/]".format(provider.name))
            return None, None
        extra = found[0].value
    client = make_client(provider, secret, demo)
    return client, provider.model_ref(provider.default_model, extra)


# --------------------------------------------------------------------------
# Отрисовка
# --------------------------------------------------------------------------

def banner(console: Console) -> None:
    console.clear()
    title = Text("🧠  ЧЕТЫРЕ СПОСОБА СПРОСИТЬ", style="bold white")
    subtitle = Text("одна задача, четыре постановки промпта\n"
                    "сравниваем верность итога и качество объяснения", style="dim")
    console.print(Panel(Group(Align.center(title), Text(), Align.center(subtitle)),
                        box=box.DOUBLE, border_style="bright_blue", padding=(1, 4)))
    console.print()


def task_panel(task: Task, judge_name: str) -> RenderableType:
    body = Text()
    body.append("Задача\n", style="bold")
    body.append("  {}\n\n".format(task.question))
    body.append("Верный ответ: ", style="bold")
    body.append("{}".format(task.answer), style="bold green")
    body.append("  — посчитан кодом: ", style="dim")
    body.append(task.verify, style="dim cyan")
    if task.trap:
        body.append("\nРядом ловушка: ", style="bold")
        body.append("{} ".format(task.trap), style="yellow")
        body.append("— {}".format(task.trap_note), style="dim")
    body.append("\n\nКачество объяснения оценивает: ", style="bold")
    body.append(judge_name, style="magenta")
    return Panel(body, title="Условия опыта", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def attempt_panel(attempt: Attempt, max_lines: int = 22) -> RenderableType:
    lines = attempt.text.strip().splitlines()
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += "\n…"
    mark = ("[green]✓ {}[/]".format(attempt.answer) if attempt.correct
            else "[red]✗ {}[/]".format(attempt.answer if attempt.answer is not None else "нет числа"))
    if attempt.truncated:
        mark = "[yellow]обрыв по лимиту[/] · " + mark
    return Panel(Markdown(shown), title="[bold {}]{}[/]".format(
        attempt.strategy.accent, attempt.strategy.title),
        subtitle=mark, subtitle_align="right",
        border_style=attempt.strategy.accent, box=box.ROUNDED, padding=(0, 1))


@dataclass
class Summary:
    """Свод по способу за все прогоны — по одному судить нельзя."""
    strategy: Strategy
    runs: int = 0
    correct: int = 0
    answers: List[str] = field(default_factory=list)
    quality: List[float] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0
    truncated: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.runs if self.runs else 0.0

    @property
    def mean_quality(self) -> Optional[float]:
        return sum(self.quality) / len(self.quality) if self.quality else None


def summarise(attempts: List[Attempt]) -> List[Summary]:
    order = [s.key for s in STRATEGIES]
    by_key: Dict[str, Summary] = {}
    for attempt in attempts:
        summary = by_key.setdefault(attempt.strategy.key, Summary(attempt.strategy))
        summary.runs += 1
        summary.correct += int(attempt.correct)
        summary.answers.append(str(attempt.answer) if attempt.answer is not None else "—")
        if attempt.quality is not None:
            summary.quality.append(attempt.quality)
        summary.tokens += attempt.tokens
        summary.seconds += attempt.seconds
        summary.truncated += int(attempt.truncated)
    return [by_key[k] for k in order if k in by_key]


def summary_table(summaries: List[Summary], task: Task, runs: int) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Способ", style="bold")
    table.add_column("Верных", justify="center")
    table.add_column("Ответы по прогонам", justify="center", style="dim")
    table.add_column("Качество", justify="right")
    table.add_column("Токенов", justify="right", style="dim")
    table.add_column("Время", justify="right", style="dim")

    for item in summaries:
        share = "[green]{} из {}[/]" if item.correct else "[red]{} из {}[/]"
        quality = item.mean_quality
        table.add_row(
            item.strategy.title,
            share.format(item.correct, item.runs),
            ", ".join(item.answers),
            "{:.1f}".format(quality) if quality else "—",
            fmt(item.tokens),
            "{:.0f} с".format(item.seconds / item.runs),
        )
    return Panel(table, title="Свод по {} {}".format(
                     runs, plural(runs, ("прогону", "прогонам", "прогонам"))), title_align="left",
                 subtitle="[dim]верный ответ — {}[/]".format(task.answer),
                 subtitle_align="right", border_style="magenta", box=box.ROUNDED)


def comparison_table(attempts: List[Attempt], task: Task) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Способ", style="bold")
    table.add_column("Ответ", justify="right")
    table.add_column("Верно", justify="center")
    table.add_column("Дошёл\nдо конца", justify="center")
    for name, _ in JUDGE_CRITERIA:
        table.add_column(name.capitalize(), justify="center")
    table.add_column("Качество", justify="right")
    table.add_column("Токенов", justify="right", style="dim")
    table.add_column("Время", justify="right", style="dim")

    for attempt in attempts:
        row = [attempt.strategy.title,
               str(attempt.answer) if attempt.answer is not None else "—",
               "[green]да[/]" if attempt.correct else "[red]нет[/]",
               "[yellow]обрыв[/]" if attempt.truncated else "да"]
        for name, _ in JUDGE_CRITERIA:
            value = attempt.scores.get(name)
            row.append(str(value) if value else "—")
        quality = attempt.quality
        row.append("{:.1f}".format(quality) if quality else "—")
        row.append(fmt(attempt.tokens))
        row.append("{:.0f} с".format(attempt.seconds))
        table.add_row(*row)

    notes = sorted({a.extracted_by for a in attempts})
    footer = "итог извлечён: " + ", ".join(notes)

    return Panel(table, title="Сравнение", title_align="left",
                 subtitle="[dim]верный ответ — {} · {}[/]".format(task.answer, footer),
                 subtitle_align="right", border_style="bright_blue", box=box.ROUNDED)


def verdict(attempts: List[Attempt], summaries: List[Summary]) -> RenderableType:
    if len(attempts) > len(summaries):
        return verdict_many(summaries)
    correct = [a for a in attempts if a.correct]
    scored = [a for a in attempts if a.quality is not None]
    text = Text()

    if correct:
        text.append("Верный итог дали: ", style="bold")
        text.append(", ".join(a.strategy.title for a in correct), style="green")
    else:
        text.append("Верный итог не дал ни один способ.", style="bold red")
    text.append("\n")

    if scored:
        best = max(scored, key=lambda a: a.quality)
        text.append("Лучшее объяснение: ", style="bold")
        text.append("{} ({:.1f} из 5)".format(best.strategy.title, best.quality),
                    style="magenta")
        text.append("\n")

    both = [a for a in correct if a.quality is not None]
    if both:
        winner = max(both, key=lambda a: a.quality)
        text.append("Лучший по совокупности: ", style="bold")
        text.append(winner.strategy.title, style="bold green")
        text.append(" — верный ответ и оценка {:.1f}.".format(winner.quality), style="dim")

    cut = [a for a in attempts if a.truncated]
    if cut:
        text.append("\nОтвет обрезан по лимиту у: {} — их итог мог не успеть "
                    "прозвучать.".format(", ".join(a.strategy.title for a in cut)),
                    style="yellow")

    cheapest = min(attempts, key=lambda a: a.tokens)
    text.append("\nДешевле всех — «{}»: {} {}.".format(
        cheapest.strategy.title, fmt(cheapest.tokens),
        plural(cheapest.tokens, ("токен", "токена", "токенов"))), style="dim")
    return Panel(text, title="Вывод", title_align="left",
                 border_style="green", box=box.ROUNDED)


def verdict_many(summaries: List[Summary]) -> RenderableType:
    """Вывод по нескольким прогонам: один прогон ничего не доказывает."""
    text = Text()
    best_accuracy = max(summaries, key=lambda s: (s.accuracy, s.mean_quality or 0))
    text.append("Чаще всех попадал в верный ответ: ", style="bold")
    text.append("{} — {} из {}".format(best_accuracy.strategy.title,
                                       best_accuracy.correct, best_accuracy.runs),
                style="green" if best_accuracy.correct else "red")

    scored = [s for s in summaries if s.mean_quality is not None]
    if scored:
        best_quality = max(scored, key=lambda s: s.mean_quality)
        text.append("\nЛучшее объяснение в среднем: ", style="bold")
        text.append("{} ({:.1f} из 5)".format(best_quality.strategy.title,
                                              best_quality.mean_quality), style="magenta")

    total_correct = sum(s.correct for s in summaries)
    total_runs = sum(s.runs for s in summaries)
    text.append("\n\nВерных ответов всего: {} из {}.".format(total_correct, total_runs),
                style="dim")
    if total_correct == 0:
        text.append(" Задача оказалась моделью не по зубам — сравнивать стоит "
                    "только качество разбора.", style="dim")
    cheapest = min(summaries, key=lambda s: s.tokens)
    text.append("\nДешевле всех — «{}»: {} {} на {} {}.".format(
        cheapest.strategy.title, fmt(cheapest.tokens),
        plural(cheapest.tokens, ("токен", "токена", "токенов")),
        cheapest.runs, plural(cheapest.runs, ("прогон", "прогона", "прогонов"))),
        style="dim")
    return Panel(text, title="Вывод", title_align="left",
                 border_style="green", box=box.ROUNDED)


def artifact_panel(attempts: List[Attempt]) -> Optional[RenderableType]:
    for attempt in attempts:
        if attempt.artifact:
            return Panel(Text(attempt.artifact.strip()[:900]),
                         title="Промпт, который модель написала себе сама",
                         title_align="left", border_style="magenta",
                         box=box.ROUNDED, padding=(0, 1))
    return None


# --------------------------------------------------------------------------
def step(console: Console, enabled: bool, prompt: str) -> bool:
    """Пауза до нажатия Enter. Возвращает False, если пользователь прервал показ."""
    if not enabled:
        return True
    try:
        console.input("\n[dim]{}[/] ".format(prompt))
        return True
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Четыре способа решить одну задачу")
    parser.add_argument("--judge", default="", metavar="ПРОВАЙДЕР",
                        help="кто оценивает качество: ключ провайдера, например gigachat")
    parser.add_argument("--runs", type=int, default=1,
                        help="сколько раз повторить каждый способ")
    parser.add_argument("--ask-keys", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--step", action="store_true",
                        help="показывать ответы и сравнение раздельно, по Enter")
    args = parser.parse_args(argv)

    console = make_console()
    try:
        provider, model, model_ref, client = setup(console, args.demo, banner=banner,
                                                   ask_keys=args.ask_keys)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    judge_client, judge_ref = (None, None)
    judge_name = "{} (та же модель)".format(model.id)
    if args.judge:
        judge_client, judge_ref = build_judge(console, args.judge, args.demo)
        if judge_client is not None:
            judge_name = "{} — другой провайдер".format(PROVIDERS[args.judge].name)
    if judge_client is None:
        judge_client, judge_ref = client, model_ref

    task = DIGIT_TASK
    console.clear()
    console.print(task_panel(task, judge_name))

    attempts: List[Attempt] = []
    try:
        for _ in range(max(1, args.runs)):
            for strategy in STRATEGIES:
                attempt = run_strategy(console, client, model_ref, task, strategy)
                extract_final(console, judge_client, judge_ref, attempt)
                judge(console, judge_client, judge_ref, task, attempt)
                attempts.append(attempt)
    except LLMError as exc:
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                            border_style="red", box=box.ROUNDED))
        return 1

    console.clear()
    console.print(task_panel(task, judge_name))
    extra = artifact_panel(attempts)
    if extra is not None:
        console.print(extra)
    if not step(console, args.step, "Enter — показать ответы по очереди"):
        return 0

    shown = attempts if args.runs == 1 else attempts[:len(STRATEGIES)]
    for index, attempt in enumerate(shown, start=1):
        if args.step:
            console.clear()
        console.print(attempt_panel(attempt))
        last = index == len(shown)
        hint = "Enter — сравнение" if last else "Enter — следующий способ"
        if not step(console, args.step, hint):
            return 0
    if args.step:
        console.clear()

    # При нескольких прогонах построчная таблица разрастается и мешает:
    # показываем свод, ради которого повторы и делались.
    if args.runs > 1:
        console.print(summary_table(summarise(attempts), task, args.runs))
    else:
        console.print(comparison_table(attempts, task))
    console.print(verdict(attempts, summarise(attempts)))
    # В пошаговом режиме ждём: иначе программа закроется мгновенно
    # и прочитать вывод будет невозможно.
    step(console, args.step, "Enter — выход")
    return 0


if __name__ == "__main__":
    sys.exit(main())
