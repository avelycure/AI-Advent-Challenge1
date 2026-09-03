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
    EXPERT_TASK_SUFFIX,
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

# Задача требует перебора вариантов, и ответы получаются длинными. Лимит
# щедрый намеренно: обрезанный ответ нечестно сравнивать с полным, а у
# бесплатных моделей окно вывода измеряется сотнями тысяч токенов.
ANSWER_MAX_TOKENS = 9000
# Промпт, который модель пишет себе, тоже обрезался — 600 токенов не хватало.
SELF_PROMPT_MAX_TOKENS = 1800
JUDGE_MAX_TOKENS = 120
# Повышенная температура намеренно: при 0.2–0.3 модель отвечает почти
# одинаково, и разница между способами постановки промпта не видна.
TEMPERATURE = 0.9


def complete_with_retry(client, model_ref: str, messages: List[dict],
                        max_tokens: int, temperature: float, attempts: int = 2):
    """Один повтор при сбое: бесплатные модели изредка возвращают пустой ответ."""
    last: Optional[LLMError] = None
    for number in range(attempts):
        try:
            return client.complete(model_ref, messages, max_tokens=max_tokens,
                                   temperature=temperature)
        except LLMError as exc:
            last = exc
            if number + 1 < attempts:
                time.sleep(2.0)
    raise last


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
    # Для способа с несколькими независимыми участниками: ответ каждого.
    parts: List[tuple] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.answer is not None and self.answer == DIGIT_TASK.answer

    @property
    def is_multi(self) -> bool:
        return bool(self.parts)

    @property
    def quality(self) -> Optional[float]:
        if len(self.scores) < len(JUDGE_CRITERIA):
            return None
        return sum(self.scores.values()) / len(self.scores)


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def sent_panel(messages: List[dict], accent: str, title: str) -> RenderableType:
    """Показать, что именно уходит в модель на этом шаге."""
    body = Text()
    for message in messages:
        role = {"system": "системное сообщение", "user": "вопрос"}.get(
            message["role"], message["role"])
        body.append("[{}]\n".format(role), style="dim")
        # Показываем запрос целиком: понять опыт без него нельзя.
        body.append(message["content"])
        body.append("\n")
    return Panel(body, title=title, title_align="left",
                 border_style=accent, box=box.ROUNDED, padding=(0, 1))


def run_strategy(console: Console, client, model_ref: str, task: Task,
                 strategy: Strategy, show: bool = True, pause=None,
                 temperature: float = TEMPERATURE) -> Attempt:
    calls = 0
    tokens = 0
    artifact = None
    started = time.time()

    def ask(prompt: str) -> str:
        """Вспомогательный запрос — нужен способу с промптом от самой модели."""
        nonlocal calls, tokens, artifact
        if show:
            console.print(sent_panel([{"role": "user", "content": prompt}],
                                     strategy.accent,
                                     "Сначала просим модель написать промпт"))
        with console.status("[bold]модель пишет промпт…[/]", spinner="dots"):
            completion = complete_with_retry(
                client, model_ref, [{"role": "user", "content": prompt}],
                SELF_PROMPT_MAX_TOKENS, temperature)
        calls += 1
        tokens += completion.prompt_tokens + completion.completion_tokens
        artifact = completion.text
        if show:
            console.print(Panel(Text(completion.text.strip()),
                                title="Промпт, который модель написала себе сама",
                                title_align="left", border_style="magenta",
                                box=box.ROUNDED, padding=(0, 1)))
            # Промпт стоит прочитать: дальше экран займёт решение по нему.
            if pause:
                pause("Enter — решать по этому промпту")
        return completion.text

    if strategy.experts:
        return run_experts(console, client, model_ref, task, strategy, show, started,
                           pause, temperature)

    messages = strategy.build(task, ask)
    if show:
        console.print(sent_panel(messages, strategy.accent, "Отправляем в модель"))
    with console.status("[bold]{} — модель отвечает…[/]".format(strategy.title),
                        spinner="dots"):
        completion = complete_with_retry(client, model_ref, messages,
                                         ANSWER_MAX_TOKENS, temperature)
    calls += 1
    tokens += completion.prompt_tokens + completion.completion_tokens

    return Attempt(strategy, completion.text, extract_answer(completion.text),
                   tokens, time.time() - started, calls, artifact,
                   truncated=(completion.finish_reason == "length"))


def run_experts(console: Console, client, model_ref: str, task: Task,
                strategy: Strategy, show: bool, started: float, pause=None,
                temperature: float = TEMPERATURE) -> Attempt:
    """Каждый эксперт отвечает своим запросом и не видит чужих ответов."""
    pieces: List[str] = []
    parts: List[tuple] = []
    tokens = 0
    truncated = False

    for name, persona in strategy.experts:
        messages = [{"role": "system", "content": persona},
                    {"role": "user", "content": task.question + EXPERT_TASK_SUFFIX}]
        if show:
            console.print(sent_panel(messages, strategy.accent,
                                     "Отдельный запрос: {}".format(name)))
        try:
            with console.status("[bold]{} отвечает…[/]".format(name), spinner="dots"):
                completion = complete_with_retry(client, model_ref, messages,
                                                 ANSWER_MAX_TOKENS, temperature)
        except LLMError as exc:
            # Отказ одного эксперта не должен рушить весь опыт: записываем
            # его как не ответившего и идём дальше.
            parts.append((name, None))
            pieces.append("### {}\n[не ответил: {}]".format(name, exc))
            if show:
                console.print(Panel(Text(str(exc), style="red"),
                                    title="[bold red]{} не ответил[/]".format(name),
                                    border_style="red", box=box.ROUNDED, padding=(0, 1)))
                if pause:
                    pause("Enter — следующий эксперт")
            continue
        tokens += completion.prompt_tokens + completion.completion_tokens
        truncated = truncated or completion.finish_reason == "length"
        value = extract_answer(completion.text)
        parts.append((name, value))
        pieces.append("### {}\n{}".format(name, completion.text.strip()))
        if show:
            console.print(Panel(Markdown(completion.text.strip()),
                                title="[bold {}]{}[/]".format(strategy.accent, name),
                                subtitle="[green]✓ {}[/]".format(value)
                                if value == task.answer
                                else "[red]✗ {}[/]".format(value if value is not None
                                                           else "нет числа"),
                                subtitle_align="right", border_style=strategy.accent,
                                box=box.ROUNDED, padding=(0, 1)))
            # Ответы длинные: без остановки предыдущий эксперт уедет за край.
            if pause and (name, None) != (strategy.experts[-1][0], None):
                pause("Enter — следующий эксперт")
            elif pause:
                pause("Enter — итог по трём экспертам")

    # Итог группы — то, на чём сошлось большинство. Единого ответа у независимых
    # экспертов нет, и выбирать чей-то один было бы произволом.
    values = [value for _, value in parts if value is not None]
    answer = None
    if values:
        answer = max(set(values), key=values.count)
        if values.count(answer) < 2:
            answer = None      # согласия нет
    return Attempt(strategy, "\n\n".join(pieces), answer, tokens,
                   time.time() - started, len(strategy.experts), None,
                   truncated=truncated, parts=parts,
                   extracted_by="большинством голосов")


def extract_final(console: Console, client, model_ref: str, attempt: Attempt) -> None:
    """Достать итоговое число из ответа отдельным запросом.

    Разбор регулярками по свободному тексту принципиально хрупок: модели
    заканчивают ответ чем угодно, и последнее число в тексте часто оказывается
    промежуточным. Поэтому итог достаёт модель, а регулярка остаётся запасным
    путём на случай отказа.
    """
    # Берём хвост, а не начало: итог модель называет в конце, а ответы длинные,
    # и первые тысячи символов — это середина рассуждения.
    prompt = EXTRACT_PROMPT.format(text=attempt.text[-5000:])
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
    # Оценщику даём начало и конец: по одной середине о полноте не судят.
    body = attempt.text
    if len(body) > 6000:
        body = body[:3000] + "\n\n[…середина пропущена…]\n\n" + body[-3000:]
    prompt = JUDGE_PROMPT.format(question=task.question, answer=task.answer, text=body)
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


def task_panel(task: Task, judge_name: str, temperature: float) -> RenderableType:
    body = Text()
    body.append("Задача\n", style="bold")
    body.append("  {}\n\n".format(task.question))
    body.append("Верный ответ: ", style="bold")
    body.append("{}".format(task.answer), style="bold green")
    body.append("  — посчитан кодом: ", style="dim")
    body.append(task.verify, style="dim cyan")
    for value, note in task.traps:
        body.append("\nЛовушка ", style="bold")
        body.append("{}".format(value), style="yellow")
        body.append(" — {}".format(note), style="dim")
    body.append("\n\nКачество объяснения оценивает: ", style="bold")
    body.append(judge_name, style="magenta")
    body.append("\nТемпература ответов: ", style="bold")
    body.append("{}".format(temperature), style="yellow")
    body.append(" — намеренно высокая, чтобы точность промпта была заметнее",
                style="dim")
    return Panel(body, title="Условия опыта", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def attempt_panel(attempt: Attempt, max_lines: int = 38) -> RenderableType:
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


def step_header(number: int, total: int, strategy: Strategy) -> RenderableType:
    body = Text()
    body.append("Шаг {} из {}: ".format(number, total), style="dim")
    body.append(strategy.title, style="bold {}".format(strategy.accent))
    body.append("\n{}".format(strategy.description), style="dim")
    return Panel(body, box=box.ROUNDED, border_style=strategy.accent, padding=(0, 1))


def result_line(attempt: Attempt) -> RenderableType:
    text = Text()
    if attempt.parts:
        text.append("Ответы экспертов: ", style="bold")
        for index, (name, value) in enumerate(attempt.parts):
            if index:
                text.append("   ·   ", style="dim")
            text.append("{} — ".format(name), style="dim")
            text.append(str(value) if value is not None else "нет числа",
                        style="green" if value == DIGIT_TASK.answer else "red")
        text.append("\n")
    text.append("Итог этого способа: ", style="bold")
    text.append(str(attempt.answer) if attempt.answer is not None else "не назван",
                style="bold green" if attempt.correct else "bold red")
    text.append(" — {}".format("верно" if attempt.correct else "неверно"),
                style="green" if attempt.correct else "red")
    if attempt.parts:
        text.append("  (по большинству)", style="dim")
    quality = attempt.quality
    if quality is not None:
        text.append("   ·   оценка объяснения: ", style="bold")
        text.append("{:.1f} из 5".format(quality), style="magenta")
        text.append("  ({})".format(", ".join(
            "{} {}".format(name, attempt.scores[name])
            for name, _ in JUDGE_CRITERIA if name in attempt.scores)), style="dim")
    if attempt.truncated:
        text.append("\nОтвет упёрся в лимит длины — итог мог не успеть прозвучать.",
                    style="yellow")
    return Panel(text, box=box.ROUNDED, border_style="grey42", padding=(0, 1))


CLEAR_NOTE = ("🧹 История очищена: следующий способ спрашивает с чистого листа. "
              "Каждый запрос независим — модель не видит прошлых ответов.")


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
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="случайность ответов: выше — сильнее влияние промпта")
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
    console.print(task_panel(task, judge_name, args.temperature))
    if not step(console, args.step, "Enter — начать первый способ"):
        return 0

    attempts: List[Attempt] = []
    total = len(STRATEGIES)
    try:
        for run in range(1, max(1, args.runs) + 1):
            # Подробно показываем первый прогон: повторы нужны ради свода,
            # и показывать их так же подробно значило бы утомить зрителя.
            live = run == 1
            if not live:
                console.print("\n[dim]Прогон {} из {} — повтор для свода…[/]".format(
                    run, args.runs))
            for number, strategy in enumerate(STRATEGIES, start=1):
                if live:
                    console.clear()
                    console.print(task_panel(task, judge_name, args.temperature))
                    console.print(step_header(number, total, strategy))
                stepping = args.step and live

                def hold(prompt: str, _console=console) -> None:
                    if stepping and not step(_console, True, prompt):
                        raise KeyboardInterrupt

                attempt = run_strategy(console, client, model_ref, task, strategy,
                                       show=live, pause=hold if stepping else None,
                                       temperature=args.temperature)
                # Итог достаём до отрисовки: иначе на панели останется число,
                # найденное регуляркой, и оно разойдётся со строкой итога.
                if not attempt.is_multi:
                    extract_final(console, judge_client, judge_ref, attempt)
                if live and not attempt.is_multi:
                    console.print(attempt_panel(attempt))
                judge(console, judge_client, judge_ref, task, attempt)
                attempts.append(attempt)
                if live:
                    console.print(result_line(attempt))
                    if number < total:
                        console.print(Text(CLEAR_NOTE, style="dim cyan"))
                    if not step(console, args.step,
                                "Enter — следующий способ" if number < total
                                else "Enter — сравнение"):
                        return 0
    except KeyboardInterrupt:
        console.print("\n[dim]Показ прерван.[/]")
        return 0
    except LLMError as exc:
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                            border_style="red", box=box.ROUNDED))
        return 1

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
