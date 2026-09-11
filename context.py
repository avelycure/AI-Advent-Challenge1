#!/usr/bin/env python3
"""Один и тот же сбор ТЗ четырьмя стратегиями, и отдельно — ветвление.

Разговор один и тот же: четырнадцать сообщений, в которых по кускам
складывается задание на мобильное приложение. Разное только то, что уходит
в модель на каждом шаге:

* **full** — вся переписка целиком, пока помещается;
* **window** — последние N сообщений, остальное просто не отправляется;
* **facts** — картотека «ключ — значение» плюс последние N сообщений;
* **summary** — пересказ начала плюс последние N сообщений.

Проверяются четыре вещи, которые по отдельности ничего не доказывают: во
что обошёлся разговор, помнит ли агент сказанное в начале, дошёл ли он до
конца и сколько сверх ответов за это заплачено служебными запросами. Забыть
всё дёшево, помнить всё дорого — интересна только середина.

Ветвление показывается отдельно: разговор доходит до выбора стека, от одной
точки отводятся две ветки с разными решениями, и в каждой задаётся один и тот
же вопрос. Ответы должны разойтись — это и есть доказательство, что ветки
независимы.

Окно намеренно сужено до нескольких тысяч токенов. У настоящих моделей оно на
сто тысяч и больше, и упираться в него значило бы потратить деньги ради того
же самого вывода.

Запуск:

    ./context.sh              на заглушке, бесплатно
    ./context.sh --live       настоящие запросы к выбранной модели
    ./context.sh --step       по шагам, между шагами ждёт Enter
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmagent import (
    Agent,
    AgentConfig,
    AgentError,
    CompressionConfig,
    ConfigError,
    ContextOverflow,
    FactsConfig,
    HistoryConfig,
    LLMError,
    MissingCredentials,
    Step,
    Transport,
    find_model,
)
from llmagent.transport import PROVIDER_ORDER, PROVIDERS, tokenizer_name
from llmagent.usage import FACTS, SUMMARY
from llmchat.app import agent_from_config, setup
from llmchat.switching import CredentialStore
from llmchat.ui import format_cost, fmt, make_console, plural

# Сбор ТЗ: сведения, которые названы по ходу и спрашиваются в конце. Каждое
# сказано один раз и больше не повторяется — именно так они и теряются.
BRIEF = [
    ("Собираем ТЗ на мобильное приложение для доставки еды. "
     "Платформа одна — Android, под iOS не делаем.", None),
    ("Бюджет проекта — 300 тысяч рублей, больше не будет.", None),
    ("Срок сдачи — к первому марта, дата жёсткая.", None),
    ("Решили: регистрация только по номеру телефона, вход через соцсети не делаем.", None),
    ("Важное предпочтение: интерфейс тёмный по умолчанию, светлую тему не делаем.", None),
]

# Вопросы между сведениями: обычный разговор, в котором сведения тонут.
QUESTIONS = [
    "Как обычно устроена оплата в приложениях доставки?",
    "Что должно быть на экране со списком ресторанов?",
    "Как показывать статус заказа, чтобы человек не звонил в поддержку?",
    "Нужен ли офлайн-режим в таком приложении?",
    "Как обрабатывать отмену заказа со стороны ресторана?",
    "Что писать в пуш-уведомлениях, чтобы их не отключили?",
    "Как устроить поиск по блюдам, а не только по ресторанам?",
    "Какие метрики собирать в первой версии?",
]

RECALLS = (
    ("На какую платформу мы делаем приложение? Ответь одним словом.", "android"),
    ("Какой у нас бюджет? Ответь числом.", "300"),
    ("К какой дате нужно сдать проект? Ответь коротко.", "март"),
    ("Как мы решили делать вход в приложение? Ответь одной фразой.", "телефон"),
)

# Ветвление: развилка и два несовместимых решения от одного места.
FORK_POINT = "Пора выбрать стек. Предложи варианты коротко."
BRANCHES = (
    ("нативная", "Решили окончательно: пишем нативно на Kotlin, Jetpack Compose."),
    ("кроссплатформа", "Решили окончательно: пишем на Flutter, один код на две платформы."),
)
BRANCH_QUESTION = "Напомни, на чём мы договорились писать приложение и почему."

# Бесплатные тарифы ограничивают частоту: Groq на этой модели отдаёт тысячу
# токенов вывода в минуту, и прогон из полусотни запросов упирается в неё
# гарантированно. Отказ 429 — это про тариф, а не про стратегию, поэтому
# запрос ждёт и повторяется, а не обрывает сравнение на середине.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_PAUSE = 40.0

# Окно, в которое загоняется разговор. Оно намеренно разное: заглушка отвечает
# сотней токенов, живая модель — тысячей, и одно и то же окно в одном случае
# не кончилось бы никогда, а в другом не вместило бы даже дословный хвост.
STUB_WINDOW = 2500
LIVE_WINDOW = 4000
WINDOW = STUB_WINDOW
ROUNDS = 14
# Сколько последних сообщений уходит дословно и сколько копится до пересказа.
# Оба числа ниже, чем по умолчанию у агента (6 и 10): в разговоре на десяток
# вопросов умолчания не успели бы сработать ни разу.
KEEP_LAST = 4
EVERY = 6

# Четыре стратегии на одном разговоре. Обрезка разрешена всем, кроме полной
# истории: у неё задача другая — показать, где кончается окно.
WAYS = (
    ("full", "вся переписка целиком", dict(on_overflow="stop")),
    ("window", "последние {} сообщений".format(KEEP_LAST), dict(on_overflow="trim")),
    ("facts", "картотека плюс последние {}".format(KEEP_LAST), dict(on_overflow="trim")),
    ("summary", "пересказ плюс последние {}".format(KEEP_LAST), dict(on_overflow="trim")),
)


@dataclass
class Recall:
    """Один проверочный вопрос в конце разговора."""

    question: str
    expected: str
    answer: str = ""
    ok: bool = False


@dataclass
class Run:
    """Одна стратегия: чем кончилась и во что обошлась."""

    title: str
    note: str
    steps: List[Step] = field(default_factory=list)
    asked: int = 0
    answered: int = 0
    broke: bool = False
    outcome: str = ""
    # Что стратегия отложила: сколько сообщений не уходит в модель и чем они
    # заменены — пересказом, картотекой или ничем.
    held_back: int = 0
    summary: str = ""
    facts: Dict[str, str] = field(default_factory=dict)
    # Почему стратегия сработала не полностью: пересказ не получен, картотека
    # не разобрана. Отдельно от ``note``: то — описание стратегии в заголовке.
    trouble: str = ""
    forgotten: int = 0
    narrowed: int = 0
    # Итог по счётчикам агента: сюда входят и служебные запросы.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    service_requests: int = 0
    cost: Optional[float] = None
    currency: str = "USD"
    recalls: List[Recall] = field(default_factory=list)
    recall_note: str = ""
    # Сколько всего стоили служебные запросы к концу каждого шага. Нужен именно
    # накопленный итог по шагам: сравнивать полный расход прогонов, дошедших
    # до разного числа вопросов, было бы подлогом.
    service_spend: List[int] = field(default_factory=list)

    def remember(self, agent: Agent) -> None:
        """Дописать последний обмен в свою историю роста.

        Своя, а не агентова: обрезка выбрасывает старые обмены из переписки, и
        после неё ``agent.growth()`` начинается заново — в отчёте это выглядело
        бы так, будто разговор подешевел сам собой.
        """
        recent = agent.growth()
        if not recent:
            return
        step = recent[-1]
        self.steps.append(Step(
            number=len(self.steps) + 1, question=step.question,
            prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens,
            cost=step.cost, currency=step.currency))
        self.service_spend.append(service_tokens(agent))

    def absorb(self, agent: Agent) -> None:
        """Забрать у агента итоговые счётчики и состояние стратегии."""
        usage = agent.usage
        self.prompt_tokens = usage.prompt_tokens
        self.completion_tokens = usage.completion_tokens
        self.requests = usage.requests
        self.service_requests = (usage.by_kind[SUMMARY].requests
                                 + usage.by_kind[FACTS].requests)
        self.currency = agent.config.model_info.currency
        self.cost = usage.cost(self.currency) or None
        self.held_back = agent.conversation.sent_from()
        self.summary = agent.conversation.summary
        self.facts = dict(agent.conversation.facts)
        self.trouble = agent.facts_note or agent.compression_note

    @property
    def remembered(self) -> int:
        return sum(1 for recall in self.recalls if recall.ok)

    @property
    def last_input(self) -> int:
        return self.steps[-1].prompt_tokens if self.steps else 0

    def spent_upto(self, count: int) -> int:
        """Во что обошлись первые ``count`` вопросов вместе со служебными."""
        answers = sum(step.tokens for step in self.steps[:count])
        return answers + (self.service_spend[count - 1] if count else 0)


def service_tokens(agent: Agent) -> int:
    """Во что обошлись служебные запросы стратегии — пересказ и картотека."""
    return (agent.usage.by_kind[SUMMARY].total_tokens
            + agent.usage.by_kind[FACTS].total_tokens)


def main(argv=None) -> int:
    # Окно и длина разговора задаются флагами и нужны всем прогонам сразу.
    # Носить их параметрами через каждую функцию было бы шумнее, чем назвать
    # один раз здесь.
    global WINDOW, ROUNDS

    parser = argparse.ArgumentParser(
        description="Четыре стратегии управления контекстом на одном сборе ТЗ")
    parser.add_argument("--live", action="store_true",
                        help="настоящие запросы к модели (по умолчанию заглушка)")
    parser.add_argument("--step", action="store_true",
                        help="показывать по шагам, между шагами ждать Enter")
    parser.add_argument("--ask-keys", action="store_true",
                        help="не подхватывать сохранённые реквизиты, спросить их заново")
    parser.add_argument("--provider", metavar="КЛЮЧ",
                        help="провайдер без расспроса, например groq")
    parser.add_argument("--model", metavar="ИМЯ", help="модель без расспроса")
    parser.add_argument("--window", type=int, metavar="ТОКЕНОВ",
                        help="во сколько токенов загнать разговор "
                             "(по умолчанию {} на заглушке и {} на живой модели)".format(
                                 STUB_WINDOW, LIVE_WINDOW))
    parser.add_argument("--rounds", type=int, default=ROUNDS, metavar="ЧИСЛО",
                        help="сколько вопросов задать (по умолчанию {})".format(ROUNDS))
    parser.add_argument("--max-tokens", type=int, metavar="ЧИСЛО",
                        help="предел длины ответа: на бесплатном тарифе короткие "
                             "ответы позволяют дойти до конца, не упираясь в частоту")
    parser.add_argument("--only", metavar="СТРАТЕГИЯ",
                        help="прогнать одну стратегию вместо всех")
    parser.add_argument("--no-branching", action="store_true",
                        help="без показа ветвления")
    args = parser.parse_args(argv)
    WINDOW = args.window or (LIVE_WINDOW if args.live else STUB_WINDOW)
    ROUNDS = args.rounds

    console = make_console()
    try:
        base = chosen_config(console, args)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130
    except (ConfigError, MissingCredentials) as exc:
        console.print(error(exc))
        return 1

    if args.max_tokens:
        base = base.with_changes(
            generation=replace(base.generation, max_tokens=args.max_tokens))

    console.clear()
    console.print(plan_panel(base))
    wait(console, args.step, "Enter — начать")

    ways = [way for way in WAYS if not args.only or way[0] == args.only]
    if not ways:
        console.print(error(ConfigError("нет стратегии «{}»; есть: {}".format(
            args.only, ", ".join(name for name, _, _ in WAYS)))))
        return 1

    runs: List[Run] = []
    try:
        for title, note, history in ways:
            run = dialog(console, base, args.live, title, note, history)
            console.print(run_panel(run))
            wait(console, args.step, "Enter — дальше")
            runs.append(run)
        console.print(summary_panel(runs))
        console.print(conclusion_panel(runs))
        if not args.no_branching:
            wait(console, args.step, "Enter — ветвление")
            console.print(branching_panel(branching(console, base, args.live)))
    except (LLMError, AgentError) as exc:
        console.print(error(exc))
        return 1
    return 0


# --------------------------------------------------------------------------
# Прогоны
# --------------------------------------------------------------------------

def history_for(strategy: str, **history) -> HistoryConfig:
    """Настройки истории для одной стратегии — одинаковые во всём, кроме неё."""
    return HistoryConfig(
        window=WINDOW, strategy=strategy, keep_last=KEEP_LAST,
        facts=FactsConfig(max_tokens=300, max_facts=20, update_every=1),
        compression=CompressionConfig(every=EVERY, max_tokens=400), **history)


def dialog(console: Console, base: AgentConfig, live: bool, title: str,
           note: str, history: dict) -> Run:
    """Провести разговор одной стратегией и проверить, что от него осталось."""
    run = Run(title, note)
    agent = Agent(base.with_changes(name=title, history=history_for(title, **history)))
    converse(console, agent, run)
    check_memory(console, agent, run, live)
    run.absorb(agent)
    return run


def converse(console: Console, agent: Agent, run: Run) -> None:
    """Сбор ТЗ: сведения вперемешку с обычными вопросами."""
    brief = list(BRIEF)
    narrowed_from = agent.conversation.keep_last
    for number in range(ROUNDS):
        # Сведения расставлены по разговору, а не свалены в начало: иначе их
        # запоминала бы любая стратегия — они попадали бы в дословный хвост.
        if brief and number % 3 == 0:
            if not ask(console, agent, run, brief.pop(0)[0]):
                return
        if not ask(console, agent, run, QUESTIONS[number % len(QUESTIONS)]):
            return
    run.narrowed = narrowed_from - agent.conversation.keep_last
    run.outcome = "дошёл до конца, места осталось {} токенов".format(
        fmt(agent.free_tokens()))


def ask(console: Console, agent: Agent, run: Run, question: str) -> bool:
    """Один вопрос. False означает, что разговор дальше не идёт."""
    run.asked += 1
    before = len(agent.conversation.messages)
    try:
        patiently(console, agent.ask, question,
                  "{}: вопрос {}".format(run.title, run.asked))
    except ContextOverflow as exc:
        run.broke = True
        run.outcome = str(exc)
        return False
    run.answered += 1
    # Сколько сообщений выброшено обрезкой: было столько-то, стало на два
    # больше, а если меньше — разницу съела обрезка.
    run.forgotten += max(0, before + 2 - len(agent.conversation.messages))
    run.remember(agent)
    return True


def patiently(console: Console, call, argument, what: str):
    """Один запрос с ожиданием, если тариф ответил отказом по частоте.

    Ждать приходится именно здесь, а не внутри агента: пауза — свойство
    бесплатного тарифа и этого показа, а не свойство разговора. Настоящему
    приложению нужно решать самому, ждать ему или ответить человеку сразу.
    """
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with console.status("[dim]{}…[/]".format(what), spinner="dots"):
                return call(argument)
        except LLMError as exc:
            if "429" not in str(exc) or attempt == RATE_LIMIT_RETRIES:
                raise
        with console.status("[dim]{}: тариф просит подождать, пробую снова "
                            "через {:.0f} с…[/]".format(what, RATE_LIMIT_PAUSE),
                            spinner="dots"):
            time.sleep(RATE_LIMIT_PAUSE)
    raise LLMError("тариф отказывает по частоте дольше, чем мы готовы ждать")


def check_memory(console: Console, agent: Agent, run: Run, live: bool) -> None:
    """Помнит ли агент то, что ему сказали по ходу сбора ТЗ.

    На заглушке проверять нечего: она отвечает заготовленным текстом и «помнит»
    ровно столько же при любой стратегии. Сравнивать по ней можно только расход.
    """
    if not live:
        run.recall_note = ("проверяется только на живой модели: заглушка "
                           "отвечает заготовленным текстом")
        return
    for question, expected in RECALLS:
        recall = Recall(question, expected)
        run.recalls.append(recall)
        try:
            recall.answer = patiently(console, agent.ask, question,
                                      "{}: проверяю память".format(run.title)).text.strip()
        except (ContextOverflow, LLMError, AgentError) as exc:
            recall.answer = "не ответил: {}".format(exc)
            continue
        recall.ok = expected.lower() in recall.answer.lower()


@dataclass
class Fork:
    """Одна ветка показа ветвления: что в ней решили и что она помнит."""

    name: str
    decision: str
    answer: str = ""
    messages: int = 0
    inherited: int = 0


def branching(console: Console, base: AgentConfig, live: bool) -> List[Fork]:
    """Две ветки от одной точки: что решили в каждой и что каждая помнит."""
    agent = Agent(base.with_changes(name="ветвление", history=history_for(
        "window", on_overflow="trim")))
    with console.status("[dim]ветвление: общее начало разговора…[/]", spinner="dots"):
        agent.ask(BRIEF[0][0])
        agent.ask(FORK_POINT)
    agent.mark_checkpoint("выбор-стека")

    forks: List[Fork] = []
    for name, decision in BRANCHES:
        branch = agent.fork(name, "выбор-стека")
        fork = Fork(name, decision, inherited=branch.inherited)
        with console.status("[dim]ветвление: ветка {}…[/]".format(name), spinner="dots"):
            agent.ask(decision)
            fork.answer = agent.ask(BRANCH_QUESTION).text.strip()
        fork.messages = len(agent.conversation.messages)
        forks.append(fork)
    return forks


def chosen_config(console: Console, args) -> AgentConfig:
    """Конфиг для всех прогонов: названный флагами или выбранный в расспросе."""
    if not (args.provider or args.model):
        return setup(console, not args.live, banner=banner, ask_keys=args.ask_keys)
    banner(console)
    provider, model = named_model(args.provider, args.model)
    config = AgentConfig(provider=provider, model=model,
                         transport=Transport(demo=not args.live, demo_delay=0.0))
    return agent_from_config(console, config, CredentialStore(
        demo=not args.live, ask_keys=args.ask_keys)).config


def named_model(provider: Optional[str], model: Optional[str]) -> tuple:
    """Пара «провайдер, модель» по флагам. Без модели берётся первая у провайдера."""
    if model:
        found, name = find_model(model)
        return (provider or found), name
    known = PROVIDERS.get(provider)
    if known is None:
        raise ConfigError("неизвестный провайдер «{}»; доступны: {}".format(
            provider, ", ".join(PROVIDER_ORDER)))
    return known.key, known.default_model.id


# --------------------------------------------------------------------------
# Показ
# --------------------------------------------------------------------------

def banner(console: Console) -> None:
    console.clear()
    console.print(Panel(
        Group(
            Align.center(Text("🧠  СТРАТЕГИИ КОНТЕКСТА", style="bold white")),
            Text(),
            Align.center(Text(
                "один сбор ТЗ четырьмя способами: вся история, окно, факты, пересказ\n"
                "и ветвление: два продолжения от одного места",
                style="dim", justify="center")),
        ),
        box=box.DOUBLE, border_style="magenta", padding=(1, 4)))


def plan_panel(base: AgentConfig) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Стратегия", style="bold", no_wrap=True)
    table.add_column("Что уходит в модель")
    table.add_column("Чем платит")
    table.add_row("full", "вся переписка целиком", "дороже всех, раньше всех упрётся в окно")
    table.add_row("window", "последние {} сообщений".format(KEEP_LAST),
                  "ничем — и потому забывает начало")
    table.add_row("facts", "картотека плюс последние {}".format(KEEP_LAST),
                  "по короткому запросу на каждый вопрос")
    table.add_row("summary", "пересказ плюс последние {}".format(KEEP_LAST),
                  "по запросу на каждые {} сообщений".format(EVERY))
    return Panel(
        Group(
            Text.from_markup(
                "Разговор один и тот же: [bold]{}[/] вопросов сбора ТЗ, в которые "
                "вплетены пять сведений — платформа, бюджет, срок, решение по входу "
                "и предпочтение по теме. В конце все пять спрашиваются обратно.\n"
                "Окно сужено до [bold]{}[/] токенов вместо {} у модели: упираться "
                "в настоящее окно значило бы потратить деньги ради того же вывода.\n"
                "Неотправленное оценивается токенизатором [bold]{}[/], "
                "всё остальное — точные числа от провайдера.\n".format(
                    ROUNDS, fmt(WINDOW), fmt(base.model_info.context_window),
                    tokenizer_name())),
            table),
        title="Что сейчас произойдёт", title_align="left",
        border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def run_panel(run: Run) -> RenderableType:
    blocks: List[RenderableType] = [growth_table(run), outcome_line(run)]
    if run.summary:
        blocks.append(summary_block(run))
    if run.facts:
        blocks.append(facts_block(run))
    if run.trouble:
        blocks.append(Text("Стратегия сработала не полностью: {}".format(run.trouble),
                           style="yellow"))
    if run.title == "facts" and not run.facts:
        blocks.append(Text(
            "Картотека пуста. На заглушке это нормально: она отвечает "
            "заготовленной прозой и объекта JSON не возвращает — запустите "
            "с --live.", style="yellow"))
    if run.recalls or run.recall_note:
        blocks.append(memory_line(run))
    return Panel(Group(*blocks), title="{} — {}".format(run.title, run.note),
                 title_align="left", box=box.ROUNDED, padding=(0, 1),
                 border_style="red" if run.broke else "magenta")


def growth_table(run: Run) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("№", style="dim", justify="right", no_wrap=True)
    table.add_column("Вопрос")
    table.add_column("Вход", justify="right", no_wrap=True)
    table.add_column("Выход", justify="right", no_wrap=True)
    table.add_column("Цена шага", justify="right", no_wrap=True)
    for step in shortened(run.steps):
        if step is None:
            table.add_row("…", Text("пропущено", style="dim"), "…", "…", "…")
            continue
        table.add_row(
            str(step.number), Text(clip(step.question, 44), style="dim"),
            fmt(step.prompt_tokens), fmt(step.completion_tokens),
            format_cost(step.cost, step.currency))
    return table


def outcome_line(run: Run) -> Text:
    line = Text()
    if run.broke:
        line.append("Встал на вопросе {}: ".format(run.asked), style="bold red")
        line.append(run.outcome)
        return line
    line.append("Прошло {} {}: ".format(
        run.answered, plural(run.answered, ("вопрос", "вопроса", "вопросов"))),
        style="bold green")
    line.append(run.outcome, style="dim")
    if run.forgotten:
        line.append("  ·  выброшено обрезкой {} {}".format(
            run.forgotten,
            plural(run.forgotten, ("сообщение", "сообщения", "сообщений"))),
            style="yellow")
    if run.narrowed:
        line.append("  ·  окно сузилось на {} {} — запрос не помещался".format(
            run.narrowed,
            plural(run.narrowed, ("сообщение", "сообщения", "сообщений"))),
            style="yellow")
    return line


def summary_block(run: Run) -> RenderableType:
    head = Text()
    head.append("Не уходит в модель: {} {} — вместо них пересказ".format(
        run.held_back, plural(run.held_back, ("сообщение", "сообщения", "сообщений"))),
        style="green")
    head.append("  ·  служебных запросов: {}".format(run.service_requests), style="dim")
    return Group(head, Text(clip(run.summary, 600), style="dim"))


def facts_block(run: Run) -> RenderableType:
    head = Text()
    head.append("Картотека: {} {}".format(
        len(run.facts), plural(len(run.facts), ("факт", "факта", "фактов"))),
        style="green")
    head.append("  ·  служебных запросов: {}".format(run.service_requests), style="dim")
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_header=False)
    table.add_column("Ключ", style="cyan", no_wrap=True)
    table.add_column("Значение", style="dim")
    for key, value in run.facts.items():
        table.add_row(key, clip(value, 70))
    return Group(head, table)


def memory_line(run: Run) -> Text:
    line = Text()
    line.append("Память: ", style="bold")
    if run.recall_note:
        line.append(run.recall_note, style="dim")
        return line
    for recall in run.recalls:
        line.append("«{}» — ".format(recall.expected), style="dim")
        line.append("помнит" if recall.ok else "не помнит",
                    style="green" if recall.ok else "red")
        line.append("  ")
    return line


def summary_panel(runs: List[Run]) -> RenderableType:
    """Все стратегии в одной таблице — ради этого всё и затевалось."""
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Стратегия", style="bold", no_wrap=True)
    table.add_column("Ответов", justify="right", no_wrap=True)
    table.add_column("Вход", justify="right", no_wrap=True)
    table.add_column("Выход", justify="right", no_wrap=True)
    table.add_column("Запросов", justify="right", no_wrap=True)
    table.add_column("Вход послед.", justify="right", no_wrap=True)
    table.add_column("Цена", justify="right", no_wrap=True)
    table.add_column("Помнит", justify="right", no_wrap=True)
    for run in runs:
        table.add_row(
            run.title, str(run.answered), fmt(run.prompt_tokens),
            fmt(run.completion_tokens),
            "{}{}".format(run.requests,
                          " (+{} служ.)".format(run.service_requests)
                          if run.service_requests else ""),
            fmt(run.last_input), format_cost(run.cost, run.currency),
            "—" if run.recall_note else "{} из {}".format(
                run.remembered, len(run.recalls)))
    return Panel(table, title="Стратегии рядом", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def conclusion_panel(runs: List[Run]) -> RenderableType:
    """Что из этих чисел следует. Считается по прогону, а не написано заранее."""
    lines: List[Text] = []
    named = {run.title: run for run in runs}
    whole = named.get("full")

    if whole:
        lines.append(bullet(
            "Полная история", "{} {} из {}, {}".format(
                whole.answered, plural(whole.answered, ("ответ", "ответа", "ответов")),
                whole.asked,
                "разговор встал: окно кончилось" if whole.broke
                else "окна хватило — попробуйте --rounds побольше")))

    common = min((len(run.steps) for run in runs), default=0)
    if common and whole:
        for run in runs:
            if run is whole:
                continue
            before, after = whole.spent_upto(common), run.spent_upto(common)
            lines.append(bullet(
                "Расход на первых {} {}: {}".format(
                    common, plural(common, ("вопросе", "вопросах", "вопросах")),
                    run.title),
                "{} токенов против {} у полной истории{}; служебные запросы "
                "стоили {}".format(
                    fmt(after), fmt(before),
                    " — на {:.0%} меньше".format(1 - after / before)
                    if before and after < before else " — дороже",
                    fmt(run.service_spend[common - 1] if run.service_spend else 0))))

    if any(run.recalls for run in runs):
        lines.append(bullet("Память", "  ".join(
            "{} — {} из {}".format(run.title, run.remembered, len(run.recalls))
            for run in runs)))
    else:
        lines.append(bullet("Память", "на заглушке не проверяется — "
                            "запустите с --live"))

    lines.append(bullet("Чем платят", "окно — ничем и потому забывает; "
                        "факты — коротким запросом на каждый вопрос; "
                        "пересказ — запросом на каждые {} сообщений".format(EVERY)))
    return Panel(Group(*lines), title="Что из этого следует", title_align="left",
                 border_style="green", box=box.ROUNDED, padding=(0, 1))


def branching_panel(forks: List[Fork]) -> RenderableType:
    """Две ветки от одного места: ответы должны разойтись."""
    blocks: List[RenderableType] = [Text.from_markup(
        "От одной точки отведены две ветки с несовместимыми решениями, и в "
        "каждой задан один и тот же вопрос: [dim]«{}»[/]\n".format(BRANCH_QUESTION))]
    for fork in forks:
        head = Text()
        head.append("Ветка {}".format(fork.name), style="bold magenta")
        head.append("  ·  унаследовано {} {}, всего {}".format(
            fork.inherited,
            plural(fork.inherited, ("сообщение", "сообщения", "сообщений")),
            fork.messages), style="dim")
        blocks.append(Group(head, Text(clip(fork.answer, 400), style="dim"), Text()))
    answers = [fork.answer for fork in forks]
    verdict = Text()
    verdict.append("Итог: ", style="bold")
    if len(set(answers)) == len(answers) and all(answers):
        verdict.append("ответы разошлись — ветки независимы, решение соседней "
                       "в этот разговор не попало.", style="green")
    else:
        verdict.append("ответы совпали — смотрите, что именно вернула модель.",
                       style="yellow")
    blocks.append(verdict)
    return Panel(Group(*blocks), title="Ветвление", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def bullet(title: str, text: str) -> Text:
    line = Text("• ", style="green")
    line.append(title + ": ", style="bold")
    line.append(text)
    return line


def shortened(steps: List[Step]):
    """Длинный список шагов с пропуском в середине: экран не резиновый."""
    if len(steps) <= 10:
        return list(steps)
    return steps[:3] + [None] + steps[-5:]


def clip(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def error(exc: Exception) -> RenderableType:
    return Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                 border_style="red", box=box.ROUNDED)


def wait(console: Console, step: bool, hint: str) -> None:
    if not step:
        return
    console.print("[dim]{}[/]".format(hint), end="")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(130)


if __name__ == "__main__":
    raise SystemExit(main())
