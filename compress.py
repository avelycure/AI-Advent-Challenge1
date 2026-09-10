#!/usr/bin/env python3
"""Один и тот же разговор тремя способами: вся история, обрезка, сжатие.

Вопросы во всех трёх одинаковые, окно одинаковое, модель одна. Разное только
то, что уходит в модель на каждом шаге:

* **вся история** — вся переписка целиком, пока помещается;
* **обрезка** — самое старое выбрасывается, чтобы влезло;
* **сжатие** — начало заменяется пересказом, последние сообщения идут дословно.

Проверяются две вещи сразу: во что обошёлся разговор и помнит ли агент то, что
ему сказали в начале. По отдельности ни то ни другое ничего не доказывает —
забыть всё дёшево, а помнить всё дорого.

Окно намеренно сужено до нескольких тысяч токенов. У настоящих моделей оно на
сто тысяч и больше, и упираться в него значило бы потратить деньги ради того
же самого вывода.

Запуск:

    ./compress.sh              на заглушке, бесплатно
    ./compress.sh --live       настоящие запросы к выбранной модели
    ./compress.sh --step       по шагам, между шагами ждёт Enter
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

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
    HistoryConfig,
    LLMError,
    MissingCredentials,
    Step,
    Transport,
    find_model,
)
from llmagent.transport import PROVIDER_ORDER, PROVIDERS, tokenizer_name
from llmagent.usage import SUMMARY
from llmchat.app import agent_from_config, setup
from llmchat.switching import CredentialStore
from llmchat.ui import format_cost, fmt, make_console, plural

# То, что сказано в начале и в середине разговора и спрашивается в конце.
# Начало проверяет сжатие на прочность: к последнему вопросу оно давно свёрнуто
# в пересказ, а обрезка его к этому времени попросту выбросила.
NAME = "Иван"
CITY = "Новосибирск"
INTRODUCTION = ("Меня зовут {}. Запомни это имя, я спрошу его в конце.".format(NAME))
MIDDLE_NOTE = ("Ещё одно: я живу в городе {}. Это тоже понадобится в конце.".format(CITY))

RECALLS = (
    ("Как меня зовут? Ответь одним словом.", NAME),
    ("В каком городе я живу? Ответь одним словом.", CITY),
)

QUESTIONS = [
    "Чем куча отличается от стека и когда выбирают кучу?",
    "Как работает алгоритм Дейкстры и в чём его ограничение?",
    "Зачем нужны красно-чёрные деревья, если есть обычные двоичные?",
    "Что такое амортизированная сложность на примере динамического массива?",
    "Чем хеш-таблица хуже дерева поиска и когда это заметно?",
    "Как устроен алгоритм Кнута — Морриса — Пратта?",
    "Что делает быструю сортировку быстрой и когда она вырождается?",
    "Зачем нужен фильтр Блума и чем он расплачивается за компактность?",
]

# Окно, в которое загоняется разговор. Оно намеренно разное: заглушка отвечает
# сотней токенов, живая модель — тысячей, и одно и то же окно в одном случае
# не кончилось бы никогда, а в другом не вместило бы даже дословный хвост.
STUB_WINDOW = 2500
LIVE_WINDOW = 4000
WINDOW = STUB_WINDOW
ROUNDS = 14
# Сколько последних сообщений сжатие оставляет дословно и сколько копит до
# пересказа. Оба числа ниже, чем по умолчанию у агента (6 и 10): в разговоре
# на десяток вопросов умолчания не успели бы сработать ни разу.
KEEP_LAST = 4
EVERY = 6


@dataclass
class Recall:
    """Один проверочный вопрос в конце разговора."""

    question: str
    expected: str
    answer: str = ""
    ok: bool = False


@dataclass
class Run:
    """Один способ вести разговор: чем кончился и во что обошёлся."""

    title: str
    note: str
    steps: List[Step] = field(default_factory=list)
    asked: int = 0
    answered: int = 0
    broke: bool = False
    outcome: str = ""
    # Сколько сообщений свёрнуто в пересказ и что в нём написано.
    folded: int = 0
    summary: str = ""
    forgotten: int = 0
    # Итог по счётчикам агента: сюда входят и запросы за пересказом.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    summary_requests: int = 0
    cost: Optional[float] = None
    currency: str = "USD"
    recalls: List[Recall] = field(default_factory=list)
    recall_note: str = ""
    # Сколько всего стоили пересказы к концу каждого шага. Нужен именно
    # накопленный итог по шагам: сравнивать полный расход прогонов, дошедших
    # до разного числа вопросов, было бы подлогом.
    folded_spend: List[int] = field(default_factory=list)

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
        self.folded_spend.append(agent.usage.by_kind[SUMMARY].total_tokens)

    def absorb(self, agent: Agent) -> None:
        """Забрать у агента итоговые счётчики и состояние пересказа."""
        usage = agent.usage
        self.prompt_tokens = usage.prompt_tokens
        self.completion_tokens = usage.completion_tokens
        self.requests = usage.requests
        self.summary_requests = usage.by_kind[SUMMARY].requests
        self.currency = agent.config.model_info.currency
        self.cost = usage.cost(self.currency) or None
        self.folded = agent.conversation.summarized
        self.summary = agent.conversation.summary

    @property
    def remembered(self) -> int:
        return sum(1 for recall in self.recalls if recall.ok)

    @property
    def last_input(self) -> int:
        return self.steps[-1].prompt_tokens if self.steps else 0

    def spent_upto(self, count: int) -> int:
        """Во что обошлись первые ``count`` вопросов вместе с пересказами."""
        answers = sum(step.tokens for step in self.steps[:count])
        return answers + (self.folded_spend[count - 1] if count else 0)


# Три способа вести один и тот же разговор.
WAYS = (
    ("Вся история", "переписка уходит в модель целиком",
     dict(on_overflow="stop", compression=CompressionConfig(enabled=False))),
    ("Обрезка", "старое выбрасывается, чтобы влезть",
     dict(on_overflow="trim", compression=CompressionConfig(enabled=False))),
    ("Сжатие", "начало заменяется пересказом",
     dict(on_overflow="trim", compression=CompressionConfig(
         enabled=True, keep_last=KEEP_LAST, every=EVERY))),
)


def main(argv=None) -> int:
    # Окно и длина разговора задаются флагами и нужны всем прогонам сразу.
    # Носить их параметрами через каждую функцию было бы шумнее, чем назвать
    # один раз здесь.
    global WINDOW, ROUNDS

    parser = argparse.ArgumentParser(
        description="Вся история, обрезка и сжатие — на одном разговоре")
    parser.add_argument("--live", action="store_true",
                        help="настоящие запросы к модели (по умолчанию заглушка)")
    parser.add_argument("--step", action="store_true",
                        help="показывать по шагам, между шагами ждать Enter")
    parser.add_argument("--ask-keys", action="store_true",
                        help="не подхватывать сохранённые реквизиты, спросить их заново")
    parser.add_argument("--provider", metavar="КЛЮЧ",
                        help="провайдер без расспроса, например openrouter")
    parser.add_argument("--model", metavar="ИМЯ", help="модель без расспроса")
    parser.add_argument("--window", type=int, metavar="ТОКЕНОВ",
                        help="во сколько токенов загнать разговор "
                             "(по умолчанию {} на заглушке и {} на живой модели)".format(
                                 STUB_WINDOW, LIVE_WINDOW))
    parser.add_argument("--rounds", type=int, default=ROUNDS, metavar="ЧИСЛО",
                        help="сколько вопросов задать (по умолчанию {})".format(ROUNDS))
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

    console.clear()
    console.print(plan_panel(base))
    wait(console, args.step, "Enter — начать")

    runs: List[Run] = []
    try:
        for title, note, history in WAYS:
            run = dialog(console, base, args.live, title, note, history)
            console.print(run_panel(run))
            wait(console, args.step, "Enter — дальше")
            runs.append(run)
    except (LLMError, AgentError) as exc:
        console.print(error(exc))
        return 1

    console.print(summary_panel(runs))
    console.print(conclusion_panel(runs))
    return 0


# --------------------------------------------------------------------------
# Прогоны
# --------------------------------------------------------------------------

def dialog(console: Console, base: AgentConfig, live: bool, title: str,
           note: str, history: dict) -> Run:
    """Провести разговор одним способом и проверить, что от него осталось."""
    run = Run(title, note)
    agent = Agent(base.with_changes(
        name=title, history=HistoryConfig(window=WINDOW, **history)))
    converse(console, agent, run)
    check_memory(console, agent, run, live)
    run.absorb(agent)
    return run


def converse(console: Console, agent: Agent, run: Run) -> None:
    """Задать вопросы по кругу, вставив в середину то, что спросят в конце."""
    if not ask(console, agent, run, INTRODUCTION):
        return
    middle = max(1, ROUNDS // 2)
    for number in range(ROUNDS):
        if number == middle and not ask(console, agent, run, MIDDLE_NOTE):
            return
        if not ask(console, agent, run, QUESTIONS[number % len(QUESTIONS)]):
            return
    run.outcome = "дошёл до конца, места осталось {} токенов".format(
        fmt(agent.free_tokens()))


def ask(console: Console, agent: Agent, run: Run, question: str) -> bool:
    """Один вопрос. False означает, что разговор дальше не идёт."""
    run.asked += 1
    before = len(agent.conversation.messages)
    try:
        with console.status("[dim]{}: вопрос {}…[/]".format(
                run.title.lower(), run.asked), spinner="dots"):
            agent.ask(question)
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


def check_memory(console: Console, agent: Agent, run: Run, live: bool) -> None:
    """Помнит ли агент то, что ему сказали в начале и в середине.

    На заглушке проверять нечего: она отвечает заготовленным текстом и «помнит»
    ровно столько же при любом способе вести разговор. Сравнивать по ней можно
    только расход.
    """
    if not live:
        run.recall_note = ("проверяется только на живой модели: заглушка "
                           "отвечает заготовленным текстом")
        return
    for question, expected in RECALLS:
        recall = Recall(question, expected)
        run.recalls.append(recall)
        try:
            with console.status("[dim]{}: проверяю память…[/]".format(
                    run.title.lower()), spinner="dots"):
                recall.answer = agent.ask(question).text.strip()
        except (ContextOverflow, LLMError, AgentError) as exc:
            recall.answer = "не ответил: {}".format(exc)
            continue
        recall.ok = expected.lower() in recall.answer.lower()


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
            Align.center(Text("🗜  СЖАТИЕ ИСТОРИИ", style="bold white")),
            Text(),
            Align.center(Text(
                "один разговор тремя способами: вся история, обрезка, пересказ\n"
                "видно, во что обходится память и чем за неё платят",
                style="dim", justify="center")),
        ),
        box=box.DOUBLE, border_style="magenta", padding=(1, 4)))


def plan_panel(base: AgentConfig) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Способ", style="bold", no_wrap=True)
    table.add_column("Что уходит в модель")
    table.add_column("Чего ждём")
    table.add_row("Вся история", "вся переписка целиком",
                  "дороже всех и раньше всех упрётся в окно")
    table.add_row("Обрезка", "последние сообщения, что влезли",
                  "дёшево, но начало разговора потеряно")
    table.add_row("Сжатие", "пересказ начала плюс последние {}".format(KEEP_LAST),
                  "дешевле полной истории, начало помнит")
    return Panel(
        Group(
            Text.from_markup(
                "Окно сужено до [bold]{}[/] токенов вместо {} у модели: упираться "
                "в настоящее окно значило бы потратить деньги ради того же вывода.\n"
                "Вопросов в каждом разговоре — [bold]{}[/], и они одинаковые. Имя "
                "названо в первом сообщении, город — в середине; оба спрашиваются "
                "в конце.\nНеотправленное оценивается токенизатором [bold]{}[/], "
                "всё остальное — точные числа от провайдера.\n".format(
                    fmt(WINDOW), fmt(base.model_info.context_window), ROUNDS,
                    tokenizer_name())),
            table),
        title="Что сейчас произойдёт", title_align="left",
        border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def run_panel(run: Run) -> RenderableType:
    blocks: List[RenderableType] = [growth_table(run), outcome_line(run)]
    if run.folded:
        blocks.append(summary_block(run))
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
    return line


def summary_block(run: Run) -> RenderableType:
    head = Text()
    head.append("Свёрнуто в пересказ: {} {}".format(
        run.folded, plural(run.folded, ("сообщение", "сообщения", "сообщений"))),
        style="green")
    head.append("  ·  запросов за пересказом: {}".format(run.summary_requests),
                style="dim")
    return Group(head, Text(clip(run.summary, 600), style="dim"))


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
    """Три способа в одной таблице — ради этого всё и затевалось."""
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Способ", style="bold", no_wrap=True)
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
                          " (+{} сжатие)".format(run.summary_requests)
                          if run.summary_requests else ""),
            fmt(run.last_input), format_cost(run.cost, run.currency),
            "—" if run.recall_note else "{} из {}".format(
                run.remembered, len(run.recalls)))
    return Panel(table, title="Три способа рядом", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def conclusion_panel(runs: List[Run]) -> RenderableType:
    """Что из этих чисел следует. Считается по прогону, а не написано заранее."""
    whole, trimmed, folded = runs
    lines: List[Text] = []

    lines.append(bullet(
        "Полная история", "{} {} из {}, {}".format(
            whole.answered, plural(whole.answered, ("ответ", "ответа", "ответов")),
            whole.asked,
            "разговор встал: окно кончилось" if whole.broke
            else "окна хватило — попробуйте --rounds побольше")))

    common = min(len(whole.steps), len(folded.steps))
    if common:
        before = whole.steps[common - 1].prompt_tokens
        after = folded.steps[common - 1].prompt_tokens
        lines.append(bullet(
            "Вход на общем шаге {}".format(common),
            "{} без сжатия против {} со сжатием{}".format(
                fmt(before), fmt(after),
                " — на {:.0%} меньше".format(1 - after / before)
                if before and after < before else "")))

    if common:
        # Сравниваются одинаковые куски разговора: у полной истории вопросов
        # меньше, и её общий расход выглядел бы скромнее просто потому, что
        # она раньше встала.
        before = whole.spent_upto(common)
        after = folded.spent_upto(common)
        lines.append(bullet(
            "Расход на первых {} {}".format(
                common, plural(common, ("вопросе", "вопросах", "вопросах"))),
            "{} токенов без сжатия против {} со сжатием — пересказ уже "
            "учтён и стоил {}{}".format(
                fmt(before), fmt(after), fmt(folded.folded_spend[common - 1]),
                "; выгода {:.0%}".format(1 - after / before)
                if before and after < before else
                "; на этой длине сжатие себя ещё не окупило")))

    shared = min(len(trimmed.steps), len(folded.steps))
    if shared:
        cheaply, dearly = trimmed.spent_upto(shared), folded.spent_upto(shared)
        lines.append(bullet(
            "Обрезка против сжатия",
            "{} токенов против {} на тех же {} вопросах — {}, но забытое "
            "обрезкой не хранится нигде".format(
                fmt(cheaply), fmt(dearly), shared,
                "обрезка дешевле" if cheaply < dearly else "сжатие дешевле")))

    if folded.recalls:
        lines.append(bullet("Память", "вся история — {} из {}, обрезка — {} из {}, "
                            "сжатие — {} из {}".format(
                                whole.remembered, len(whole.recalls),
                                trimmed.remembered, len(trimmed.recalls),
                                folded.remembered, len(folded.recalls))))
    else:
        lines.append(bullet("Память", "на заглушке не проверяется — "
                            "запустите с --live"))

    lines.append(bullet("Чем платим", "сжатие — это лишний запрос к модели раз в "
                        "{} сообщений; он окупается тем, что не уходит в модель "
                        "каждый следующий раз".format(EVERY)))
    return Panel(Group(*lines), title="Что из этого следует", title_align="left",
                 border_style="green", box=box.ROUNDED, padding=(0, 1))


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
