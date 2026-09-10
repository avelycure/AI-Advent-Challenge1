#!/usr/bin/env python3
"""Три диалога рядом: короткий, длинный и тот, что не влезает в окно.

Вопросы во всех трёх одинаковые — меняется только их количество и то, сколько
места агенту оставили. Так видно главное свойство диалога с моделью: каждый
следующий вопрос отправляет всю переписку заново, поэтому дорожает не потому,
что стал длиннее, а потому, что стал не первым.

Окно намеренно сужено до нескольких тысяч токенов. У настоящих моделей оно на
сто тысяч и больше, и упереться в него значило бы потратить деньги и время
ради того же самого вывода.

Запуск:

    ./tokens.sh              на заглушке, бесплатно
    ./tokens.sh --live       настоящие запросы к выбранной модели
    ./tokens.sh --step       по шагам, между шагами ждёт Enter
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
from llmchat.app import agent_from_config, setup
from llmchat.switching import CredentialStore
from llmchat.ui import format_cost, fmt, make_console, plural

# Имя называется первым вопросом и спрашивается последним. Это самая простая
# проверка памяти: короткий диалог имя помнит, обрезанный — уже нет.
NAME = "Иван"
INTRODUCTION = "Меня зовут {}. Запомни это имя, я спрошу его в конце.".format(NAME)
RECALL = "Как меня зовут? Ответь одним словом."

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

# Окно на пять тысяч токенов: короткий диалог укладывается с запасом, длинный
# подходит к краю, а третий заведомо не влезает.
WINDOW = 5000

SHORT_STEPS = 2
LONG_STEPS = 8
# Заведомо больше, чем поместится: сколько именно влезет, покажет сам прогон.
OVERFLOW_STEPS = 40
# На сколько вопросов обрезка должна пережить отказ, чтобы это было видно.
TRIM_MARGIN = 5


@dataclass
class Run:
    """Один диалог целиком: чем кончился и во что обошёлся."""

    title: str
    note: str
    steps: List[Step] = field(default_factory=list)
    # Сколько вопросов задано и сколько из них дошло до модели: у переполненного
    # диалога это разные числа, и в этом весь смысл прогона.
    asked: int = 0
    answered: int = 0
    forgotten: int = 0
    outcome: str = ""
    broke: bool = False
    remembered: Optional[bool] = None
    # Почему память не проверена: на заглушке проверять нечего, а в тесном
    # окне сам вопрос «как меня зовут» уже может не поместиться.
    recall_note: str = ""

    def remember(self, agent: Agent) -> None:
        """Дописать последний обмен в свою историю.

        Своя, а не агентова: обрезка выбрасывает старые обмены из переписки,
        и после неё ``agent.growth()`` начинается заново — в отчёте о росте
        цены это выглядело бы так, будто диалог подешевел.
        """
        recent = agent.growth()
        if not recent:
            return
        step = recent[-1]
        cost = step.cost
        total_cost = cost if self.cost is None else (
            self.cost + (cost or 0.0) if cost is not None else self.cost)
        self.steps.append(Step(
            number=len(self.steps) + 1, question=step.question,
            prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens,
            cost=cost, currency=step.currency,
            total_tokens=self.total_tokens + step.tokens, total_cost=total_cost))

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> Optional[float]:
        return self.steps[-1].total_cost if self.steps else None



    @property
    def currency(self) -> str:
        return self.steps[-1].currency if self.steps else "USD"

    @property
    def growth_ratio(self) -> float:
        """Во сколько раз последний запрос дороже первого."""
        if len(self.steps) < 2 or not self.steps[0].prompt_tokens:
            return 1.0
        return self.steps[-1].prompt_tokens / self.steps[0].prompt_tokens


def main(argv=None) -> int:
    # Окно задаётся флагом и нужно всем прогонам сразу. Носить одно число
    # пятым параметром через каждую функцию было бы шумнее, чем назвать его
    # один раз здесь.
    global WINDOW

    parser = argparse.ArgumentParser(
        description="Короткий, длинный и переполненный диалог — рядом")
    parser.add_argument("--live", action="store_true",
                        help="настоящие запросы к модели (по умолчанию заглушка)")
    parser.add_argument("--step", action="store_true",
                        help="показывать по шагам, между шагами ждать Enter")
    parser.add_argument("--ask-keys", action="store_true",
                        help="не подхватывать сохранённые реквизиты, спросить их заново")
    parser.add_argument("--provider", metavar="КЛЮЧ",
                        help="провайдер без расспроса, например openrouter")
    parser.add_argument("--model", metavar="ИМЯ", help="модель без расспроса")
    parser.add_argument("--window", type=int, default=WINDOW, metavar="ТОКЕНОВ",
                        help="во сколько токенов загнать диалог (по умолчанию {})".format(
                            WINDOW))
    args = parser.parse_args(argv)
    WINDOW = args.window

    console = make_console()
    try:
        base = chosen_config(console, args)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130
    except (ConfigError, MissingCredentials) as exc:
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                            border_style="red", box=box.ROUNDED))
        return 1

    console.clear()
    console.print(plan_panel(base))
    wait(console, args.step, "Enter — начать")

    runs: List[Run] = []
    try:
        for build in (short_dialog, long_dialog, overflowing_dialog):
            runs.append(show(console, build(console, base, args.live), args.step))
        # Обрезке хватает нескольких вопросов сверх того, на чём встал прежний
        # прогон: дальше повторялось бы одно и то же за настоящие токены.
        runs.append(show(console, trimming_dialog(
            console, base, args.live, runs[-1].asked + TRIM_MARGIN), args.step))
    except (LLMError, AgentError) as exc:
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Ошибка",
                            border_style="red", box=box.ROUNDED))
        return 1

    console.print(summary_panel(runs))
    console.print(conclusion_panel(runs))
    return 0


def chosen_config(console: Console, args) -> AgentConfig:
    """Конфиг для всех прогонов: названный флагами или выбранный в расспросе."""
    if not (args.provider or args.model):
        return setup(console, not args.live, banner=banner, ask_keys=args.ask_keys)
    # Названные флагом провайдер и модель — уже сделанный выбор; расспрос
    # поверх него только затирал бы названное, как это было в самом чате.
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
# Прогоны
# --------------------------------------------------------------------------

def short_dialog(console: Console, base: AgentConfig, live: bool) -> Run:
    run = Run("Короткий диалог", "{} вопроса — окно почти пустое".format(SHORT_STEPS))
    agent = make_agent(base, "short")
    converse(console, agent, run, SHORT_STEPS)
    finish(console, agent, run, live)
    return run


def long_dialog(console: Console, base: AgentConfig, live: bool) -> Run:
    run = Run("Длинный диалог", "{} вопросов — окно подходит к краю".format(LONG_STEPS))
    agent = make_agent(base, "long")
    converse(console, agent, run, LONG_STEPS)
    finish(console, agent, run, live)
    return run


def overflowing_dialog(console: Console, base: AgentConfig, live: bool) -> Run:
    run = Run("Сверх лимита", "спрашиваем, пока агент не откажет")
    agent = make_agent(base, "overflow")
    converse(console, agent, run, OVERFLOW_STEPS)
    return run


def trimming_dialog(console: Console, base: AgentConfig, live: bool,
                    rounds: int) -> Run:
    run = Run("С обрезкой", "то же самое, но history.on_overflow=trim")
    agent = make_agent(base, "trim", on_overflow="trim")
    converse(console, agent, run, rounds)
    finish(console, agent, run, live)
    return run


def show(console: Console, run: Run, step: bool) -> Run:
    console.print(run_panel(run))
    wait(console, step, "Enter — дальше")
    return run


def converse(console: Console, agent: Agent, run: Run, rounds: int) -> None:
    """Задать вопросы по кругу, считая всё, что случилось по дороге."""
    ask(console, agent, run, INTRODUCTION)
    for number in range(rounds):
        question = QUESTIONS[number % len(QUESTIONS)]
        if not ask(console, agent, run, question):
            return


def finish(console: Console, agent: Agent, run: Run, live: bool) -> None:
    """Диалог дошёл до конца: проверить память и подписать итог."""
    if not live:
        run.recall_note = "проверяется только на живой модели: заглушка " \
                          "отвечает заготовленным текстом"
    else:
        run.remembered = check_memory(console, agent)
        if run.remembered is None:
            run.recall_note = "проверить не вышло: сам вопрос про имя в окно " \
                              "уже не помещается"
    if run.forgotten:
        run.outcome = "дошёл до конца, забыв {} {}".format(
            run.forgotten,
            plural(run.forgotten, ("сообщение", "сообщения", "сообщений")))
    else:
        run.outcome = "дошёл до конца, места осталось {} токенов".format(
            fmt(agent.free_tokens()))


def ask(console: Console, agent: Agent, run: Run, question: str) -> bool:
    """Один вопрос. False означает, что диалог дальше не идёт."""
    run.asked += 1
    label = "{}: вопрос {}…".format(run.title.lower(), run.asked)
    before = len(agent.conversation.messages)
    try:
        with console.status("[dim]{}[/]".format(label), spinner="dots"):
            agent.ask(question)
    except ContextOverflow as exc:
        run.broke = True
        run.outcome = str(exc)
        return False
    run.answered += 1
    # Сколько сообщений забыто обрезкой: было столько-то, стало на два больше,
    # а если меньше — разницу съела обрезка.
    run.forgotten += max(0, before + 2 - len(agent.conversation.messages))
    run.remember(agent)
    return True


def check_memory(console: Console, agent: Agent) -> Optional[bool]:
    """Помнит ли модель имя, названное в самом начале.

    None означает, что проверка не состоялась: вопрос про имя — такой же
    вопрос, и в переполненное окно он не помещается наравне с прочими.
    """
    try:
        with console.status("[dim]проверяю память…[/]", spinner="dots"):
            answer = agent.ask(RECALL).text
    except (ContextOverflow, LLMError, AgentError):
        return None
    return NAME.lower() in answer.lower()


def make_agent(base: AgentConfig, name: str, **history) -> Agent:
    """Один и тот же конфиг во всех прогонах, кроме окна и поведения при отказе.

    Сжатие истории здесь выключено намеренно. Оно ровно затем и сделано, чтобы
    разговор в окно помещался, и с ним этот показ мерил бы не переполнение и не
    обрезку, а то, как хорошо сжатие им мешает случиться. Сжатие рядом с ними
    показывает ``./compress.sh``.
    """
    return Agent(base.with_changes(
        name=name, history=HistoryConfig(
            window=WINDOW, compression=CompressionConfig(enabled=False), **history)))


# --------------------------------------------------------------------------
# Показ
# --------------------------------------------------------------------------

def banner(console: Console) -> None:
    console.clear()
    console.print(Panel(
        Group(
            Align.center(Text("📊  ТОКЕНЫ", style="bold white")),
            Text(),
            Align.center(Text(
                "короткий, длинный и переполненный диалог — одними и теми же вопросами\n"
                "видно, как растёт цена и что ломается, когда место кончилось",
                style="dim", justify="center")),
        ),
        box=box.DOUBLE, border_style="magenta", padding=(1, 4)))


def plan_panel(base: AgentConfig) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Прогон", style="bold", no_wrap=True)
    table.add_column("Вопросов", justify="right", no_wrap=True)
    table.add_column("Что проверяем")
    table.add_row("Короткий", str(SHORT_STEPS), "точка отсчёта: цена первого вопроса")
    table.add_row("Длинный", str(LONG_STEPS),
                  "во сколько раз дорожает тот же вопрос к концу")
    table.add_row("Сверх лимита", "до отказа", "на каком вопросе агент остановится")
    table.add_row("С обрезкой", "до отказа",
                  "диалог продолжается — но чем за это платит модель")
    return Panel(
        Group(
            Text.from_markup(
                "Окно сужено до [bold]{}[/] токенов вместо {} у модели: упираться "
                "в настоящее окно значило бы потратить деньги ради того же вывода.\n"
                "Неотправленное оценивается токенизатором [bold]{}[/], всё "
                "остальное — точные числа от провайдера.\n".format(
                    fmt(WINDOW), fmt(base.model_info.context_window), tokenizer_name())),
            table),
        title="Что сейчас произойдёт", title_align="left",
        border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def run_panel(run: Run) -> RenderableType:
    blocks: List[RenderableType] = [growth_table(run)]
    blocks.append(outcome_line(run))
    if not run.broke:
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
    table.add_column("Всего", style="bold", justify="right", no_wrap=True)
    table.add_column("Цена шага", justify="right", no_wrap=True)
    table.add_column("Итогом", style="bold", justify="right", no_wrap=True)
    for step in shortened(run.steps):
        if step is None:
            table.add_row("…", Text("пропущено", style="dim"), "…", "…", "…", "…", "…")
            continue
        table.add_row(
            str(step.number), Text(clip(step.question, 38), style="dim"),
            fmt(step.prompt_tokens), fmt(step.completion_tokens), fmt(step.total_tokens),
            format_cost(step.cost, step.currency),
            format_cost(step.total_cost, step.currency))
    return table


def outcome_line(run: Run) -> Text:
    line = Text()
    if run.broke:
        line.append("Сломалось на вопросе {}: ".format(run.asked), style="bold red")
        line.append(run.outcome)
        return line
    line.append("Прошло {} {} из {}: ".format(
        run.answered, plural(run.answered, ("вопрос", "вопроса", "вопросов")),
        run.asked), style="bold green")
    line.append(run.outcome, style="dim")
    return line


def memory_line(run: Run) -> Text:
    """Помнит ли модель имя из первого сообщения — самая простая мера потерь."""
    line = Text()
    line.append("Имя из первого сообщения: ", style="dim")
    if run.remembered is None:
        line.append(run.recall_note, style="dim")
        return line
    line.append("названо верно" if run.remembered else "модель его не знает",
                style="green" if run.remembered else "red")
    if not run.remembered and run.forgotten:
        line.append("  — начало разговора обрезано и в модель больше не уходит",
                    style="dim")
    return line


def summary_panel(runs: List[Run]) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True)
    table.add_column("Прогон", style="bold", no_wrap=True)
    table.add_column("Обменов", justify="right", no_wrap=True)
    table.add_column("Вход", justify="right", no_wrap=True)
    table.add_column("Выход", justify="right", no_wrap=True)
    table.add_column("Всего", style="bold", justify="right", no_wrap=True)
    table.add_column("Цена", justify="right", no_wrap=True)
    table.add_column("Дороже\nпервого", justify="right", no_wrap=True)
    table.add_column("Чем кончился", style="dim")
    for run in runs:
        table.add_row(
            run.title, str(len(run.steps)), fmt(run.prompt_tokens),
            fmt(run.completion_tokens), fmt(run.total_tokens),
            format_cost(run.cost, run.currency),
            "в {:.0f} {}".format(run.growth_ratio,
                                 plural(round(run.growth_ratio), ("раз", "раза", "раз"))),
            "отказ" if run.broke else ("обрезка" if run.forgotten else "дошёл"))
    return Panel(table, title="Сравнение", title_align="left",
                 border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def conclusion_panel(runs: List[Run]) -> RenderableType:
    """Три вывода, каждый — из чисел выше, а не из общих соображений."""
    # Прогоны берутся по своему месту, а не поиском первого сломавшегося:
    # в узком окне ломается и длинный диалог, и выводы съезжали бы на него.
    short, long_run, broken, trimmed = runs

    lines = [
        "[bold]Платят за всю переписку, а не за вопрос.[/] В коротком диалоге "
        "вход первого вопроса — {} токенов, в длинном последний стоит уже {}: "
        "вопросы той же длины, разница в том, что перед ними.".format(
            fmt(short.steps[0].prompt_tokens) if short.steps else "—",
            fmt(long_run.steps[-1].prompt_tokens) if long_run.steps else "—"),
        "[bold]Выход почти не растёт, вход растёт всегда.[/] Получено за длинный "
        "диалог {} токенов против {} отправленных — платит именно история.".format(
            fmt(long_run.completion_tokens), fmt(long_run.prompt_tokens)),
    ]
    if broken is not None:
        lines.append(
            "[bold]При переполнении ломается не ответ, а сам запрос.[/] Диалог "
            "встал на вопросе {}: он не ушёл в модель, ответа нет, деньги за "
            "него не потрачены. Без проверки на этом месте был бы код 400 от "
            "провайдера, из которого не видно ни размера, ни того, что "
            "сокращать.".format(broken.asked))
    memory = ("и имя из первого сообщения модель уже не называет"
              if trimmed.remembered is False else
              "и начало разговора в модель больше не уходит")
    # Про деньги — только там, где они есть: на бесплатном тарифе строка
    # «заплачено больше: бесплатно против бесплатно» выглядела бы насмешкой.
    paid = (" Заплачено при этом больше: {} против {} — разговор просто "
            "продолжался дольше.".format(format_cost(trimmed.cost, trimmed.currency),
                                         format_cost(broken.cost, broken.currency))
            if broken and trimmed.cost and broken.cost else
            " Токенов при этом потрачено больше: {} против {} — разговор просто "
            "продолжался дольше.".format(fmt(trimmed.total_tokens),
                                         fmt(broken.total_tokens)) if broken else "")
    lines.append(
        "[bold]Обрезка спасает диалог ценой памяти.[/] С ней прошло {} {} вместо "
        "{}, но забыто {} {}, {}.{}".format(
            trimmed.answered, plural(trimmed.answered, ("вопрос", "вопроса", "вопросов")),
            broken.answered if broken else "—", trimmed.forgotten,
            plural(trimmed.forgotten, ("сообщение", "сообщения", "сообщений")),
            memory, paid))
    return Panel(Text.from_markup("\n\n".join(lines)), title="Выводы",
                 title_align="left", border_style="green", box=box.ROUNDED,
                 padding=(0, 1))


# Длинный прогон целиком не нужен: важны начало, конец и то, что между ними
# расстояние. Середина сворачивается в одну строку.
HEAD, TAIL = 3, 5


def shortened(steps: List[Step]):
    if len(steps) <= HEAD + TAIL + 1:
        return list(steps)
    return list(steps[:HEAD]) + [None] + list(steps[-TAIL:])


def clip(text: str, width: int) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else line[:width - 1] + "…"


def wait(console: Console, step: bool, hint: str) -> None:
    if not step:
        return
    console.print("[dim]{}[/]".format(hint), end="")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        console.print()


if __name__ == "__main__":
    raise SystemExit(main())
