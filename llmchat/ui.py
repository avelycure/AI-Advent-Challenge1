"""Отрисовка интерфейса в терминале на rich."""
from __future__ import annotations

import os
import re
import sys
import termios
from typing import List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from llmagent import Message
from llmagent.params import SPECS
from llmagent.transport import (
    PROVIDER_ORDER,
    PROVIDERS,
    ModelInfo,
    ProviderInfo,
    find_sources,
    mask,
)

from .session import Session

BAR_WIDTH = 26
USER_ACCENT = "cyan"


def make_console() -> Console:
    return Console(highlight=False)


def read_user_line(console: Console, prompt: str) -> str:
    """Прочитать реплику пользователя, не разрывая вставку из буфера обмена.

    Терминал отдаёт вставленный текст как обычный набор, поэтому перевод строки
    внутри вставки для ``input()`` неотличим от нажатия Enter: многострочная
    вставка уходила в модель отдельным запросом на каждую строку.
    """
    _plain_paste()
    first = console.input(prompt)
    tail = _pasted_tail()
    whole = first + "\n" + tail if tail else first
    whole = PASTE_MARKERS.sub("", whole)
    return "\n".join(line.rstrip("\r") for line in whole.split("\n")).strip()


# Обрамление «скобочной вставки». Программа её отключает, но терминал может
# прислать обрамление и без спроса — тогда оно попадёт в текст сообщения.
PASTE_MARKERS = re.compile(r"\x1b\[20[01]~")

# Сколько десятых долей секунды ждать продолжения вставки, прежде чем считать
# её законченной. Ожидание обязательно: входной буфер терминала меньше длинной
# вставки, поэтому её хвост дописывается только после того, как буфер
# освободится — то есть уже после первого чтения.
PASTE_IDLE_TENTHS = 1


def _plain_paste() -> None:
    """Отключить «скобочную вставку»: libedit на macOS её не понимает.

    Терминал обрамляет вставку управляющими последовательностями, а libedit
    съедает только их начало и оставляет в строке хвосты «200~» и «201~».
    """
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()


def _pasted_tail() -> str:
    """Остаток вставки, пришедший тем же залпом, что и первая строка.

    Читать его обычным способом нельзя: в каноническом режиме терминал
    придерживает незавершённую строку до Enter, и она не видна ни ``select``,
    ни ``read``. Поэтому режим на мгновение снимается — тогда всё, что уже
    лежит во входной очереди, читается сразу.

    Успеть набрать это вручную за время между Enter и чтением человек не может,
    так что остаток — всегда вставка.
    """
    if not sys.stdin.isatty():
        return ""
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return ""

    raw = termios.tcgetattr(fd)
    raw[3] &= ~termios.ICANON
    raw[6][termios.VMIN] = 0
    raw[6][termios.VTIME] = PASTE_IDLE_TENTHS
    chunks = []
    try:
        termios.tcsetattr(fd, termios.TCSANOW, raw)
        while True:
            data = os.read(fd, 4096)
            if not data:
                break
            chunks.append(data)
    except OSError:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
    return b"".join(chunks).decode("utf-8", "replace")


def fmt(number: int) -> str:
    """1234567 -> '1 234 567' (узкие пробелы читаются лучше запятых)."""
    return "{:,}".format(number).replace(",", " ")


# --------------------------------------------------------------------------
# Экран запуска
# --------------------------------------------------------------------------

def format_seconds(seconds: float) -> str:
    if seconds < 10:
        return "{:.1f} с".format(seconds)
    if seconds < 60:
        return "{:.0f} с".format(seconds)
    return "{:.0f} мин {:02.0f} с".format(seconds // 60, seconds % 60)


CURRENCY_SIGNS = {"USD": "$", "RUB": "₽"}


def format_cost(amount: Optional[float], currency: str = "USD") -> str:
    """Цена в валюте прайса. Прочерк означает, что цена модели не задана."""
    if amount is None:
        return "—"
    if amount == 0:
        return "бесплатно"
    sign = CURRENCY_SIGNS.get(currency, currency + " ")
    if amount < 0.000001:
        # Иначе округление до шести знаков даёт «$0.» и выглядит как ошибка.
        return "<{}0.000001".format(sign)
    if amount < 0.001:
        return "{}{}".format(sign, "{:.6f}".format(amount).rstrip("0"))
    if amount < 1:
        # Хвостовые нули срезаются, но два знака остаются: иначе цена прайса
        # выглядит рвано — «$0.7500» рядом с «$4.50».
        digits = "{:.4f}".format(amount).rstrip("0")
        whole, _, fraction = digits.partition(".")
        return "{}{}.{}".format(sign, whole, fraction.ljust(2, "0"))
    return "{}{:.2f}".format(sign, amount)


def session_cost_label(session: Session) -> str:
    """Итог по сессии. «≥» означает, что часть запросов посчитать не удалось."""
    if session.requests == 0:
        return "—"
    parts = [format_cost(value, currency)
             for currency, value in sorted(session.total_costs.items())]
    if not parts:
        return "—"
    label = " + ".join(parts)
    return "≥ " + label if session.unpriced_requests else label


def show_banner(console: Console) -> None:
    console.clear()
    title = Text("✨  LLM CHAT", style="bold white")
    subtitle = Text(
        "терминальный клиент к DeepSeek, ChatGPT, YandexGPT и GigaChat\n"
        "полная история диалога · счётчики токенов · тема разговора в шапке",
        style="dim",
    )
    console.print(
        Panel(
            Group(Align.center(title), Text(), Align.center(subtitle)),
            box=box.DOUBLE,
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()


def choose_provider(console: Console) -> ProviderInfo:
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False, expand=False)
    table.add_column("№", style="bold", justify="right")
    table.add_column("Провайдер", style="bold")
    table.add_column("Модели", style="dim")
    table.add_column("Что нужно для доступа", style="dim")

    for index, key in enumerate(PROVIDER_ORDER, start=1):
        provider = PROVIDERS[key]
        if provider.oauth is not None:
            access = "ключ авторизации (обмен по OAuth)"
        elif provider.extra_field is not None:
            access = "API-ключ и {}".format(provider.extra_field.title.split(" (")[0].lower())
        else:
            access = "API-ключ"
        table.add_row(
            str(index),
            Text(provider.name, style=provider.accent),
            ", ".join(model.id for model in provider.models),
            access,
        )

    console.print(Panel(table, title="Шаг 1 · Выберите LLM", title_align="left",
                        border_style="bright_blue", box=box.ROUNDED))
    choice = Prompt.ask(
        "\n[bold]Номер провайдера[/]",
        choices=[str(i) for i in range(1, len(PROVIDER_ORDER) + 1)],
        default="1",
    )
    return PROVIDERS[PROVIDER_ORDER[int(choice) - 1]]


def choose_source(console: Console, sources, what: str, secret: bool) -> Optional[str]:
    """Предложить готовое значение из переменной окружения или файла.

    Возвращает None, если подходящего источника нет или пользователь отказался —
    тогда значение спрашивается вручную.
    """
    if not sources:
        return None

    def shown(source):
        return mask(source.value) if secret else source.value

    if len(sources) == 1:
        source = sources[0]
        console.print()
        if source.warning:
            console.print("[yellow]Внимание: {}[/]".format(source.warning))
        use = Confirm.ask(
            "Найден {}: [bold]{}[/] ([dim]{}[/]). Использовать?".format(
                what, source.label, shown(source)),
            default=True)
        return source.value if use else None

    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("№", style="bold", justify="right")
    table.add_column("Источник")
    table.add_column("Значение", style="dim")
    for index, source in enumerate(sources, start=1):
        table.add_row(str(index), source.label, shown(source))
    table.add_row("0", "ввести вручную", "")

    console.print()
    console.print(Panel(table, title="Найдено несколько источников", title_align="left",
                        border_style="bright_blue", box=box.ROUNDED))
    for source in sources:
        if source.warning:
            console.print("[yellow]Внимание: {}[/]".format(source.warning))
    choice = Prompt.ask("\n[bold]Номер источника[/]",
                        choices=[str(i) for i in range(0, len(sources) + 1)], default="1")
    return None if choice == "0" else sources[int(choice) - 1].value


def ask_extra_field(console: Console, provider: ProviderInfo, offer_saved: bool = True,
                    title: Optional[str] = None) -> str:
    """Запросить дополнительный реквизит — например, каталог Yandex Cloud."""
    extra = provider.extra_field
    assert extra is not None

    body = Text()
    body.append(extra.help_text + "\n\n")
    body.append(extra.title, style="bold")
    body.append("\n{}".format(extra.hint), style="dim")

    console.print()
    console.print(Panel(body, title=title or "Шаг 2 · Дополнительный реквизит",
                        title_align="left",
                        border_style=provider.accent, box=box.ROUNDED))

    if offer_saved:
        saved = choose_source(console, find_sources(None, extra.files),
                              what=extra.title.split(" (")[0].lower(), secret=False)
        if saved:
            return saved

    while True:
        console.print()
        value = Prompt.ask("[bold]{}[/]".format(extra.title)).strip()
        if not value:
            console.print("[red]Значение не может быть пустым.[/]")
            continue
        if not value.isascii():
            console.print("[red]Значение должно состоять из латинских букв и цифр.[/]")
            continue
        return value


def choose_model(console: Console, provider: ProviderInfo, step: int = 3) -> ModelInfo:
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("№", style="bold", justify="right")
    table.add_column("Модель", style="bold")
    table.add_column("Описание", style="dim")
    table.add_column("Контекст", style="dim", justify="right")

    for index, model in enumerate(provider.models, start=1):
        table.add_row(str(index), model.id, model.label, fmt(model.context_window))

    console.print()
    console.print(Panel(table, title="Шаг {} · Выберите модель".format(step), title_align="left",
                        border_style=provider.accent, box=box.ROUNDED))
    choice = Prompt.ask(
        "\n[bold]Номер модели[/]",
        choices=[str(i) for i in range(1, len(provider.models) + 1)],
        default="1",
    )
    return provider.models[int(choice) - 1]


def ask_token(console: Console, provider: ProviderInfo, step: int = 2,
              offer_saved: bool = True, title: Optional[str] = None) -> str:
    body = Text()
    body.append("Нужен {} ".format(provider.key_phrase))
    body.append(provider.name, style="bold {}".format(provider.accent))
    body.append(".\n\n")
    body.append("Получить его можно здесь:\n")
    body.append("  {}\n".format(provider.token_url), style="bold underline bright_blue")
    for note in provider.notes:
        body.append("\n• {}\n".format(note), style="dim")
    body.append("\nКлюч используется только для запросов к провайдеру, ", style="dim")
    body.append("никуда не сохраняется и при вводе не отображается.\n", style="dim")
    if provider.key_files:
        body.append("Чтобы не вводить его каждый раз, положите ключ в {} "
                    "или задайте переменную {}.".format(
                        provider.key_files[0], provider.api_key_env), style="dim")

    console.print()
    console.print(Panel(body,
                        title=title or "Шаг {} · {}".format(step, provider.key_title),
                        title_align="left", border_style=provider.accent, box=box.ROUNDED))

    if offer_saved:
        saved = choose_source(console, find_sources(provider.api_key_env, provider.key_files),
                              what=provider.key_phrase, secret=True)
        if saved:
            return saved

    while True:
        console.print()
        token = Prompt.ask(
            "[bold]Вставьте {}[/] [dim]({})[/]".format(provider.key_phrase,
                                                       provider.key_hint),
            password=True,
        ).strip()
        if not token:
            console.print("[red]Ключ не может быть пустым.[/]")
            continue
        if not token.isascii():
            # Кириллическая «с» вместо латинской ломает HTTP-заголовок,
            # а ошибка при этом выглядит непонятно — ловим сразу.
            console.print("[red]В ключе есть символы вне латиницы. Похоже, при копировании "
                          "попала кириллическая буква — скопируйте ключ заново.[/]")
            continue
        return token


# --------------------------------------------------------------------------
# Основной кадр диалога
# --------------------------------------------------------------------------

def build_header(session: Session) -> RenderableType:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(
        Text(session.topic, style="bold white"),
        Text(
            "{} · {}".format(session.provider.name, session.model.id),
            style="dim {}".format(session.provider.accent),
        ),
    )
    return Panel(
        grid,
        title="💬 Сессия {}".format(session.agent.session_id),
        title_align="left",
        border_style=session.provider.accent,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _bar(ratio: float, color: str, width: int = BAR_WIDTH) -> Text:
    filled = min(width, max(0, int(round(ratio * width))))
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey35")
    return bar


def plural(number: int, forms) -> str:
    """forms = ('сообщение', 'сообщения', 'сообщений')"""
    n = abs(number) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def build_stats(session: Session, width: int) -> RenderableType:
    used = session.context_used()
    # Полоса — доля всего окна модели; цвет — по давлению на доступный бюджет,
    # чтобы предупреждение краснело раньше, чем окно физически кончится.
    ratio = session.window_ratio()
    pressure = session.fill_ratio()
    color = "green" if pressure < 0.6 else ("yellow" if pressure < 0.85 else "red")
    bar_width = max(8, min(BAR_WIDTH, width // 4))

    left_top = Text()
    left_top.append("Контекст ", style="bold")
    left_top.append_text(_bar(ratio, color, bar_width))
    left_top.append("  {:>3.0f}%".format(ratio * 100), style="bold {}".format(color))

    right_top_short = Text(
        "{} / {} токенов".format(fmt(used), fmt(session.context_limit)), style="dim"
    )
    right_top_full = right_top_short.copy()
    right_top_full.append(
        "  ·  резерв под ответ {}".format(fmt(session.output_reserve)), style="dim"
    )

    left_bottom = Text()
    left_bottom.append("Свободно ", style="bold")
    left_bottom.append(fmt(session.free_tokens()), style=color)
    left_bottom.append("  ·  до конца ", style="bold")
    if session.exchanges == 0:
        left_bottom.append("оценка после ответа", style="dim")
    else:
        remaining = session.remaining_exchanges()
        left_bottom.append(
            "≈{} {}".format(remaining, plural(remaining, ("сообщение", "сообщения", "сообщений"))),
            style=color,
        )

    right_bottom = Text()
    right_bottom.append("потрачено ", style="bold")
    right_bottom.append(fmt(session.total_tokens), style="magenta")
    right_bottom.append(" {} · {} {}".format(
        plural(session.total_tokens, ("токен", "токена", "токенов")),
        session.requests,
        plural(session.requests, ("запрос", "запроса", "запросов")),
    ), style="dim")
    right_bottom.append(" · ", style="grey35")
    right_bottom.append(session_cost_label(session), style="magenta")

    # Рамка съедает 2 символа, внутренние отступы — ещё 2; между колонками нужен зазор.
    inner = width - 4
    left_need = max(left_top.cell_len, left_bottom.cell_len)
    right_need = max(right_bottom.cell_len, 0)

    right_top = None
    if left_need + 2 + max(right_top_full.cell_len, right_need) <= inner:
        right_top = right_top_full
    elif left_need + 2 + max(right_top_short.cell_len, right_need) <= inner:
        right_top = right_top_short

    changed = session.params.changed_fields()
    params_line = None
    if changed:
        params_line = Text()
        params_line.append("Параметры: ", style="bold")
        params_line.append(" · ".join("{}={}".format(name, value) for name, value in changed),
                           style="yellow")

    if right_top is not None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(ratio=1)  # растягивающийся зазор между колонками
        grid.add_column(justify="right")
        grid.add_row(left_top, "", right_top)
        grid.add_row(left_bottom, "", right_bottom)
        body = Group(grid, params_line) if params_line else grid
        subtitle = "[dim]/help — команды · /new — новый диалог · /exit — выход[/]"
    else:
        # Узкий терминал: колонки рядом не помещаются — выкладываем строками.
        parts = [left_top, right_top_short, left_bottom, right_bottom]
        if params_line:
            parts.append(params_line)
        body = Group(*parts)
        subtitle = "[dim]/help · /new · /exit[/]"

    return Panel(
        body,
        box=box.ROUNDED,
        border_style="grey42",
        padding=(0, 1),
        subtitle=subtitle,
        subtitle_align="right",
    )


def build_message(message: Message, session: Session) -> RenderableType:
    if message.role == "user":
        if message.note:
            # Заметку не задавал человек — это принятый в память итог
            # под-агента, и подписывать её «Вы» было бы неправдой.
            return Panel(
                Text(message.content, style="dim"),
                title="[bold green]⤷ В памяти: итог под-агента {}[/]".format(message.model),
                title_align="left",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        return Panel(
            Text(message.content),
            title="[bold {}]🧑 Вы[/]".format(USER_ACCENT),
            title_align="left",
            border_style=USER_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    accent = message.accent or session.provider.accent
    return Panel(
        Markdown(message.content),
        title="[bold {}]🤖 {}[/]".format(accent, message.model or session.model.id),
        title_align="left",
        subtitle=answer_metrics(message),
        subtitle_align="right",
        border_style=accent,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def answer_metrics(message: Message) -> Optional[Text]:
    """Подпись под ответом: чего стоил именно этот запрос."""
    if not message.completion_tokens and not message.elapsed:
        return None
    line = Text(no_wrap=True)
    line.append(format_seconds(message.elapsed), style="dim")
    line.append(" · ", style="grey35")
    line.append("{} → {} ток.".format(fmt(message.prompt_tokens),
                                      fmt(message.completion_tokens)), style="dim")
    if message.reasoning_tokens:
        line.append(" · ", style="grey35")
        line.append("размышление {}".format(fmt(message.reasoning_tokens)), style="yellow")
    line.append(" · ", style="grey35")
    line.append(format_cost(message.cost, message.cost_currency), style="magenta")
    return line


def _height(console: Console, renderable: RenderableType) -> int:
    options = console.options.update(height=None)
    return len(console.render_lines(renderable, options, pad=False))


def _empty_hint() -> RenderableType:
    hint = Text()
    hint.append("Диалог пуст. ", style="bold")
    hint.append("Напишите первый вопрос — вся переписка будет отправляться\n"
                "в модель целиком, поэтому она помнит контекст разговора.", style="dim")
    return Panel(hint, box=box.ROUNDED, border_style="grey42", padding=(1, 2))


def render_frame(console: Console, session: Session, notice: Optional[RenderableType] = None) -> None:
    """Перерисовать экран: шапка сверху, история по центру, счётчики снизу."""
    console.clear()
    header = build_header(session)
    stats = build_stats(session, console.width)

    # 3 строки резерва: пустая строка, строка ввода и запас на перенос.
    budget = console.height - _height(console, header) - _height(console, stats) - 3
    if notice is not None:
        budget -= _height(console, notice) + 1

    blocks: List[RenderableType] = []
    used_lines = 0
    shown = 0
    if session.messages:
        for message in reversed(session.messages):
            block = build_message(message, session)
            block_height = _height(console, block)
            if blocks and used_lines + block_height > budget:
                break
            blocks.append(block)
            used_lines += block_height
            shown += 1
        blocks.reverse()
    hidden = len(session.messages) - shown

    console.print(header)
    if hidden > 0:
        console.print(
            Align.center(
                Text("↑ выше ещё {} {} — команда /history покажет всё".format(
                    hidden, plural(hidden, ("сообщение", "сообщения", "сообщений"))),
                    style="dim italic")
            )
        )
    if not session.messages:
        console.print(_empty_hint())
    for block in blocks:
        console.print(block)
    if notice is not None:
        console.print(notice)
    console.print(stats)


def render_history(console: Console, session: Session) -> None:
    console.clear()
    console.print(build_header(session))
    if not session.messages:
        console.print(_empty_hint())
    for message in session.messages:
        console.print(build_message(message, session))
    console.print(build_stats(session, console.width))
    console.input("\n[dim]Enter — вернуться в диалог[/] ")


def info_panel(text: str, title: str = "Информация", style: str = "bright_blue") -> RenderableType:
    return Panel(Text.from_markup(text), title=title, title_align="left",
                 border_style=style, box=box.ROUNDED, padding=(0, 1))


def error_panel(text: str) -> RenderableType:
    return info_panel("[red]{}[/]".format(text), title="⚠ Ошибка", style="red")


def warning_panel(text: str) -> RenderableType:
    return info_panel("[yellow]{}[/]".format(text), title="⚠ Внимание", style="yellow")


COMMANDS: List[tuple] = [
    ("/help", "эта справка"),
    ("/history", "показать всю переписку целиком"),
    ("/stats", "подробная статистика по токенам"),
    ("/config", "конфиг агента целиком — его можно сохранить и запустить с --config"),
    ("/agent", "вызвать под-агента с нужным конфигом; без аргументов — список конфигов"),
    ("/change_llm_params", "изменить параметры генерации; без аргументов — "
                           "таблица с текущими значениями"),
    ("/reset_llm_params", "вернуть параметры к значениям по умолчанию"),
    ("/change_model", "сменить модель или провайдера; без аргументов — список с номерами"),
    ("/retry", "переспросить последний вопрос на текущей модели, не меняя историю"),
    ("/new", "начать диалог заново (история очищается)"),
    ("/exit", "выход (также Ctrl+D)"),
]

COMBINE_HINTS: List[tuple] = [
    ("/change_llm_params max_tokens=200 temperature=0.3", "несколько параметров сразу, через пробел"),
    ("/change_llm_params stop=Вопрос:|Ответ:", "несколько стоп-строк, через |"),
    ('/change_llm_params stop="Вопрос пользователя:"', "значение с пробелами — в кавычках"),
    ("/change_llm_params reset", "то же, что /reset_llm_params"),
    ("/change_model 7", "переключиться на модель под номером 7 из списка"),
    ("/change_model gpt-5.5", "то же самое по имени модели"),
    ("/change_model openai gpt-5.5", "если имя есть у нескольких провайдеров"),
]


def help_panel() -> RenderableType:
    """Справка по командам и параметрам генерации с примерами записи."""
    return Panel(
        Group(_commands_table(), _params_table(), _combine_table(),
              Text.from_markup(
                  "\n[dim]Любой другой текст отправляется в модель "
                  "вместе со всей историей диалога.[/]")),
        title="Команды", title_align="left",
        border_style="bright_blue", box=box.ROUNDED, padding=(0, 1))


def _commands_table() -> Table:
    table = _help_table()
    table.add_column("Команда", style="bold", no_wrap=True)
    table.add_column("Действие")
    for name, description in COMMANDS:
        table.add_row(name, Text(description, style="dim"))
    return table


def _params_table() -> Table:
    """Параметры перечисляются по их же описаниям в коде, чтобы не разъезжались."""
    table = _help_table(title="Параметры генерации — задаются как имя=значение")
    table.add_column("Как записать", style="cyan", no_wrap=True)
    table.add_column("Что делает")
    table.add_column("Допустимо", style="italic")
    for spec in SPECS.values():
        table.add_row(spec.sample, Text(spec.description, style="dim"),
                      Text(spec.limits, style="dim"))
    return table


def _combine_table() -> Table:
    table = _help_table(title="Как комбинировать")
    table.add_column("Пример", style="cyan", no_wrap=True)
    table.add_column("Что получится")
    for example, meaning in COMBINE_HINTS:
        table.add_row(example, Text(meaning, style="dim"))
    return table


def _help_table(title: Optional[str] = None) -> Table:
    return Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True,
                 title=title, title_justify="left", title_style="bold")
