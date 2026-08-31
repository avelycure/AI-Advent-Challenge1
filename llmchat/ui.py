"""Отрисовка интерфейса в терминале на rich."""
from __future__ import annotations

from typing import List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .providers import PROVIDER_ORDER, PROVIDERS, ModelInfo, ProviderInfo
from .session import Message, Session

BAR_WIDTH = 26
USER_ACCENT = "cyan"


def make_console() -> Console:
    return Console(highlight=False)


def fmt(number: int) -> str:
    """1234567 -> '1 234 567' (узкие пробелы читаются лучше запятых)."""
    return "{:,}".format(number).replace(",", " ")


# --------------------------------------------------------------------------
# Экран запуска
# --------------------------------------------------------------------------

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


def ask_extra_field(console: Console, provider: ProviderInfo) -> str:
    """Запросить дополнительный реквизит — например, каталог Yandex Cloud."""
    extra = provider.extra_field
    assert extra is not None

    body = Text()
    body.append(extra.help_text + "\n\n")
    body.append(extra.title, style="bold")
    body.append("\n{}".format(extra.hint), style="dim")

    console.print()
    console.print(Panel(body, title="Шаг 2 · Дополнительный реквизит", title_align="left",
                        border_style=provider.accent, box=box.ROUNDED))
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


def ask_token(console: Console, provider: ProviderInfo, env_value: Optional[str],
              step: int = 2) -> str:
    body = Text()
    body.append("Нужен {} ".format(provider.key_phrase))
    body.append(provider.name, style="bold {}".format(provider.accent))
    body.append(".\n\n")
    body.append("Получить его можно здесь:\n")
    body.append("  {}\n".format(provider.token_url), style="bold underline bright_blue")
    for note in provider.notes:
        body.append("\n• {}\n".format(note), style="dim")
    body.append("\nКлюч используется только для запросов к провайдеру, ", style="dim")
    body.append("никуда не сохраняется и при вводе не отображается.", style="dim")

    console.print()
    console.print(Panel(body, title="Шаг {} · {}".format(step, provider.key_title),
                        title_align="left", border_style=provider.accent, box=box.ROUNDED))

    if env_value:
        console.print()
        masked = mask_key(env_value)
        use_env = Confirm.ask(
            "Найден ключ в переменной [bold]{}[/] ({}). Использовать его?".format(
                provider.api_key_env, masked
            ),
            default=True,
        )
        if use_env:
            return env_value

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


def mask_key(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return "{}…{}".format(key[:6], key[-4:])


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
        title="💬 Тема диалога",
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
        "  ·  резерв под ответ {}".format(fmt(session.model.output_reserve)), style="dim"
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

    # Рамка съедает 2 символа, внутренние отступы — ещё 2; между колонками нужен зазор.
    inner = width - 4
    left_need = max(left_top.cell_len, left_bottom.cell_len)
    right_need = max(right_bottom.cell_len, 0)

    right_top = None
    if left_need + 2 + max(right_top_full.cell_len, right_need) <= inner:
        right_top = right_top_full
    elif left_need + 2 + max(right_top_short.cell_len, right_need) <= inner:
        right_top = right_top_short

    if right_top is not None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(ratio=1)  # растягивающийся зазор между колонками
        grid.add_column(justify="right")
        grid.add_row(left_top, "", right_top)
        grid.add_row(left_bottom, "", right_bottom)
        body = grid
        subtitle = "[dim]/help — команды · /new — новый диалог · /exit — выход[/]"
    else:
        # Узкий терминал: колонки рядом не помещаются — выкладываем строками.
        body = Group(left_top, right_top_short, left_bottom, right_bottom)
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
        return Panel(
            Text(message.content),
            title="[bold {}]🧑 Вы[/]".format(USER_ACCENT),
            title_align="left",
            border_style=USER_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    return Panel(
        Markdown(message.content),
        title="[bold {}]🤖 {}[/]".format(session.provider.accent, session.model.id),
        title_align="left",
        border_style=session.provider.accent,
        box=box.ROUNDED,
        padding=(0, 1),
    )


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


HELP_TEXT = (
    "[bold]/help[/]     — эта справка\n"
    "[bold]/history[/]  — показать всю переписку целиком\n"
    "[bold]/stats[/]    — подробная статистика по токенам\n"
    "[bold]/new[/]      — начать диалог заново (история очищается)\n"
    "[bold]/exit[/]     — выход (также Ctrl+D)\n\n"
    "[dim]Любой другой текст отправляется в модель вместе со всей историей диалога.[/]"
)
