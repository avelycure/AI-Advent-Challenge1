"""Сценарий работы приложения: выбор модели, ввод ключа и цикл диалога."""
from __future__ import annotations

import argparse
import importlib
from typing import List, Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .client import LLMError, make_client
from .params import SPECS, GenerationParams, apply, format_value, parse_command
from .session import Session
from .tokens import tokenizer_name
from .ui import (
    HELP_TEXT,
    ask_extra_field,
    ask_token,
    choose_model,
    choose_provider,
    error_panel,
    fmt,
    info_panel,
    make_console,
    plural,
    render_frame,
    render_history,
    show_banner,
    warning_panel,
)

TOPIC_MAX_TOKENS = 32
TOPIC_REFRESH_EVERY = 5
TOPIC_SYSTEM = "Ты придумываешь короткие заголовки для диалогов."
TOPIC_REQUEST = (
    "Ниже начало диалога пользователя с ассистентом.\n\n{excerpt}\n\n"
    "Сформулируй тему этого диалога в три-пять слов на языке диалога. "
    "Ответь только темой: без кавычек, пояснений и точки в конце."
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chat.py",
        description="Терминальный чат с LLM (DeepSeek или ChatGPT) с историей и счётчиками токенов.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="проверочный режим без обращения к сети: ответы генерирует локальная заглушка",
    )
    parser.add_argument(
        "--ask-keys",
        action="store_true",
        help="не подхватывать сохранённые реквизиты, спросить их заново",
    )
    return parser.parse_args(argv)


def enable_line_editing() -> None:
    """Стрелки и история ввода в строке запроса, если readline доступен."""
    try:
        # Импорт ради побочного эффекта: readline включает стрелки, Home/End
        # и историю ввода в стандартном input().
        importlib.import_module("readline")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Запуск
# --------------------------------------------------------------------------

def setup(console: Console, demo: bool, ask_keys: bool = False):
    """Провести пользователя по шагам настройки и вернуть готовое состояние."""
    show_banner(console)
    provider = choose_provider(console)

    if demo:
        console.print()
        console.print(info_panel(
            "Демонстрационный режим: запросы в сеть не уходят, ответы формирует "
            "локальная заглушка. Реквизиты можно ввести любые.",
            title="Режим проверки", style="yellow",
        ))

    # У части провайдеров кроме ключа нужен ещё один реквизит — например,
    # каталог Yandex Cloud, без которого не собрать адрес модели.
    step = 2
    extra: Optional[str] = None
    if provider.extra_field is not None:
        extra = ask_extra_field(console, provider, offer_saved=not ask_keys)
        step += 1

    client = None
    while client is None:
        token = ask_token(console, provider, step=step, offer_saved=not ask_keys)
        candidate = make_client(provider, token, demo)
        console.print()
        with console.status("[bold]Проверяю доступ…[/]", spinner="dots"):
            try:
                candidate.validate_key()
            except LLMError as exc:
                console.print(error_panel(str(exc)))
                console.print("[dim]Попробуйте ввести ключ ещё раз (Ctrl+C — выход).[/]")
                continue
        console.print(info_panel("[green]Доступ подтверждён.[/]", title="Готово", style="green"))
        client = candidate

    model = choose_model(console, provider, step=step + 1)
    return provider, model, provider.model_ref(model, extra), client


# --------------------------------------------------------------------------
# Тема диалога
# --------------------------------------------------------------------------

def clean_topic(raw: str) -> str:
    topic = raw.strip().splitlines()[0].strip()
    topic = topic.strip("«»\"'`*# .")
    if len(topic) > 64:
        topic = topic[:61].rstrip() + "…"
    return topic


def update_topic(client, session: Session) -> None:
    """Отдельный дешёвый запрос за темой; его токены идут в общий счёт."""
    excerpt_parts = []
    for message in session.messages[-6:]:
        who = "Пользователь" if message.role == "user" else "Ассистент"
        excerpt_parts.append("{}: {}".format(who, message.content[:400]))
    excerpt = "\n".join(excerpt_parts)

    messages = [
        {"role": "system", "content": TOPIC_SYSTEM},
        {"role": "user", "content": TOPIC_REQUEST.format(excerpt=excerpt)},
    ]
    try:
        completion = client.complete(
            session.model_ref, messages, max_tokens=TOPIC_MAX_TOKENS, temperature=0.3
        )
    except LLMError:
        return  # тема — украшение, из-за неё диалог ломать нельзя
    topic = clean_topic(completion.text)
    if topic:
        session.topic = topic
    session.record_side_usage(completion.prompt_tokens, completion.completion_tokens)


def should_update_topic(session: Session) -> bool:
    return session.exchanges == 1 or session.exchanges % TOPIC_REFRESH_EVERY == 0


# --------------------------------------------------------------------------
# Команды
# --------------------------------------------------------------------------

def stats_panel(session: Session) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Провайдер", session.provider.name)
    table.add_row("Модель", session.model.id)
    if session.model_ref != session.model.id:
        table.add_row("Идентификатор для API", session.model_ref)
    table.add_row("Окно контекста", "{} токенов".format(fmt(session.context_limit)))
    reserve = "{} токенов".format(fmt(session.output_reserve))
    if session.params.max_tokens is not None:
        reserve += " (задан max_tokens)"
    table.add_row("Резерв под ответ", reserve)
    table.add_row("Доступно под историю", "{} токенов".format(fmt(session.input_budget)))
    table.add_row("Занято историей", "{} токенов ({:.0f}% окна)".format(
        fmt(session.context_used()), session.window_ratio() * 100))
    table.add_row("Свободно", "{} токенов".format(fmt(session.free_tokens())))
    table.add_row("Сообщений в истории", str(len(session.messages)))
    table.add_row("Обменов «вопрос — ответ»", str(session.exchanges))
    table.add_row("Средний обмен", "{} токенов".format(fmt(session.avg_exchange_tokens())))
    table.add_row("Хватит ещё на", "≈{} сообщений".format(session.remaining_exchanges()))
    table.add_row("Запросов к API", str(session.requests))
    table.add_row("Токенов отправлено", fmt(session.total_prompt_tokens))
    table.add_row("Токенов получено", fmt(session.total_completion_tokens))
    table.add_row("Всего потрачено", fmt(session.total_tokens))
    table.add_row("Оценка неотправленного", tokenizer_name())
    return Panel(table, title="📊 Статистика сессии", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def params_panel(session: Session) -> RenderableType:
    """Показать действующие параметры генерации и подсказку по команде."""
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Параметр", style="bold")
    table.add_column("Значение", justify="right")
    table.add_column("Что делает", style="dim")

    values = {
        "max_tokens": ("{} (резерв модели)".format(session.output_reserve)
                       if session.params.max_tokens is None
                       else str(session.params.max_tokens)),
        "temperature": format_value(session.params.temperature),
        "top_p": format_value(session.params.top_p),
        "stop": format_value(session.params.stop),
        "response_format": format_value(session.params.response_format),
    }
    default = GenerationParams()
    for name, spec in SPECS.items():
        changed = getattr(session.params, name) != getattr(default, name)
        table.add_row(
            name,
            "[bold yellow]{}[/]".format(values[name]) if changed else "[dim]{}[/]".format(values[name]),
            spec.description,
        )

    hint = Text.from_markup(
        "\n[dim]Изменить:[/] [bold]/change_llm_params max_tokens=200 temperature=0.3[/]\n"
        "[dim]Сбросить:[/] [bold]/reset_llm_params[/]   "
        "[dim]Несколько стоп-строк — через |[/]")
    return Panel(Group(table, hint), title="⚙ Параметры генерации", title_align="left",
                 subtitle="[dim]жёлтым — изменённые[/]", subtitle_align="right",
                 border_style="cyan", box=box.ROUNDED, padding=(0, 1))


def farewell(console: Console, session: Session) -> None:
    console.print()
    console.print(Panel(
        Text.from_markup(
            "Диалог «[bold]{}[/]» завершён.\n"
            "Обменов: [bold]{}[/] · запросов к API: [bold]{}[/] · "
            "потрачено токенов: [bold magenta]{}[/]".format(
                session.topic, session.exchanges, session.requests, fmt(session.total_tokens))
        ),
        title="До встречи 👋", title_align="left", border_style="bright_blue", box=box.ROUNDED,
    ))


# --------------------------------------------------------------------------
# Основной цикл
# --------------------------------------------------------------------------

def handle_params_command(session: Session, argument: str) -> RenderableType:
    """Применить /change_llm_params и вернуть панель с результатом."""
    updates, errors, reset = parse_command(argument)

    if reset:
        session.params = GenerationParams()
        return info_panel("Параметры генерации вернулись к значениям по умолчанию.",
                          title="Сброшено", style="green")

    if not argument.strip():
        return params_panel(session)

    if errors and not updates:
        return error_panel("\n".join(errors))

    session.params = apply(session.params, updates)
    lines = ["Применено: " + ", ".join(
        "[bold]{}[/]=[yellow]{}[/]".format(name, format_value(value))
        for name, value in updates.items())]
    if errors:
        lines.append("[red]Не принято: {}[/]".format("; ".join(errors)))
    lines.append("[dim]Действует для всех следующих запросов.[/]")
    return info_panel("\n".join(lines), title="Параметры генерации", style="cyan")


def chat_loop(console: Console, client, session: Session) -> None:
    notice: Optional[RenderableType] = None

    while True:
        render_frame(console, session, notice)
        notice = None

        try:
            raw = console.input("[bold cyan]Вы ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw:
            continue

        if raw.startswith("/"):
            command, _, argument = raw.partition(" ")
            command = command.lower()
            if command in ("/exit", "/quit", "/q"):
                break
            if command == "/help":
                notice = info_panel(HELP_TEXT, title="Команды")
                continue
            if command == "/stats":
                notice = stats_panel(session)
                continue
            if command == "/history":
                render_history(console, session)
                continue
            if command == "/change_llm_params":
                notice = handle_params_command(session, argument)
                continue
            if command == "/reset_llm_params":
                session.params = GenerationParams()
                notice = info_panel("Параметры генерации вернулись к значениям "
                                    "по умолчанию.", title="Сброшено", style="green")
                continue
            if command == "/new":
                session.reset()
                notice = info_panel("История очищена, контекст свободен.",
                                    title="Новый диалог", style="green")
                continue
            notice = error_panel("Неизвестная команда {}. Наберите /help.".format(command))
            continue

        session.add_user(raw)
        if session.is_full():
            session.drop_last_user()
            notice = error_panel(
                "Контекст заполнен: сообщение не помещается в окно модели. "
                "Начните новый диалог командой /new."
            )
            continue

        render_frame(console, session)
        try:
            with console.status("[bold]{} думает…[/]".format(session.model.id), spinner="dots"):
                completion = client.complete(
                    session.model_ref,
                    session.api_messages(),
                    max_tokens=session.output_reserve,
                    temperature=session.params.temperature,
                    top_p=session.params.top_p,
                    stop=session.params.stop,
                    response_format=session.params.response_format_arg,
                )
        except LLMError as exc:
            session.drop_last_user()
            notice = error_panel(str(exc))
            continue
        except KeyboardInterrupt:
            session.drop_last_user()
            notice = info_panel("Запрос отменён, сообщение не отправлено.",
                                title="Отмена", style="yellow")
            continue

        session.add_assistant(completion.text)
        session.record_main_usage(completion.prompt_tokens, completion.completion_tokens)

        if completion.finish_reason == "length":
            notice = warning_panel(
                "Ответ обрезан: упёрся в max_tokens = {}. Модель не договорила. "
                "Увеличьте лимит командой /change_llm_params max_tokens=… "
                "или сбросьте параметры.".format(fmt(session.output_reserve)))

        if completion.dropped_params:
            notice = warning_panel(
                "Провайдер не принял: {}. Параметр убран из запроса, чтобы диалог "
                "не прервался, но он не действует.".format(
                    ", ".join(completion.dropped_params)))

        if should_update_topic(session):
            with console.status("[dim]Определяю тему диалога…[/]", spinner="dots"):
                update_topic(client, session)

        if session.free_tokens() < session.avg_exchange_tokens() * 2:
            free = session.free_tokens()
            notice = warning_panel(
                "Контекст почти заполнен: свободно {} {}. "
                "Скоро понадобится /new, иначе диалог придётся начать заново.".format(
                    fmt(free), plural(free, ("токен", "токена", "токенов")))
            )

    farewell(console, session)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    enable_line_editing()
    console = make_console()

    try:
        provider, model, model_ref, client = setup(console, args.demo,
                                                   ask_keys=args.ask_keys)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    session = Session(provider=provider, model=model, model_ref=model_ref)
    chat_loop(console, client, session)
    return 0
