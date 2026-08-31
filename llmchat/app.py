"""Сценарий работы приложения: выбор модели, ввод ключа и цикл диалога."""
from __future__ import annotations

import argparse
import importlib
import os
from typing import List, Optional

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .client import DemoClient, LLMClient, LLMError
from .session import Session
from .tokens import tokenizer_name
from .ui import (
    HELP_TEXT,
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

def setup(console: Console, demo: bool):
    show_banner(console)
    provider = choose_provider(console)

    if demo:
        console.print()
        console.print(info_panel(
            "Демонстрационный режим: запросы в сеть не уходят, ответы формирует "
            "локальная заглушка. Ключ можно ввести любой.",
            title="Режим проверки", style="yellow",
        ))

    client = None
    while client is None:
        token = ask_token(console, provider, os.environ.get(provider.api_key_env))
        factory = DemoClient if demo else LLMClient
        candidate = factory(provider, token)
        console.print()
        with console.status("[bold]Проверяю ключ…[/]", spinner="dots"):
            try:
                candidate.validate_key()
            except LLMError as exc:
                console.print(error_panel(str(exc)))
                console.print("[dim]Попробуйте ввести ключ ещё раз (Ctrl+C — выход).[/]")
                continue
        console.print(info_panel("[green]Ключ принят.[/]", title="Готово", style="green"))
        client = candidate

    model = choose_model(console, provider)
    return provider, model, client


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
            session.model.id, messages, max_tokens=TOPIC_MAX_TOKENS, temperature=0.3
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
    table.add_row("Окно контекста", "{} токенов".format(fmt(session.context_limit)))
    table.add_row("Резерв под ответ", "{} токенов".format(fmt(session.model.output_reserve)))
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
            command = raw.split()[0].lower()
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
                    session.model.id,
                    session.api_messages(),
                    max_tokens=session.model.output_reserve,
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
        provider, model, client = setup(console, args.demo)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    session = Session(provider=provider, model=model)
    chat_loop(console, client, session)
    return 0
