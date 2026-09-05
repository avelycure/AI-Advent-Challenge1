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

from . import switching
from .client import LLMError
from .params import SPECS, GenerationParams, apply, format_value, parse_command
from .session import Session
from .tokens import tokenizer_name
from .ui import (
    choose_model,
    format_cost,
    format_seconds,
    session_cost_label,
    choose_provider,
    error_panel,
    fmt,
    help_panel,
    info_panel,
    make_console,
    plural,
    read_user_line,
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

def setup(console: Console, demo: bool, banner=None, ask_keys: bool = False,
          pool=None):
    """Провести пользователя по шагам настройки и вернуть готовое состояние.

    ``pool`` передаёт вызывающий, которому нужен доступ к введённым реквизитам
    после настройки — например, чтобы потом переключать провайдера. Без него
    пул создаётся на один раз, и возвращаемой четвёрки достаточно.
    """
    (banner or show_banner)(console)
    provider = choose_provider(console)

    if demo:
        console.print()
        console.print(info_panel(
            "Демонстрационный режим: запросы в сеть не уходят, ответы формирует "
            "локальная заглушка. Реквизиты можно ввести любые.",
            title="Режим проверки", style="yellow",
        ))

    if pool is None:
        pool = switching.ClientPool(demo=demo, ask_keys=ask_keys)
    ready = pool.ready_for(console, provider, step=2)

    # Реквизит провайдера занимает отдельный шаг, и выбор модели съезжает на него.
    model_step = 4 if provider.extra_field is not None else 3
    model = choose_model(console, provider, step=model_step)
    return provider, model, provider.model_ref(model, ready.extra), ready.client


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
    session.record_side_request(completion)


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
    if session.total_reasoning_tokens:
        table.add_row("Из них на размышление", "{} (входят в полученные)".format(
            fmt(session.total_reasoning_tokens)))
    table.add_row("Всего потрачено", fmt(session.total_tokens))
    table.add_row("Время в запросах", format_seconds(session.total_seconds))
    table.add_row("В среднем на запрос", format_seconds(session.avg_seconds))
    table.add_row("Цена модели за 1M", price_label(session.provider, session.model))
    table.add_row("Стоимость сессии", session_cost_label(session))
    if session.unpriced_requests:
        table.add_row("Не посчитано", "{} {} — цена модели не задана".format(
            session.unpriced_requests,
            plural(session.unpriced_requests, ("запрос", "запроса", "запросов"))))
    table.add_row("Оценка неотправленного", tokenizer_name())
    return Panel(table, title="📊 Статистика сессии", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def price_label(provider, model) -> str:
    """Цена текущей модели: вход и выход за миллион токенов."""
    if provider.free:
        return "бесплатный тариф"
    if not model.priced:
        return "не задана"
    return "{} вход · {} выход".format(format_cost(model.input_price),
                                       format_cost(model.output_price))


def params_panel(session: Session) -> RenderableType:
    """Показать действующие параметры генерации и подсказку по команде."""
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Параметр", style="bold")
    table.add_column("Значение", justify="right")
    table.add_column("Что делает и как задать")

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
        explain = Text(spec.description, style="dim")
        explain.append("\nдопустимо: " + spec.limits, style="dim italic")
        for example in spec.examples:
            explain.append("\n  /change_llm_params " + example, style="cyan")
        table.add_row(
            name,
            "[bold yellow]{}[/]".format(values[name]) if changed else "[dim]{}[/]".format(values[name]),
            explain,
        )

    hint = Text.from_markup(
        "\n[dim]Несколько параметров сразу — через пробел, несколько стоп-строк — через |[/]\n"
        "[dim]Значение с пробелами берите в кавычки:[/] "
        "[cyan]/change_llm_params stop=\"Вопрос пользователя:\"[/]\n"
        "[dim]Сбросить всё:[/] [bold]/reset_llm_params[/]")
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

PARAMS_HINT = ("\n[dim]Список параметров с примерами: [/][bold]/help[/][dim] "
               "или [/][bold]/change_llm_params[/][dim] без аргументов.[/]")


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
        return error_panel("\n".join(errors) + PARAMS_HINT)

    session.params = apply(session.params, updates)
    lines = ["Применено: " + ", ".join(
        "[bold]{}[/]=[yellow]{}[/]".format(name, format_value(value))
        for name, value in updates.items())]
    if errors:
        lines.append("[red]Не принято: {}[/]{}".format("; ".join(errors), PARAMS_HINT))
    lines.append("[dim]Действует для всех следующих запросов.[/]")
    return info_panel("\n".join(lines), title="Параметры генерации", style="cyan")


def chat_loop(console: Console, pool, session: Session) -> None:
    notice: Optional[RenderableType] = None

    while True:
        render_frame(console, session, notice)
        notice = None

        try:
            raw = read_user_line(console, "[bold cyan]Вы ›[/] ")
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
                notice = help_panel()
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
            if command in ("/change_model", "/model", "/models"):
                notice = switching.handle_command(console, pool, session, argument)
                continue
            if command == "/retry":
                notice = retry_last(console, pool, session)
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
        completion, notice = request_answer(console, pool, session, session.api_messages())
        if completion is None:
            session.drop_last_user()
            continue

        if should_update_topic(session):
            with console.status("[dim]Определяю тему диалога…[/]", spinner="dots"):
                update_topic(pool.current, session)

        if session.free_tokens() < session.avg_exchange_tokens() * 2:
            free = session.free_tokens()
            notice = warning_panel(
                "Контекст почти заполнен: свободно {} {}. "
                "Скоро понадобится /new, иначе диалог придётся начать заново.".format(
                    fmt(free), plural(free, ("токен", "токена", "токенов")))
            )

    farewell(console, session)


def request_answer(console: Console, pool, session: Session, messages):
    """Один запрос к модели: спиннер, разбор ошибок, разбор замечаний.

    Возвращает пару «ответ, замечание». Ответ равен None, если запрос не удался;
    что делать с историей в этом случае, решает вызывающий.
    """
    try:
        with console.status("[bold]{} думает…[/]".format(session.model.id), spinner="dots"):
            completion = pool.current.complete(
                session.model_ref,
                messages,
                max_tokens=session.output_reserve,
                temperature=session.params.temperature,
                top_p=session.params.top_p,
                stop=session.params.stop,
                response_format=session.params.response_format_arg,
            )
    except LLMError as exc:
        return None, error_panel(str(exc))
    except KeyboardInterrupt:
        return None, info_panel("Запрос отменён, сообщение не отправлено.",
                                title="Отмена", style="yellow")

    session.record_answer(completion)
    return completion, answer_notice(session, completion)


def retry_last(console: Console, pool, session: Session) -> Optional[RenderableType]:
    """Задать последний вопрос ещё раз — на той модели, что выбрана сейчас."""
    index = session.last_user_index()
    if index < 0:
        return error_panel("Повторять нечего: в диалоге ещё не было вопросов.")

    render_frame(console, session)
    completion, notice = request_answer(console, pool, session,
                                        session.api_messages_upto(index))
    if completion is None:
        return notice
    return notice or info_panel(
        "Вопрос задан ещё раз модели [bold]{}[/]. В запрос ушла история по этот "
        "вопрос включительно — прежние ответы на него в неё не попали, поэтому "
        "модели поставлена ровно та же задача.\n"
        "[dim]Ответов на этот вопрос в переписке теперь {}; в следующий запрос "
        "уйдёт только последний.[/]".format(
            session.model.id, len(session.messages) - index - 1),
        title="🔁 Переспрошено", style="cyan")


def answer_notice(session: Session, completion) -> Optional[RenderableType]:
    if completion.dropped_params:
        return warning_panel(
            "Провайдер не принял: {}. Параметр убран из запроса, чтобы диалог "
            "не прервался, но он не действует.".format(
                ", ".join(completion.dropped_params)))
    if completion.notes:
        return info_panel("\n".join(completion.notes),
                          title="Запрос подправлен", style="cyan")
    if completion.finish_reason == "length":
        return warning_panel(
            "Ответ обрезан: упёрся в max_tokens = {}. Модель не договорила. "
            "Увеличьте лимит командой /change_llm_params max_tokens=… "
            "или сбросьте параметры.".format(fmt(session.output_reserve)))
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    enable_line_editing()
    console = make_console()

    pool = switching.ClientPool(demo=args.demo, ask_keys=args.ask_keys)
    try:
        provider, model, model_ref, _ = setup(console, args.demo,
                                              ask_keys=args.ask_keys, pool=pool)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    session = Session(provider=provider, model=model, model_ref=model_ref)
    chat_loop(console, pool, session)
    return 0
