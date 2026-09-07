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

from llmagent import (
    DEFAULT_CONFIG,
    Agent,
    AgentConfig,
    AgentError,
    ConfigError,
    InputRejected,
    LLMError,
    MissingCredentials,
    Transport,
)
from llmagent.params import SPECS, GenerationParams, apply, format_value, parse_command
from llmagent.transport import tokenizer_name
from llmagent.usage import JUDGE, MAIN, REPAIR, SIDE

from . import switching
from .session import Session
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
    parser.add_argument(
        "--config",
        metavar="ФАЙЛ",
        help="конфиг агента (JSON или YAML): провайдер, модель, промпт, политики",
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
          store=None, base: Optional[AgentConfig] = None) -> AgentConfig:
    """Провести пользователя по шагам настройки и вернуть конфиг агента.

    Возвращается именно конфиг, а не готовый агент: вызывающий волен создать
    по нему одного агента, а волен и десяток — на том и держится вся затея.

    ``store`` передаёт вызывающий, которому нужен доступ к введённым реквизитам
    после настройки — например, чтобы потом переключать провайдера.
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

    if store is None:
        store = switching.CredentialStore(demo=demo, ask_keys=ask_keys)
    credentials = store.ready_for(console, provider, step=2)

    # Реквизит провайдера занимает отдельный шаг, и выбор модели съезжает на него.
    model_step = 4 if provider.extra_field is not None else 3
    model = choose_model(console, provider, step=model_step)
    return store.config_for(provider, model, credentials.key, credentials.extra, base)


# --------------------------------------------------------------------------
# Тема диалога
# --------------------------------------------------------------------------

def clean_topic(raw: str) -> str:
    topic = raw.strip().splitlines()[0].strip()
    topic = topic.strip("«»\"'`*# .")
    if len(topic) > 64:
        topic = topic[:61].rstrip() + "…"
    return topic


def update_topic(session: Session) -> None:
    """Отдельный дешёвый запрос за темой; его токены идут в общий счёт.

    Идёт через ``ask_messages``, а не мимо агента: тема стоит денег, и её
    расход обязан попасть в тот же счётчик, что и всё остальное.
    """
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
        completion = session.agent.ask_messages(
            messages, kind=SIDE, max_tokens=TOPIC_MAX_TOKENS, temperature=0.3,
            stop=[], response_format={})
    except (LLMError, AgentError):
        return  # тема — украшение, из-за неё диалог ломать нельзя
    topic = clean_topic(completion.text)
    if topic:
        session.topic = topic


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
    breakdown = kind_breakdown(session)
    if breakdown:
        table.add_row("Из них", breakdown)
    table.add_row("Токенов отправлено", fmt(session.total_prompt_tokens))
    table.add_row("Токенов получено", fmt(session.total_completion_tokens))
    if session.total_reasoning_tokens:
        table.add_row("Из них на размышление", "{} (входят в полученные)".format(
            fmt(session.total_reasoning_tokens)))
    table.add_row("Всего потрачено", fmt(session.total_tokens))
    table.add_row("Время в запросах", format_seconds(session.total_seconds))
    table.add_row("В среднем на запрос", format_seconds(session.avg_seconds))
    table.add_row("Цена модели за 1M", price_label(session.provider, session.model))
    if session.model.price_note:
        table.add_row("Оговорка к цене", session.model.price_note)
    table.add_row("Стоимость сессии", session_cost_label(session))
    if session.unpriced_requests:
        table.add_row("Не посчитано", "{} {} — цена модели не задана".format(
            session.unpriced_requests,
            plural(session.unpriced_requests, ("запрос", "запроса", "запросов"))))
    table.add_row("Оценка неотправленного", tokenizer_name())
    return Panel(table, title="📊 Статистика сессии", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def kind_breakdown(session: Session) -> str:
    """Расход по назначению запроса: ответ, переспрос, оценка, служебное.

    Строка появляется, только когда назначений больше одного: в обычном диалоге
    все запросы основные, и повторять это незачем.
    """
    parts = ["{} — {} {}".format(kind, spent.requests,
                                 plural(spent.requests, ("запрос", "запроса", "запросов")))
             for kind, spent in session.agent.usage.by_kind.items() if spent.requests]
    return " · ".join(parts) if len(parts) > 1 else ""


def config_panel(session: Session) -> RenderableType:
    """Действующий конфиг агента целиком — тот самый, что можно унести в файл."""
    import yaml

    body = yaml.safe_dump(session.agent.config.to_dict(), allow_unicode=True,
                          sort_keys=False, default_flow_style=False)
    hint = Text.from_markup(
        "\n[dim]Это и есть «коробка в виде данных»: сохраните вывод в файл и "
        "запустите[/] [cyan]./run.sh --config мой.yaml[/][dim].\n"
        "Ключ намеренно заменён на ***: конфиг задуман переносимым.[/]")
    return Panel(Group(Text(body.rstrip()), hint), title="🧩 Конфиг агента",
                 title_align="left", border_style="green", box=box.ROUNDED, padding=(0, 1))


def price_label(provider, model) -> str:
    """Цена текущей модели: вход и выход за миллион токенов."""
    if provider.free:
        return "бесплатный тариф"
    if not model.priced:
        return "не задана"
    return "{} вход · {} выход".format(format_cost(model.input_price, model.currency),
                                       format_cost(model.output_price, model.currency))


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


def chat_loop(console: Console, store, session: Session) -> None:
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
            if command == "/config":
                notice = config_panel(session)
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
                notice = switching.handle_command(console, store, session, argument)
                continue
            if command == "/retry":
                notice = retry_last(console, session)
                continue
            if command == "/new":
                session.reset()
                notice = info_panel("История очищена, контекст свободен.",
                                    title="Новый диалог", style="green")
                continue
            notice = error_panel("Неизвестная команда {}. Наберите /help.".format(command))
            continue

        try:
            # Входная политика агента срабатывает до показа: в переписке должно
            # оказаться ровно то, что уйдёт в модель.
            prepared = session.agent.prepare_input(raw)
        except InputRejected as exc:
            notice = error_panel("Запрос отклонён входной политикой: {}.".format(exc))
            continue

        session.add_user(prepared)
        if session.is_full():
            session.drop_last_user()
            notice = error_panel(
                "Контекст заполнен: сообщение не помещается в окно модели. "
                "Начните новый диалог командой /new."
            )
            continue

        render_frame(console, session)
        result, notice = request_answer(console, session)
        if result is None:
            session.drop_last_user()
            continue

        if should_update_topic(session):
            with console.status("[dim]Определяю тему диалога…[/]", spinner="dots"):
                update_topic(session)

        if session.free_tokens() < session.avg_exchange_tokens() * 2:
            free = session.free_tokens()
            notice = warning_panel(
                "Контекст почти заполнен: свободно {} {}. "
                "Скоро понадобится /new, иначе диалог придётся начать заново.".format(
                    fmt(free), plural(free, ("токен", "токена", "токенов")))
            )

    farewell(console, session)


def request_answer(console: Console, session: Session, ask=None):
    """Один запрос к агенту: спиннер, разбор ошибок, разбор замечаний.

    Возвращает пару «результат, замечание». Результат равен None, если запрос
    не удался; что делать с историей в этом случае, решает вызывающий.
    """
    ask = ask or session.agent.answer_pending
    try:
        with console.status("[bold]{} думает…[/]".format(session.model.id), spinner="dots"):
            result = ask()
    except (LLMError, AgentError) as exc:
        return None, error_panel(str(exc))
    except KeyboardInterrupt:
        return None, info_panel("Запрос отменён, сообщение не отправлено.",
                                title="Отмена", style="yellow")
    return result, answer_notice(session, result)


def retry_last(console: Console, session: Session) -> Optional[RenderableType]:
    """Задать последний вопрос ещё раз — на той модели, что выбрана сейчас."""
    index = session.last_user_index()
    if index < 0:
        return error_panel("Повторять нечего: в диалоге ещё не было вопросов.")

    render_frame(console, session)
    result, notice = request_answer(console, session, ask=session.agent.retry_last)
    if result is None:
        return notice
    return notice or info_panel(
        "Вопрос задан ещё раз модели [bold]{}[/]. В запрос ушла история по этот "
        "вопрос включительно — прежние ответы на него в неё не попали, поэтому "
        "модели поставлена ровно та же задача.\n"
        "[dim]Ответов на этот вопрос в переписке теперь {}; в следующий запрос "
        "уйдёт только последний.[/]".format(
            session.model.id, len(session.messages) - index - 1),
        title="🔁 Переспрошено", style="cyan")


def answer_notice(session: Session, result) -> Optional[RenderableType]:
    if result.validation is not None and not result.validation.ok:
        return warning_panel(
            "Ответ не прошёл выходную политику за {} {}: {}.".format(
                result.attempts, plural(result.attempts, ("попытку", "попытки", "попыток")),
                result.validation.failure_summary()))
    if result.attempts > 1:
        return info_panel(
            "Форма ответа сошлась не сразу — понадобилось {} {}. "
            "Токены переспросов вошли в общий счёт.".format(
                result.attempts, plural(result.attempts, ("попытка", "попытки", "попыток"))),
            title="Переспрос", style="cyan")
    if result.dropped_params:
        return warning_panel(
            "Провайдер не принял: {}. Параметр убран из запроса, чтобы диалог "
            "не прервался, но он не действует.".format(
                ", ".join(result.dropped_params)))
    if result.notes:
        return info_panel("\n".join(result.notes),
                          title="Запрос подправлен", style="cyan")
    if result.finish_reason == "length":
        return warning_panel(
            "Ответ обрезан: упёрся в max_tokens = {}. Модель не договорила. "
            "Увеличьте лимит командой /change_llm_params max_tokens=… "
            "или сбросьте параметры.".format(fmt(session.output_reserve)))
    return None


def load_base_config(path: Optional[str], demo: bool) -> AgentConfig:
    """Конфиг из файла или встроенный по умолчанию, с поправкой на --demo."""
    base = AgentConfig.from_file(path) if path else DEFAULT_CONFIG
    if demo and not base.transport.demo:
        base = base.with_changes(transport=Transport(demo=True))
    return base


def agent_from_config(console: Console, config: AgentConfig, store) -> Agent:
    """Собрать агента по готовому конфигу, спросив только недостающее.

    Заданный файлом конфиг проходит без единого вопроса — ровно так же, как
    его прошли бы сто конфигов в массовом прогоне. Человека тревожат только
    тогда, когда ключа нет ни в конфиге, ни в окружении, ни в файле.
    """
    agent = Agent(config)
    try:
        agent.credentials
        return agent
    except MissingCredentials as exc:
        console.print()
        console.print(info_panel(
            "{}\n[dim]Введите его — он подставится в конфиг и дальше "
            "спрашиваться не будет.[/]".format(exc),
            title="Нужен реквизит", style="yellow"))
    provider = config.provider_info
    credentials = store.ready_for(console, provider)
    return Agent(config.with_changes(api_key=credentials.key, api_extra=credentials.extra))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    enable_line_editing()
    console = make_console()

    try:
        base = load_base_config(args.config, args.demo)
    except ConfigError as exc:
        console.print(error_panel("Конфиг не принят: {}".format(exc)))
        return 2

    # Демонстрационный режим объявляет и флаг, и конфиг: хранилище реквизитов
    # обязано знать об обоих, иначе оно потребует живой ключ для заглушки.
    store = switching.CredentialStore(demo=base.transport.demo, ask_keys=args.ask_keys)
    try:
        if args.config:
            # Конфиг задан целиком — шаги настройки не нужны, и это главное,
            # что отличает агента от прежнего сценария с вопросами в консоли.
            show_banner(console)
            agent = agent_from_config(console, base, store)
        else:
            agent = Agent(setup(console, args.demo, ask_keys=args.ask_keys,
                                store=store, base=base))
    except ConfigError as exc:
        console.print(error_panel("Конфиг не принят: {}".format(exc)))
        return 2
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    chat_loop(console, store, Session(agent))
    return 0
