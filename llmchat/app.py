"""Сценарий работы приложения: выбор модели, ввод ключа и цикл диалога."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from typing import List, Optional

from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from llmagent import (
    DEFAULT_CONFIG,
    SessionNotFound,
    Agent,
    AgentConfig,
    AgentError,
    ConfigError,
    ContextOverflow,
    InputRejected,
    LLMError,
    MissingCredentials,
    MemoryStore,
    SessionRecord,
    SessionStore,
    TASK_STAGES,
    Transport,
    find_model,
    overrides,
    restore_agent,
)
from llmagent.config import (
    FACTS as STRATEGY_FACTS,
    FULL as STRATEGY_FULL,
    STRATEGIES,
    SUMMARY as STRATEGY_SUMMARY,
)
from llmagent.params import SPECS, GenerationParams, apply, format_value, parse_command
from llmagent.transport import tokenizer_name
from llmagent.usage import JUDGE, MAIN, REPAIR, SIDE

from . import subagent, switching
from .session import DEFAULT_TOPIC, Session
from .ui import (
    branches_panel,
    choose_model,
    facts_panel,
    invariants_panel,
    memory_panel,
    task_panel,
    strategy_panel,
    format_cost,
    format_seconds,
    session_cost_label,
    choose_provider,
    error_panel,
    fmt,
    help_panel,
    info_panel,
    enable_bracketed_paste,
    make_console,
    plural,
    read_user_line,
    render_frame,
    render_history,
    show_banner,
    tokens_panel,
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
        help="конфиг агента: короткое имя из configs/ или путь к файлу JSON либо YAML",
    )
    parser.add_argument(
        "--ask",
        metavar="ТЕКСТ",
        help="задать один вопрос и выйти, без диалога и без вопросов в консоль",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="с --ask: вывести итог одним объектом JSON и ничего больше",
    )

    tune = parser.add_argument_group(
        "правки конфига",
        "накрывают конфиг сверху: названное поле меняется, остальные остаются")
    tune.add_argument("--model", metavar="ИМЯ",
                      help="модель; провайдер определится сам, либо запись провайдер/модель")
    tune.add_argument("--provider", metavar="КЛЮЧ", help="провайдер")
    tune.add_argument("--temperature", type=float, metavar="ЧИСЛО",
                      help="разброс ответов: ниже — предсказуемее")
    tune.add_argument("--max-tokens", type=int, metavar="ЧИСЛО",
                      help="предел длины ответа")
    tune.add_argument("--system-prompt", metavar="ТЕКСТ",
                      help="кем агент себя считает и как отвечает")
    tune.add_argument("--append-system-prompt", metavar="ТЕКСТ",
                      help="дописать к действующему системному промпту")
    tune.add_argument("--format", metavar="ВИД",
                      help="требуемая форма ответа: free, json, yaml или md")
    tune.add_argument("--max-cost", type=float, metavar="СУММА",
                      help="потолок трат за сессию")
    tune.add_argument("--no-history", action="store_true",
                      help="без истории диалога; профиль пользователя сохраняется")
    tune.add_argument("--strategy", metavar="ВИД", choices=STRATEGIES,
                      help="как собирается запрос: {} (по умолчанию {})".format(
                          ", ".join(STRATEGIES), STRATEGY_SUMMARY))
    tune.add_argument("--no-compress", action="store_true",
                      help="то же, что --strategy full: в модель уходит вся переписка")
    tune.add_argument("--compress", action="store_true",
                      help="то же, что --strategy summary (так и работает по умолчанию)")
    tune.add_argument("--keep-last", type=int, metavar="ЧИСЛО",
                      help="сколько последних сообщений уходит в модель дословно")
    tune.add_argument("--compress-every", type=int, metavar="ЧИСЛО",
                      help="сколько несжатых сообщений копить до пересказа")
    tune.add_argument("--agent-name", metavar="ИМЯ",
                      help="подпись агента (видна в панелях под-агентов)")
    tune.add_argument("--set", action="append", default=[], metavar="ПОЛЕ=ЗНАЧЕНИЕ",
                      help="любое поле конфига по полному пути; можно повторять")
    tune.add_argument("--show-config", action="store_true",
                      help="напечатать собранный конфиг и выйти")

    live = parser.add_argument_group("сессии")
    live.add_argument("--name", metavar="ИМЯ",
                      help="имя сессии: по нему её видно в списке и можно вернуться")
    live.add_argument("--session-id", metavar="ID", help="свой идентификатор сессии")
    live.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                      help="продолжить самую свежую сохранённую сессию")
    live.add_argument("-r", "--resume", nargs="?", const="", metavar="НОМЕР|ИМЯ|ID",
                      help="вернуться в сессию; без значения предложит выбрать")
    live.add_argument("--sessions", nargs="?", const="", metavar="СЛОВО",
                      help="показать сохранённые сессии; со словом — только те, "
                           "где оно встречается")
    live.add_argument("--rm-session", metavar="НОМЕР|ИМЯ|ID",
                      help="удалить сессию; all — удалить все")
    live.add_argument("--no-save", action="store_true",
                      help="не сохранять эту сессию на диск")
    return parser.parse_args(argv)


def enable_line_editing() -> None:
    """Стрелки, история ввода и целая вставка из буфера обмена."""
    try:
        # Импорт ради побочного эффекта: readline включает стрелки, Home/End
        # и историю ввода в стандартном input().
        importlib.import_module("readline")
    except Exception:
        pass
    enable_bracketed_paste()


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

    Строка прячется только в одном случае: все запросы основные. Тогда она и
    правда лишняя. А вот единственный запрос вида «под-агент» скрывать нельзя:
    человек увидел бы «запросов к API — 1» и решил, что это его собственный
    вопрос, хотя своих вопросов он ещё не задавал.
    """
    spent_by_kind = [(kind, spent) for kind, spent in session.agent.usage.by_kind.items()
                     if spent.requests]
    if len(spent_by_kind) == 1 and spent_by_kind[0][0] == MAIN:
        return ""
    return " · ".join("{} — {} {}".format(
        kind, spent.requests,
        plural(spent.requests, ("запрос", "запроса", "запросов")))
        for kind, spent in spent_by_kind)


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
        return error_panel("\n".join(errors) + PARAMS_HINT, markup=True)

    session.params = apply(session.params, updates)
    lines = ["Применено: " + ", ".join(
        "[bold]{}[/]=[yellow]{}[/]".format(name, format_value(value))
        for name, value in updates.items())]
    if errors:
        lines.append("[red]Не принято: {}[/]{}".format("; ".join(errors), PARAMS_HINT))
    lines.append("[dim]Действует для всех следующих запросов.[/]")
    return info_panel("\n".join(lines), title="Параметры генерации", style="cyan")


def chat_loop(console: Console, keys, session: Session,
              sessions: Optional[SessionStore] = None,
              memory_store: Optional[MemoryStore] = None) -> None:
    """Цикл диалога. ``sessions`` задан — разговор пишется на диск после ответа."""
    notice: Optional[RenderableType] = None

    def remember() -> None:
        if sessions is None:
            return
        try:
            # Имя придумывается само, если человек его не задал: пустая
            # подпись в списке бесполезна, а тему модель уже сочинила.
            if not session.title:
                session.title = session.suggest_title()
            saved = sessions.save(session.agent, topic=session.topic,
                                  title=session.title, seen_at=session.seen_at)
            session.seen_at = saved.updated_at
            if saved.forked_from:
                console.print(warning_panel(
                    "Эту сессию успел изменить кто-то ещё — похоже, она открыта "
                    "во втором окне.\nЧтобы ничего не потерялось, разговор "
                    "продолжается отдельной сессией [bold]{}[/].\n"
                    "[dim]Прежняя ({}) осталась целой.[/]".format(
                        saved.session_id, saved.forked_from)))
        except (OSError, AgentError) as exc:
            # Сессия — удобство, а не суть: не записалась, так не записалась,
            # но терять из-за этого уже полученный ответ недопустимо.
            console.print(warning_panel("Сессию не удалось сохранить: {}".format(exc)))

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
            if command in ("/tokens", "/token"):
                notice = tokens_panel(session)
                continue
            if command in ("/summary", "/sum"):
                notice = summary_panel(session)
                continue
            if command == "/compress":
                notice = compress_now(console, session)
                remember()
                continue
            if command in ("/strategy", "/strategies"):
                notice = switch_strategy(session, argument)
                remember()
                continue
            if command in ("/facts", "/fact"):
                notice = handle_facts(console, session, argument)
                remember()
                continue
            if command in ("/memory", "/mem"):
                notice = handle_memory(session, argument, memory_store)
                remember()
                continue
            if command in ("/profile", "/persona"):
                notice = handle_profile(session, argument, memory_store)
                remember()
                continue
            if command in ("/task", "/state"):
                notice = handle_task(session, argument)
                remember()
                continue
            if command in ("/invariant", "/invariants"):
                notice = handle_invariant(session, argument)
                remember()
                continue
            if command == "/pause":
                notice = pause_task(session)
                remember()
                continue
            if command == "/resume":
                notice = resume_task(session)
                remember()
                continue
            if command in ("/checkpoint", "/point"):
                notice = mark_checkpoint(session, argument)
                remember()
                continue
            if command == "/branch":
                notice = make_branch(session, argument)
                remember()
                continue
            if command == "/branches":
                notice = branches_panel(session)
                continue
            if command == "/switch":
                notice = switch_branch(session, argument)
                remember()
                continue
            if command == "/config":
                notice = config_panel(session)
                continue
            if command in ("/rename", "/name"):
                notice = rename_session(session, argument)
                remember()
                continue
            if command == "/history":
                render_history(console, session)
                continue
            # Команды ниже меняют конфиг, а не переписку, и сохранять их надо
            # сразу: переключить модель и выйти, не задав вопроса, — обычное
            # дело, и возврат в сессию должен вернуть именно новую модель.
            if command == "/change_llm_params":
                notice = handle_params_command(session, argument)
                remember()
                continue
            if command == "/reset_llm_params":
                session.params = GenerationParams()
                notice = info_panel("Параметры генерации вернулись к значениям "
                                    "по умолчанию.", title="Сброшено", style="green")
                remember()
                continue
            if command in ("/change_model", "/model", "/models"):
                notice = switching.handle_command(console, keys, session, argument)
                remember()
                continue
            if command == "/retry":
                notice = retry_last(console, session)
                remember()
                continue
            if command in ("/agent", "/agents", "/sub"):
                notice = delegate(console, session, argument)
                remember()
                continue
            if command == "/new":
                # Новый идентификатор, а не очистка на месте: прежний разговор
                # остаётся на диске, и его не затирает первый же новый ответ.
                session.agent.new_session()
                session.reset()
                # Имя тоже забываем: иначе новый разговор сохранился бы под
                # именем прежнего, и оба стали бы неотличимы в списке.
                session.title = ""
                session.seen_at = time.time()
                notice = info_panel(
                    "История очищена, контекст свободен.\n"
                    "[dim]Это новая сессия {}; прежняя осталась в списке "
                    "--sessions.[/]".format(session.agent.session_id),
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
        try:
            forgotten = session.agent.fit_context()
        except ContextOverflow as exc:
            session.drop_last_user()
            notice = error_panel(
                "{}.\n[dim]Начните новый диалог командой /new, сократите вопрос "
                "или разрешите забывать начало разговора: "
                "--set history.on_overflow=trim.[/]".format(escape(str(exc))),
                markup=True)
            continue
        if forgotten:
            # Молча сократить отправляемое нельзя: следующий ответ модели будет
            # выглядеть беспамятным, и человек не поймёт, почему.
            console.print(warning_panel(overflow_notice(session, forgotten)))
        elif session.agent.sends_anyway and not session.agent.breakdown().fits:
            # Запрос заведомо не влезает, но отправляется намеренно: человек
            # попросил показать не нашу проверку, а ответ провайдера.
            over = session.agent.breakdown()
            console.print(warning_panel(
                "Запрос не помещается в окно на {} {}, но уходит как есть: "
                "on_overflow=send. Дальше слово за провайдером — покажем, что "
                "он ответит.".format(
                    fmt(over.excess),
                    plural(over.excess, ("токен", "токена", "токенов")))))

        render_frame(console, session)
        result, notice = request_answer(console, session)
        if result is None:
            session.drop_last_user()
            continue

        for call in result.tool_calls:
            console.print(subagent.tool_panel(call))

        if should_update_topic(session):
            with console.status("[dim]Определяю тему диалога…[/]", spinner="dots"):
                update_topic(session)
        remember()

        if session.free_tokens() < session.avg_exchange_tokens() * 2:
            free = session.free_tokens()
            notice = warning_panel(
                "Контекст почти заполнен: свободно {} {}. "
                "Скоро понадобится /new, иначе диалог придётся начать заново.".format(
                    fmt(free), plural(free, ("токен", "токена", "токенов")))
            )

    farewell(console, session)


def compress_now(console: Console, session: Session) -> RenderableType:
    """Сжать историю прямо сейчас, не дожидаясь порога.

    Печатается ожидание: пересказ — обычный запрос к модели, и он идёт
    столько же, сколько ответ на вопрос.
    """
    if not session.messages:
        return warning_panel("Сжимать нечего: разговор ещё не начался.")
    console.print("[dim]Пересказываю начало разговора…[/]")
    try:
        folded = session.agent.compress_history(force=True)
    except (AgentError, LLMError) as exc:
        return error_panel("Пересказ не получен: {}".format(exc))
    if not folded:
        return info_panel(
            "Сжимать пока нечего: все сообщения и так уходят в модель дословно.",
            title="Сжатие", style="cyan")
    return summary_panel(session, folded=folded)


def summary_panel(session: Session, folded: int = 0) -> RenderableType:
    """Пересказ начала разговора и то, что он даёт."""
    breakdown = session.breakdown()
    if not session.summary:
        return info_panel(waiting_for_summary(session), title="Пересказа пока нет",
                          style="cyan")
    head = Text()
    if folded:
        head.append("Свёрнуто сообщений: {}\n".format(folded), style="green")
    head.append("В модель вместо {} сообщений начала уходит пересказ на {} "
                "{} — на {} {} меньше в каждом запросе.\n".format(
                    breakdown.held_back, fmt(breakdown.summary),
                    plural(breakdown.summary, ("токен", "токена", "токенов")),
                    fmt(breakdown.saved),
                    plural(breakdown.saved, ("токен", "токена", "токенов"))),
                style="dim")
    return Panel(Group(head, Text(session.summary)), title="Пересказ начала",
                 title_align="left", border_style="magenta", box=box.ROUNDED,
                 padding=(0, 1))


def waiting_for_summary(session: Session) -> str:
    """Почему пересказа ещё нет и когда он появится."""
    policy = session.compression
    if session.strategy != STRATEGY_SUMMARY:
        return ("Сейчас работает стратегия [bold]{}[/]: пересказ при ней не "
                "пишется.\n[dim]Перейти на пересказ:[/] [cyan]/strategy summary[/]"
                .format(session.strategy))
    if session.compression_note:
        return ("Прошлое сжатие не удалось: {}\n"
                "[dim]В модель пока уходит вся переписка. Попробовать ещё раз:[/] "
                "[cyan]/compress[/]".format(escape(session.compression_note)))
    left = policy.every - max(0, len(session.messages) - session.keep_last)
    return ("Начало разговора свернётся в пересказ, когда сверх последних {} "
            "сообщений накопится ещё {}: осталось {}.\n"
            "[dim]Не дожидаться:[/] [cyan]/compress[/]".format(
                session.keep_last, policy.every, max(1, left)))


# --------------------------------------------------------------------------
# Стратегия контекста, факты и ветки
# --------------------------------------------------------------------------

def overflow_notice(session: Session, forgotten: int) -> str:
    """Что именно случилось с перепиской, когда она перестала помещаться.

    При скользящем окне и фактах сообщения остаются на месте — сужается окно;
    при полной истории и пересказе начало выбрасывается насовсем. Написать
    одно вместо другого значило бы соврать в обе стороны сразу.
    """
    count = "{} {}".format(forgotten, plural(
        forgotten, ("сообщение", "сообщения", "сообщений")))
    if session.history.slides:
        return ("Контекст переполнился, и окно сузилось на {}: в модель уходит "
                "более короткий хвост переписки. Сама переписка цела — "
                "смотрите /history.".format(count))
    return ("Контекст переполнился, и начало разговора забыто: {} больше "
            "не уходят в модель. Отвечать она будет только по остатку."
            .format(count))


def switch_strategy(session: Session, argument: str) -> RenderableType:
    """Переключить стратегию управления контекстом или показать список."""
    wanted = " ".join(argument.split()).lower()
    if not wanted:
        return strategy_panel(session)
    if wanted not in STRATEGIES:
        return error_panel("Нет стратегии «{}». Есть: {}.".format(
            escape(wanted), ", ".join(STRATEGIES)))
    if wanted == session.strategy:
        return strategy_panel(session, changed="Она и так уже работала.")
    was = session.strategy
    session.agent.reconfigure(history=session.history.with_strategy(wanted))
    return strategy_panel(session, changed=switch_note(session, was))


def switch_note(session: Session, was: str) -> str:
    """Что изменится в следующем запросе после смены стратегии."""
    kept = []
    if session.summary:
        kept.append("пересказ начала сохранён")
    if session.facts:
        kept.append("картотека из {} {} сохранена".format(
            len(session.facts), plural(len(session.facts),
                                       ("факта", "фактов", "фактов"))))
    saved = "; ".join(kept) + ". " if kept else ""
    if session.strategy == STRATEGY_FACTS and not session.facts:
        return (saved + "Картотека пока пуста: она наполнится после следующего "
                "вопроса, а до того запрос собирается как при window.")
    if session.strategy == STRATEGY_SUMMARY and not session.summary:
        return saved + "Пересказ появится, когда накопится порог сжатия."
    if session.strategy == STRATEGY_FULL:
        return saved + "Со следующего вопроса в модель снова уходит вся переписка."
    return saved + "Была {}.".format(was)


def handle_facts(console: Console, session: Session, argument: str) -> RenderableType:
    """Картотека фактов: показать, поправить руками, обновить или очистить."""
    wanted = argument.strip()
    if not wanted:
        return facts_panel(session)
    if wanted.lower() == "reset":
        session.agent.conversation.forget_facts()
        return facts_panel(session)
    if wanted.lower() == "update":
        console.print("[dim]Разбираю разговор на факты…[/]")
        try:
            changed = session.agent.update_facts()
        except (AgentError, LLMError, ValueError) as exc:
            return error_panel("Картотека не обновлена: {}".format(escape(str(exc))))
        if not changed:
            return facts_panel(session)
        return facts_panel(session, changed=changed)
    if "=" not in wanted:
        return error_panel(
            "Нужно [cyan]/facts ключ=значение[/], [cyan]/facts ключ=[/] чтобы "
            "забыть, либо [cyan]/facts update[/] и [cyan]/facts reset[/].",
            markup=True)
    key, _, value = wanted.partition("=")
    changed = session.agent.remember_fact(key, value)
    if not changed:
        return facts_panel(session)
    return facts_panel(session, changed=changed)


def handle_memory(session: Session, argument: str,
                  store: Optional[MemoryStore] = None) -> RenderableType:
    """Явно направить запись в рабочую или долговременную память."""
    words = argument.strip().split(None, 2)
    if not words:
        return memory_panel(session)

    aliases = {"short": "short", "краткая": "short",
               "work": "working", "working": "working", "рабочая": "working",
               "long": "long", "long-term": "long", "долгая": "long"}
    layer = aliases.get(words[0].lower(), words[0].lower())
    try:
        if layer == "short":
            if len(words) > 1:
                return error_panel(
                    "Краткосрочная память заполняется обычными репликами и "
                    "очищается /new; вручную в неё не записывают.")
            return memory_panel(session)

        if layer == "working":
            assignment = argument.strip().partition(" ")[2].strip()
            if not assignment:
                return memory_panel(session)
            key, value = _memory_assignment(assignment)
            changed = session.agent.remember("working", key, value)
            action = "записано" if value else "забыто"
            note = "Рабочая память: {} «{}»{}".format(
                action, key, "" if changed else " (без изменений)")
            return memory_panel(session, note)

        if layer == "long":
            if len(words) < 2:
                return memory_panel(session)
            section = words[1].lower()
            if len(words) < 3:
                return error_panel(
                    "Нужна запись: /memory long "
                    "profile|rules|decisions|knowledge ключ=значение")
            key, value = _memory_assignment(words[2])
            if store is not None:
                # Store перечитывает данные под блокировкой и возвращает свежий
                # снимок. Live-память меняется только после успешной записи.
                memory, changed = store.remember(section, key, value)
                session.agent.replace_long_term_memory(memory)
            else:
                changed = session.agent.remember("long", key, value, section=section)
            action = "записано" if value else "забыто"
            note = "Долговременная память · {}: {} «{}»{}".format(
                section, action, key, "" if changed else " (без изменений)")
            return memory_panel(session, note)

        return error_panel(
            "Неизвестный слой «{}». Используйте short, working или long.".format(
                words[0]))
    except (AgentError, OSError) as exc:
        return error_panel(str(exc))


def handle_profile(session: Session, argument: str,
                   store: Optional[MemoryStore] = None) -> RenderableType:
    """Показать или изменить профиль без длинного префикса /memory long."""
    assignment = argument.strip()
    if not assignment:
        return memory_panel(session)
    return handle_memory(session, "long profile " + assignment, store)


INVARIANT_CATEGORY_ALIASES = {
    "architecture": "architecture",
    "архитектура": "architecture",
    "technical-decision": "technical-decision",
    "decision": "technical-decision",
    "решение": "technical-decision",
    "stack": "stack",
    "стек": "stack",
    "business-rule": "business-rule",
    "business": "business-rule",
    "бизнес": "business-rule",
}


def handle_invariant(session: Session, argument: str) -> RenderableType:
    """Показать и править строгие ограничения прямо из диалога."""
    words = argument.strip().split(None, 1)
    if not words:
        return invariants_panel(session)
    action = words[0].lower()
    rest = words[1].strip() if len(words) > 1 else ""
    try:
        if action in ("add", "set"):
            parts = rest.split(None, 2)
            if len(parts) < 3:
                return error_panel(
                    "Нужно: /invariant add категория id правило [| причина].")
            category = INVARIANT_CATEGORY_ALIASES.get(
                parts[0].lower(), parts[0].lower())
            rule, separator, reason = parts[2].partition("|")
            changed = session.agent.set_invariant(
                parts[1], category, rule.strip(), reason.strip() if separator else "")
            note = "Инвариант {} «{}». Действует со следующей реплики.".format(
                "сохранён" if changed else "не изменился", parts[1])
            return invariants_panel(session, note)
        if action in ("remove", "delete", "rm"):
            if not rest:
                return error_panel("Нужно назвать id: /invariant remove stack.python.")
            changed = session.agent.remove_invariant(rest)
            note = ("Инвариант «{}» удалён." if changed
                    else "Инвариант «{}» не найден.").format(rest)
            return invariants_panel(session, note)
        if action in ("clear", "reset"):
            removed = session.agent.clear_invariants()
            return invariants_panel(
                session, "Удалено инвариантов: {}.".format(removed))
        return error_panel(
            "Неизвестное действие «{}». Используйте add, remove или clear.".format(
                action))
    except (AgentError, ConfigError) as exc:
        return error_panel(str(exc))


def handle_task(session: Session, argument: str) -> RenderableType:
    """Показать или изменить состояние конечного автомата задачи."""
    words = argument.strip().split(None, 1)
    if not words:
        return task_panel(session)
    action = words[0].lower()
    details = words[1] if len(words) > 1 else ""
    step, expected = _task_details(details)

    try:
        if action == "start":
            session.agent.start_task(step, expected)
            return task_panel(session, "Новая задача начата с этапа planning.")
        if action in ("step", "update", "set"):
            session.agent.update_task(step, expected)
            return task_panel(session, "Текущий шаг обновлён.")
        if action == "next":
            previous = session.task_state.stage
            session.agent.advance_task(step, expected)
            return task_panel(session, "Переход: {} → {}.".format(
                previous, session.task_state.stage))
        if action == "reset":
            session.agent.reset_task()
            return task_panel(session, "Состояние задачи очищено.")
        if action in TASK_STAGES:
            if not session.task_state.active:
                if action != TASK_STAGES[0]:
                    return error_panel(
                        "Новая задача начинается с planning: /task start шаг | действие.")
                session.agent.start_task(step, expected)
            else:
                session.agent.transition_task(action, step, expected)
            return task_panel(session, "Этап задачи: {}.".format(action))
        return error_panel(
            "Неизвестное действие «{}». Используйте start, step, next, "
            "planning, execution, validation, done или reset.".format(action))
    except AgentError as exc:
        return error_panel(escape(str(exc)))


def pause_task(session: Session) -> RenderableType:
    """Поставить автомат на паузу, не меняя его координаты."""
    try:
        session.agent.pause_task()
    except AgentError as exc:
        return error_panel(escape(str(exc)))
    return task_panel(session, "Задача приостановлена; этап и шаг сохранены.")


def resume_task(session: Session) -> RenderableType:
    """Снять паузу и показать точку, с которой продолжится работа."""
    try:
        session.agent.resume_task()
    except AgentError as exc:
        return error_panel(escape(str(exc)))
    return task_panel(session, "Задача продолжена с сохранённого места.")


def _task_details(text: str):
    """Разобрать ``текущий шаг | ожидаемое действие``."""
    step, separator, expected = text.partition("|")
    return step.strip(), expected.strip() if separator else ""


def _memory_assignment(text: str):
    key, separator, value = text.partition("=")
    key = key.strip()
    if not separator or not key:
        raise AgentError(
            "Запись памяти задаётся как ключ=значение; пустое значение удаляет ключ.")
    return key, value.strip()


def mark_checkpoint(session: Session, argument: str) -> RenderableType:
    """Запомнить место разговора, от которого потом отводят ветки."""
    if not session.messages:
        return warning_panel("Запоминать нечего: разговор ещё не начался.")
    try:
        point = session.agent.mark_checkpoint(argument)
    except AgentError as exc:
        return error_panel(escape(str(exc)))
    return branches_panel(session, note=(
        "Точка «{}» поставлена: {} {} разговора. Отвести от неё ветку — "
        "/branch имя {}.".format(
            point.label, point.messages,
            plural(point.messages, ("сообщение", "сообщения", "сообщений")),
            point.label)))


def make_branch(session: Session, argument: str) -> RenderableType:
    """Отвести ветку от точки возврата и продолжать разговор в ней."""
    name, _, label = argument.strip().partition(" ")
    if not name:
        return error_panel(
            "У ветки должно быть имя: [cyan]/branch вариант-а[/] или "
            "[cyan]/branch вариант-б выбор-стека[/] — от названной точки.",
            markup=True)
    try:
        branch = session.agent.fork(name, label.strip())
    except AgentError as exc:
        return error_panel(escape(str(exc)))
    return branches_panel(session, note=(
        "Ветка «{}» отведена от точки «{}»: унаследовано {} {}. Всё сказанное "
        "дальше остаётся в ней и в соседние ветки не попадёт.".format(
            branch.name, branch.checkpoint, branch.inherited,
            plural(branch.inherited, ("сообщение", "сообщения", "сообщений")))))


def switch_branch(session: Session, argument: str) -> RenderableType:
    """Перейти в другую ветку разговора."""
    if not argument.strip():
        return branches_panel(session)
    try:
        branch = session.agent.switch_branch(argument)
    except AgentError as exc:
        return error_panel(escape(str(exc)))
    return branches_panel(session, note=(
        "Разговор продолжается в ветке «{}»: {} {}, из них своих {}.".format(
            branch.name, len(branch.conversation.messages),
            plural(len(branch.conversation.messages),
                   ("сообщение", "сообщения", "сообщений")), branch.own)))


def rename_session(session: Session, argument: str) -> RenderableType:
    """Переименовать сессию. Имя — то, по чему её потом узнают в списке."""
    wanted = " ".join(argument.split())
    if not wanted:
        return info_panel(
            "Сейчас сессия зовётся «[bold]{}[/]».\n"
            "[dim]Переименовать:[/] [cyan]/rename Разбор алгоритмов[/]".format(
                session.label), title="Имя сессии", style="cyan")
    if len(wanted) > 60:
        return error_panel("Имя длиннее 60 знаков — в списке оно не поместится.")
    session.title = wanted
    return info_panel(
        "Сессия теперь зовётся «[bold]{}[/]».\n"
        "[dim]Вернуться в неё:[/] [cyan]--resume \"{}\"[/]".format(wanted, wanted),
        title="Переименовано", style="green")


def delegate(console: Console, session: Session, argument: str) -> RenderableType:
    """Запустить под-агента отдельным процессом и принять его итог.

    Своей сессии под-агент не показывает: сюда возвращается только ответ, и он
    ложится в память этой сессии помеченной заметкой. Расход под-агента идёт
    в общий счёт отдельной строкой — платил тот же кошелёк.
    """
    name, tweaks, question = subagent.split_request(argument)
    if not name:
        return subagent.catalog_panel()
    try:
        # Имя проверяем до запуска процесса: «не справился» — неверная подпись
        # для того, что даже не начиналось.
        subagent.resolve_config(name)
    except ConfigError as exc:
        return error_panel(str(exc))
    if not question:
        return error_panel("Нечего спрашивать. Нужно: /agent {} <вопрос>, "
                           "либо /agent {} поле=значение -- <вопрос>".format(name, name))

    label = name + (" (" + " ".join(tweaks) + ")" if tweaks else "")
    with console.status("[bold]под-агент {} работает в своём процессе…[/]".format(label),
                        spinner="dots"):
        delegation = subagent.run(name, question, session.agent.config, tweaks)

    if delegation.ok:
        session.agent.record_delegation(delegation.name, delegation.question,
                                        delegation.text,
                                        subagent.caveat_of(delegation))
        session.agent.absorb_delegation(delegation.prompt_tokens,
                                        delegation.completion_tokens,
                                        delegation.seconds, delegation.cost,
                                        delegation.currency)
    return subagent.panel(delegation)


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
    if result.refused:
        identifiers = ", ".join(
            item.get("id", "") for item in result.invariant_violations)
        return info_panel(
            "Конфликт с инвариантами: {}. Решение не было выдано.".format(
                identifiers or "неизвестно"),
            title="Инварианты", style="yellow")
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
            "или сбросьте параметры.\n"
            "[dim]Контекст тут ни при чём: место в окне ещё есть ({} токенов), "
            "кончился именно отведённый ответу предел.[/]".format(
                fmt(session.output_reserve), fmt(session.free_tokens())))
    return None


# Именованные флаги и поля, которые они задают. Всё остальное — через --set:
# длинный список флагов вреден, справку перестают читать.
FLAG_FIELDS = (
    ("provider", "provider"),
    ("temperature", "generation.temperature"),
    ("max_tokens", "generation.max_tokens"),
    ("system_prompt", "system_prompt"),
    ("format", "output.format"),
    ("max_cost", "budget.max_cost"),
    ("agent_name", "name"),
    ("keep_last", "history.keep_last"),
    ("compress_every", "history.compression.every"),
    ("strategy", "history.strategy"),
)


def read_config(path: Optional[str]) -> AgentConfig:
    """Конфиг из строки JSON, из короткого имени в ``configs/`` или из файла.

    Короткое имя важно потому, что команду кладут в PATH и зовут из любого
    каталога, где относительного ``configs/demo.yaml`` попросту нет. Строка
    JSON нужна для разового запуска, под который заводить файл не хочется.
    """
    if not path:
        return DEFAULT_CONFIG
    if path.lstrip().startswith("{"):
        try:
            payload = json.loads(path)
        except ValueError as exc:
            raise ConfigError("конфиг в командной строке не разобран: {}".format(exc))
        return AgentConfig.from_dict(payload)
    return AgentConfig.from_file(str(subagent.resolve_config(path)))


def with_demo(config: AgentConfig, demo: bool) -> AgentConfig:
    if demo and not config.transport.demo:
        return config.with_changes(transport=Transport(demo=True))
    return config


def named_pairs(args: argparse.Namespace) -> List[tuple]:
    """Именованные флаги как список правок «путь — значение»."""
    pairs: List[tuple] = []
    if args.model:
        # Провайдер выводится из имени модели, но явный --provider его перебьёт:
        # он идёт ниже по списку, а последняя правка на поле побеждает.
        provider, model = find_model(args.model)
        pairs += [("provider", provider), ("model", model)]
    for attribute, path in FLAG_FIELDS:
        value = getattr(args, attribute, None)
        if value is not None:
            # Значение отдаём как есть: argparse уже привёл его к типу, и
            # прогон через запись строкой и обратно только добавлял бы
            # способов ошибиться — на 0.0000001 он и ошибался.
            pairs.append((path, value))
    if args.no_history:
        pairs.append(("history.enabled", False))
    # Прежние имена того же выбора. Они старше стратегий и остались затем, что
    # ими набраны все прежние запуски и показы дня 9.
    if args.no_compress:
        pairs.append(("history.strategy", STRATEGY_FULL))
    if args.compress:
        pairs.append(("history.strategy", STRATEGY_SUMMARY))
    # Явная стратегия идёт последней: она называет то же поле прямо, а
    # последняя правка на поле побеждает.
    if args.strategy:
        pairs.append(("history.strategy", args.strategy))
    return pairs


def build_config(args: argparse.Namespace, base: AgentConfig) -> AgentConfig:
    """Собрать конфиг из слоёв: основа, именованные флаги, --set, дописка.

    Порядок от общего к частному. ``--set`` перебивает именованный флаг, потому
    что назвать поле полным путём — более явное указание. Дописка к системному
    промпту идёт последней: она должна лечь на то, что получилось.
    """
    config = with_demo(base, args.demo)
    config = overrides.apply_pairs(config, named_pairs(args))
    config = overrides.apply(config, args.set or [])
    if args.append_system_prompt:
        config = overrides.append_system_prompt(config, args.append_system_prompt)
    return config


NAME_COLUMN_WIDTH = 34


def one_line(value: object, budget: int) -> str:
    """Значение одной строкой, не длиннее отведённого места.

    Обрезаем посередине, а не с конца: правки вроде ``--append-system-prompt``
    ложатся именно в хвост, и потеряв его, человек не смог бы проверить, что
    его правка применилась. Перенос строки заменяем видимым знаком, иначе
    длинный промпт разорвал бы строку поля пополам.
    """
    if value is None:
        return "—"
    shown = " ↵ ".join(part.strip() for part in str(value).split("\n") if part.strip())
    if len(shown) <= budget:
        return shown
    tail = max(12, budget // 3)
    head = budget - tail - 1
    return shown[:head] + "…" + shown[len(shown) - tail:]


def resolved_config_panel(config: AgentConfig, width: int = 100) -> RenderableType:
    """Собранный конфиг плоским перечнем: видно, что дали все слои вместе."""
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False, expand=True)
    table.add_column("Поле", style="bold", no_wrap=True, width=NAME_COLUMN_WIDTH)
    table.add_column("Значение", no_wrap=True, overflow="ignore")

    budget = max(30, width - NAME_COLUMN_WIDTH - 8)
    default = overrides.describe(AgentConfig())
    plain = dict(overrides.flatten(default))
    for path, value in overrides.flatten(overrides.describe(config)):
        shown = one_line(value, budget)
        changed = plain.get(path) != value
        table.add_row(path, "[bold yellow]{}[/]".format(shown) if changed
                      else "[dim]{}[/]".format(shown))

    hint = Text.from_markup(
        "\n[dim]Жёлтым — то, что отличается от значений по умолчанию.\n"
        "Ключ намеренно заменён на ***: конфиг задуман переносимым.[/]")
    return Panel(Group(table, hint), title="🧩 Собранный конфиг", title_align="left",
                 border_style="green", box=box.ROUNDED, padding=(0, 1))


def sessions_panel(records: List[SessionRecord], root: str,
                   needle: str = "") -> RenderableType:
    if not records:
        if needle:
            return info_panel(
                "По слову «{}» ничего не нашлось. Показать все — --sessions "
                "без слова.".format(needle), title="Сессии", style="yellow")
        return info_panel(
            "Сохранённых сессий нет. Они появляются после первого ответа "
            "в диалоге; одиночные запросы через --ask не сохраняются.",
            title="Сессии", style="yellow")

    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False, expand=True)
    table.add_column("№", justify="right", style="dim", no_wrap=True)
    table.add_column("Имя", style="bold", no_wrap=True)
    table.add_column("Когда", no_wrap=True, style="dim")
    table.add_column("Обменов", justify="right", style="dim")
    table.add_column("О чём говорили")

    for record in records:
        name = shorten_to(record.label, 28)
        about = shorten_to(record.first_question, 52)
        # Имя, придуманное из первого вопроса, повторяло бы его в соседней
        # колонке. Повтор занимает самое широкое место в списке и ничего
        # не добавляет, поэтому второй раз показываем прочерк.
        if _same_start(name, about):
            about = "—"
        table.add_row(str(record.number), name, when(record.updated_at),
                      str(record.exchanges), about)

    hint = Text.from_markup(
        "\n[cyan]--resume {}[/][dim] — по номеру · [/][cyan]--resume {}[/]"
        "[dim] — по имени · [/][cyan]--continue[/][dim] — в самую свежую[/]\n"
        "[cyan]--resume[/][dim] без имени — предложит выбрать · [/]"
        "[cyan]--sessions слово[/][dim] — поиск по переписке · [/]"
        "[cyan]--rm-session {}[/][dim] — удалить[/]\n"
        "[dim]Переписка лежит файлами в {} и читается только вами.[/]".format(
            1, quoted(records[0].label), 1, root))
    title = "🗂 Сохранённые сессии" if not needle else \
        "🗂 Сессии по слову «{}»".format(needle)
    return Panel(Group(table, hint), title=title, title_align="left",
                 border_style="cyan", box=box.ROUNDED, padding=(0, 1))


def _same_start(left: str, right: str) -> bool:
    """Одно ли и то же говорят две подписи, с точностью до обрезки."""
    shorter, longer = sorted((left.rstrip("…"), right.rstrip("…")), key=len)
    return bool(shorter) and longer.startswith(shorter)


def shorten_to(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "—"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def quoted(name: str) -> str:
    """Имя для примера в подсказке: с пробелами — в кавычках."""
    short = shorten_to(name, 20)
    return '"{}"'.format(short) if " " in short else short


def when(moment: float) -> str:
    """Время последней правки словами: точная дата в списке ни к чему."""
    if not moment:
        return "—"
    gap = max(0.0, time.time() - moment)
    if gap < 90:
        return "только что"
    if gap < 3600:
        return "{:.0f} мин назад".format(gap / 60)
    if gap < 86400:
        return "{:.0f} ч назад".format(gap / 3600)
    if gap < 86400 * 7:
        return "{:.0f} дн назад".format(gap / 86400)
    return time.strftime("%d.%m.%Y", time.localtime(moment))


def agent_from_config(console: Console, config: AgentConfig, store,
                      session_id: Optional[str] = None,
                      long_term_memory=None) -> Agent:
    """Собрать агента по готовому конфигу, спросив только недостающее.

    Заданный файлом конфиг проходит без единого вопроса — ровно так же, как
    его прошли бы сто конфигов в массовом прогоне. Человека тревожат только
    тогда, когда ключа нет ни в конфиге, ни в окружении, ни в файле.
    """
    agent = Agent(config, session_id=session_id,
                  toolbox=subagent.toolbox_for(config),
                  long_term_memory=long_term_memory)
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
    ready = config.with_changes(api_key=credentials.key, api_extra=credentials.extra)
    return Agent(ready, session_id=session_id, toolbox=subagent.toolbox_for(ready),
                 long_term_memory=long_term_memory)


def one_shot(args: argparse.Namespace, sessions: SessionStore,
             memory_store: Optional[MemoryStore] = None) -> int:
    """Один вопрос без диалога: этим режимом родитель вызывает под-агента.

    Ни одного вопроса в консоль здесь не задаётся — процесс может быть
    дочерним, и спрашивать некого. Не нашлось реквизитов — сразу ошибка.
    В режиме ``--json`` в stdout уходит только объект JSON: его разбирает
    вызывающий, и любая посторонняя строка сломала бы разбор.

    Сессию этот режим читает, но не записывает: иначе каждый вызов под-агента
    оставлял бы после себя файл.
    """
    def fail(message: str, code: int, session_id: str = "") -> int:
        if args.json:
            payload = {"ok": False, "error": message}
            if session_id:
                payload["session_id"] = session_id
            json.dump(payload, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        else:
            print(message, file=sys.stderr)
        return code

    if args.resume == "":
        # Выбирать не у кого: этот режим бывает дочерним процессом, и вопрос
        # в stdout сломал бы разбор ответа у вызывающего.
        return fail("с --ask сессию надо назвать: --resume <номер|имя|id>", 2)

    try:
        record, base = choose_base(args, sessions)
        config = build_config(args, base)
        toolbox = subagent.toolbox_for(config)
        long_term = memory_store.load() if memory_store is not None else None
        if record is not None:
            agent = restore_agent(record, config, toolbox=toolbox,
                                  long_term_memory=long_term)
        else:
            agent = Agent(config, session_id=args.session_id, toolbox=toolbox,
                          long_term_memory=long_term)
        # Реквизиты ищем сразу, а не при первом запросе: у отсутствия ключа
        # должен быть свой код возврата, иначе вызывающий не отличит его от
        # отказа провайдера и станет повторять безнадёжный вызов.
        agent.credentials
    except MissingCredentials as exc:
        return fail(str(exc), 3)
    except ConfigError as exc:
        return fail("конфиг не принят: {}".format(exc), 2)
    except AgentError as exc:
        # Любая другая беда агента — тоже ответ, а не трассировка в stdout:
        # её читает вызывающий, и он ждёт объект JSON.
        return fail(str(exc), 2)

    try:
        result = agent.ask(args.ask)
    except (InputRejected, AgentError, LLMError) as exc:
        return fail(str(exc), 1, agent.session_id)

    if args.json:
        payload = result.to_dict()
        payload["session_id"] = agent.session_id
        payload["usage"] = agent.usage.snapshot()
        json.dump(payload, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(result.text)
    return 0


def list_sessions(console: Console, sessions: SessionStore, needle: str = "") -> int:
    records = sessions.matching(needle) if needle else sessions.recent()
    console.print(sessions_panel(records, str(sessions.root), needle))
    return 0


def remove_sessions(console: Console, sessions: SessionStore, target: str) -> int:
    """Удалить сессию или все сразу. Переписка лежит на диске — её надо чем-то убирать."""
    if target.strip().lower() == "all":
        removed = sessions.clear()
        console.print(info_panel("Удалено {} {}.".format(
            removed, plural(removed, ("сессия", "сессии", "сессий"))),
            title="Сессии удалены", style="green"))
        return 0
    try:
        found = sessions.remove(target)
    except AgentError as exc:
        # Неоднозначное имя: сказать «нет такой» было бы неправдой, а удалить
        # наугад — потерей чужого разговора.
        console.print(error_panel(str(exc)))
        return 1
    if found:
        console.print(info_panel("Сессия {} удалена.".format(target),
                                 title="Готово", style="green"))
        return 0
    console.print(error_panel("Нет сессии «{}». Список — --sessions.".format(target)))
    return 1


def pick_record(args: argparse.Namespace,
                sessions: SessionStore) -> Optional[SessionRecord]:
    """Найти сессию, в которую велено вернуться."""
    if args.resume:
        return sessions.find(args.resume)
    if args.resume == "":
        # «--resume» без значения: человек хочет вернуться, но не помнит куда.
        return choose_record(sessions)
    if args.continue_session:
        record = sessions.latest()
        if record is None:
            raise ConfigError("возвращаться некуда: сохранённых сессий нет")
        return record
    return None


def choose_record(sessions: SessionStore) -> SessionRecord:
    """Показать список и спросить, какую сессию продолжить."""
    records = sessions.recent(limit=20)
    if not records:
        raise ConfigError("возвращаться некуда: сохранённых сессий нет")
    if len(records) == 1:
        return records[0]

    console = make_console()
    console.print(sessions_panel(records, str(sessions.root)))
    try:
        answer = read_user_line(console, "[bold cyan]Какую продолжить? ›[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        raise ConfigError("выбор прерван")
    if not answer:
        raise ConfigError("сессия не выбрана")
    return sessions.find(answer)


def choose_base(args: argparse.Namespace, sessions: SessionStore):
    """Сессия, в которую возвращаемся, и основа конфига для неё.

    Если велено вернуться в сессию, её конфиг становится основой — но заданный
    руками ``--config`` всё равно главнее: раз человек назвал его вместе с
    возвратом, значит хочет продолжить разговор на других настройках.
    """
    record = pick_record(args, sessions)
    if args.config or record is None:
        return record, read_config(args.config)
    return record, record.to_config()


def prepare_agent(console: Console, args: argparse.Namespace, sessions: SessionStore,
                  keys, record: Optional[SessionRecord], config: AgentConfig,
                  memory_store: Optional[MemoryStore] = None) -> Agent:
    """Собрать агента: из сохранённой сессии, из конфига или через расспрос."""
    agent = build_agent(console, args, sessions, keys, record, config, memory_store)
    return seed_credentials(agent, keys)


def seed_credentials(agent: Agent, keys) -> Agent:
    """Рассказать хранилищу ключей о том, что агент уже знает.

    Иначе первая же смена модели внутри того же провайдера спросит ключ,
    который у программы и так на руках.
    """
    try:
        keys.remember(agent.config.provider_info, agent.credentials)
    except AgentError:
        pass       # реквизитов нет — спросим, когда действительно понадобятся
    return agent


def build_agent(console: Console, args: argparse.Namespace, sessions: SessionStore,
                keys, record: Optional[SessionRecord], config: AgentConfig,
                memory_store: Optional[MemoryStore] = None) -> Agent:
    long_term = memory_store.load() if memory_store is not None else None
    if record is not None:
        agent = restore_agent(record, config, toolbox=subagent.toolbox_for(config),
                              long_term_memory=long_term)
        console.print(returned_panel(record, agent))
        return agent

    session_id = check_session_id(args.session_id, sessions)
    if args.config or args.provider or args.model:
        # Провайдер или модель, названные флагом, — это уже сделанный выбор.
        # Расспрос поверх него не только лишний: он ещё и затирал названное
        # своими ответами, и документированная команда работала не так.
        show_banner(console)
        return agent_from_config(console, config, keys, session_id, long_term)
    chosen = setup(console, args.demo, ask_keys=args.ask_keys, store=keys, base=config)
    return Agent(chosen, session_id=session_id,
                 toolbox=subagent.toolbox_for(chosen),
                 long_term_memory=long_term)


def check_session_id(session_id: Optional[str], sessions: SessionStore) -> Optional[str]:
    """Проверить заданный идентификатор до начала разговора.

    Негодный отвергаем сразу: иначе он всплыл бы при первой попытке записи —
    то есть уже после ответа, который было бы обидно потерять. Занятый тоже
    отвергаем: почти наверняка это попытка продолжить сессию, и промолчать
    значило бы затереть сохранённый разговор.
    """
    if not session_id:
        return None
    try:
        sessions.path_for(session_id)
    except AgentError as exc:
        raise ConfigError("негодный --session-id: {}".format(exc))
    try:
        sessions.load(session_id)
    except SessionNotFound:
        return session_id
    raise ConfigError("сессия «{}» уже сохранена. Продолжить её — --resume {}, "
                      "или задайте другой идентификатор".format(session_id, session_id))


def returned_panel(record: SessionRecord, agent: Agent) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Сессия", "{} · {}".format(record.label, record.session_id))
    table.add_row("Последний раз", when(record.updated_at))
    if record.first_question:
        table.add_row("Начиналась с", shorten_to(record.first_question, 64))
    table.add_row("Переписка", "{} {} · {} {}".format(
        len(record.messages), plural(len(record.messages),
                                     ("сообщение", "сообщения", "сообщений")),
        record.exchanges, plural(record.exchanges, ("обмен", "обмена", "обменов"))))
    table.add_row("Уже потрачено", "{} токенов".format(fmt(record.total_tokens)))
    table.add_row("Модель", "{} · {}".format(agent.config.provider_info.name,
                                             agent.config.model))
    if record.model and record.model != agent.config.model:
        table.add_row("Была модель", record.model)
    return Panel(table, title="↩ Возвращаюсь в сессию", title_align="left",
                 border_style="green", box=box.ROUNDED, padding=(0, 1))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sessions = SessionStore()
    memory_store = MemoryStore()
    console = make_console()

    if args.sessions is not None:
        return list_sessions(console, sessions, args.sessions)
    if args.rm_session:
        return remove_sessions(console, sessions, args.rm_session)

    # Одиночный запрос разбирает конфиг сам и сам отвечает на беды: до этой
    # ветки нельзя напечатать ни панели, ни вопроса, иначе вызывающий получит
    # рамки вместо JSON.
    if args.ask is not None:
        return one_shot(args, sessions, memory_store)

    try:
        record, base = choose_base(args, sessions)
        config = build_config(args, base)
    except (ConfigError, AgentError) as exc:
        console.print(error_panel(str(exc)))
        return 2

    if args.show_config:
        console.print(resolved_config_panel(config, console.width))
        return 0

    enable_line_editing()
    keys = switching.CredentialStore(demo=config.transport.demo, ask_keys=args.ask_keys)
    try:
        agent = prepare_agent(console, args, sessions, keys, record, config,
                              memory_store)
    except (ConfigError, AgentError) as exc:
        console.print(error_panel(str(exc)))
        return 2
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Отменено.[/]")
        return 130

    topic = record.topic if record is not None and record.topic else DEFAULT_TOPIC
    title = args.name or (record.title if record is not None else "")
    live = Session(agent, topic=topic, title=title)
    # Время прочитанной записи запоминаем: по нему потом видно, не изменил ли
    # эту сессию кто-то ещё, пока мы разговаривали.
    live.seen_at = record.updated_at if record is not None else time.time()
    chat_loop(console, keys, live, sessions=None if args.no_save else sessions,
              memory_store=memory_store)
    return 0
