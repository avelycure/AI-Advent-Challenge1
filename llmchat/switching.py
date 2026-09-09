"""Переключение модели и провайдера прямо во время диалога.

Смена модели не трогает историю: один и тот же вопрос можно переспросить на
другой модели и сравнить ответы, не выходя из сессии. Ключи провайдеров
запоминаются на время работы программы, поэтому возврат к уже опробованному
провайдеру ничего не спрашивает.

Спрашивать реквизиты — работа интерфейса, а не агента: агент их только
принимает готовыми в конфиге. Поэтому здесь остался ровно диалог с человеком,
а собранное уходит в ``Agent.reconfigure``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from llmagent import Agent, AgentConfig, Credentials, LLMError
from llmagent.transport import PROVIDER_ORDER, PROVIDERS, ModelInfo, ProviderInfo

from .session import Session
from .ui import ask_extra_field, ask_token, error_panel, fmt, info_panel, plural


def handle_command(console: Console, store: "CredentialStore", session: Session,
                   argument: str) -> RenderableType:
    """Обработать /change_model и вернуть панель с результатом."""
    if not argument.strip():
        return catalog_panel(session, store)
    choice, complaint = resolve(argument)
    if choice is None:
        return error_panel(complaint)
    return switch(console, store, session, choice)


def catalog_panel(session: Session, store: "CredentialStore") -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False, expand=True)
    table.add_column("Провайдер", style="bold", no_wrap=True)
    table.add_column("Ключ", no_wrap=True)
    table.add_column("Модели")

    for provider, choices in grouped():
        table.add_row(
            Text(provider.name, style=provider.accent),
            Text("готов", style="green") if store.has(provider) else Text("спросит", style="dim"),
            model_chips(choices, session),
        )

    hint = Text.from_markup(
        "\n[dim]Текущая модель — жёлтым. «спросит» значит, что при первом переходе "
        "к провайдеру понадобится ключ; дальше он запомнен до конца работы.[/]\n"
        "[cyan]/change_model 7[/][dim] — по номеру · [/]"
        "[cyan]/change_model gpt-5.5[/][dim] — по имени · [/]"
        "[cyan]/retry[/][dim] — переспросить последний вопрос на текущей модели[/]")
    return Panel(Group(table, hint), title="🔀 Модели", title_align="left",
                 border_style="cyan", box=box.ROUNDED, padding=(0, 1))


def resolve(argument: str) -> Tuple[Optional["Choice"], str]:
    """Разобрать аргумент: номер, имя модели или «провайдер модель»."""
    parts = argument.split()
    choices = catalog()

    if len(parts) == 1 and parts[0].isdigit():
        number = int(parts[0])
        if 1 <= number <= len(choices):
            return choices[number - 1], ""
        return None, "Нет модели под номером {}: в списке их {}.".format(number, len(choices))

    found = [choice for choice in choices if matches(choice, parts)]
    if len(found) == 1:
        return found[0], ""
    if not found:
        return None, ("Не нашёл модель «{}». Полный список — /change_model "
                      "без аргументов.".format(argument.strip()))
    return None, ("«{}» есть у нескольких провайдеров — уточните номером: {}.".format(
        argument.strip(), ", ".join(str(choice.number) for choice in found)))


def switch(console: Console, store: "CredentialStore", session: Session,
           choice: "Choice") -> RenderableType:
    if choice.provider.key == session.provider.key and choice.model.id == session.model.id:
        return info_panel("Модель {} уже выбрана.".format(choice.model.id),
                          title="Без изменений", style="yellow")

    was = "{} · {}".format(session.provider.name, session.model.id)
    was_window = session.context_limit
    try:
        credentials = store.ready_for(console, choice.provider)
    except LLMError as exc:
        return error_panel("Переключиться не удалось: {}\nМодель осталась прежней.".format(exc))
    except (KeyboardInterrupt, EOFError):
        return info_panel("Ввод прерван, модель осталась прежней.",
                          title="Отмена", style="yellow")

    session.agent.reconfigure(provider=choice.provider.key, model=choice.model.id,
                              api_key=credentials.key, api_extra=credentials.extra)
    return switch_notice(session, was, was_window)


def switch_notice(session: Session, was: str, was_window: int) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Было", was)
    table.add_row("Стало", "[bold]{} · {}[/]".format(session.provider.name, session.model.id))

    window = "{} токенов".format(fmt(session.context_limit))
    if session.context_limit != was_window:
        window += " (было {})".format(fmt(was_window))
    table.add_row("Окно контекста", window)
    table.add_row("История", "сохранена, {} {}".format(
        len(session.messages),
        plural(len(session.messages), ("сообщение", "сообщения", "сообщений"))))

    blocks: List[RenderableType] = [table]
    if session.is_full():
        blocks.append(Text.from_markup(
            "\n[yellow]История не помещается в окно новой модели: начало разговора "
            "будет забыто, и отвечать она станет только по остатку.[/]"
            if session.agent.config.history.on_overflow == "trim" else
            "\n[red]История не помещается в окно новой модели: следующий вопрос не "
            "примется. Нужна модель с окном побольше или /new.[/]"))
    elif session.messages:
        blocks.append(Text.from_markup(
            "\n[dim]Переспросить последний вопрос на этой модели — [/][bold]/retry[/]"))
    return Panel(Group(*blocks), title="🔀 Модель переключена", title_align="left",
                 border_style=session.provider.accent, box=box.ROUNDED, padding=(0, 1))


def model_chips(choices: List["Choice"], session: Session) -> Text:
    chips = Text()
    for choice in choices:
        if len(chips):
            chips.append("   ")
        current = (choice.provider.key == session.provider.key
                   and choice.model.id == session.model.id)
        chips.append(str(choice.number), style="bold yellow" if current else "bold cyan")
        chips.append(" " + choice.model.id, style="bold yellow" if current else "dim")
    return chips


def grouped() -> List[Tuple[ProviderInfo, List["Choice"]]]:
    order: List[Tuple[ProviderInfo, List["Choice"]]] = []
    by_provider: Dict[str, List["Choice"]] = {}
    for choice in catalog():
        if choice.provider.key not in by_provider:
            by_provider[choice.provider.key] = []
            order.append((choice.provider, by_provider[choice.provider.key]))
        by_provider[choice.provider.key].append(choice)
    return order


def matches(choice: "Choice", parts: List[str]) -> bool:
    if choice.model.id.lower() != parts[-1].lower():
        return False
    if len(parts) == 1:
        return True
    wanted = " ".join(parts[:-1]).lower()
    return (choice.provider.key.startswith(wanted)
            or choice.provider.name.lower().startswith(wanted))


def catalog() -> List["Choice"]:
    """Все модели всех провайдеров, пронумерованные подряд."""
    choices: List["Choice"] = []
    for key in PROVIDER_ORDER:
        provider = PROVIDERS[key]
        for model in provider.models:
            choices.append(Choice(len(choices) + 1, provider, model))
    return choices


class CredentialStore:
    """Реквизиты провайдеров: за время работы программы спрашиваются один раз.

    Хранит то, что ввёл человек, и ничего больше: клиенты и соединения — забота
    агента, а один и тот же ключ он переиспользует через свой реестр.
    """

    def __init__(self, demo: bool = False, ask_keys: bool = False) -> None:
        self.demo = demo
        self.ask_keys = ask_keys
        self._known: Dict[str, Credentials] = {}

    def has(self, provider: ProviderInfo) -> bool:
        return provider.key in self._known

    def remember(self, provider: ProviderInfo, credentials: Credentials) -> None:
        """Запомнить уже известные реквизиты провайдера.

        Нужно на запуске с готовым конфигом: ключ там уже есть, и если о нём
        не сказать, первое же переключение модели внутри того же провайдера
        станет спрашивать ключ заново.
        """
        self._known.setdefault(provider.key, credentials)

    def ready_for(self, console: Console, provider: ProviderInfo,
                  step: Optional[int] = None) -> Credentials:
        """Реквизиты провайдера: из памяти либо спрошенные и проверенные.

        ``step`` задаётся только на первоначальной настройке, где панели
        пронумерованы шагами; посреди диалога нумерации нет.
        """
        if provider.key in self._known:
            return self._known[provider.key]

        extra = None
        if provider.extra_field is not None:
            extra = ask_extra_field(
                console, provider, offer_saved=not self.ask_keys,
                title=None if step else "Дополнительный реквизит · {}".format(provider.name))
            if step is not None:
                step += 1

        credentials = self.connect(console, provider, extra, step)
        self._known[provider.key] = credentials
        return credentials

    def connect(self, console: Console, provider: ProviderInfo, extra: Optional[str],
                step: Optional[int]) -> Credentials:
        """Спрашивать ключ, пока он не подойдёт. Ctrl+C прерывает ввод."""
        while True:
            token = ask_token(
                console, provider, step=step or 2, offer_saved=not self.ask_keys,
                title=None if step else "{} · {}".format(provider.key_title, provider.name))
            console.print()
            with console.status("[bold]Проверяю доступ…[/]", spinner="dots"):
                try:
                    self.probe(provider, token, extra).validate_credentials()
                except LLMError as exc:
                    console.print(error_panel(str(exc)))
                    console.print("[dim]Попробуйте ввести ключ ещё раз (Ctrl+C — отмена).[/]")
                    continue
            console.print(info_panel("[green]Доступ подтверждён.[/]",
                                     title="Готово", style="green"))
            return Credentials(token, extra, "введён вручную")

    def probe(self, provider: ProviderInfo, token: str, extra: Optional[str]) -> Agent:
        """Одноразовый агент для проверки ключа. Токенов не тратит."""
        return Agent(self.config_for(provider, provider.default_model, token, extra))

    def config_for(self, provider: ProviderInfo, model: ModelInfo, key: str,
                   extra: Optional[str], base: Optional[AgentConfig] = None) -> AgentConfig:
        """Конфиг агента под выбранного провайдера с введёнными реквизитами.

        Основание сохраняется целиком: политики и бюджет из файла не должны
        пропадать оттого, что провайдера выбрали руками.
        """
        source = base if base is not None else AgentConfig()
        transport = source.transport
        if self.demo and not transport.demo:
            transport = replace(transport, demo=True)
        return source.with_changes(
            provider=provider.key, model=model.id, api_key=key, api_extra=extra,
            transport=transport)


@dataclass(frozen=True)
class Choice:
    number: int
    provider: ProviderInfo
    model: ModelInfo
