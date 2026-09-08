"""Инструменты, которые модель может вызвать сама.

Устройство разделено надвое намеренно. Какие инструменты доступны — это данные,
и они живут в конфиге агента. Что инструмент делает — это код, и его передаёт
вызывающий: запуск под-агента отдельным процессом умеет интерфейс, а коробка
о процессах и экранах знать не должна.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .errors import AgentError

# Имя штатного инструмента: запустить под-агента с выбранным конфигом.
SPAWN_AGENT = "spawn_agent"


# Ключ, под которым инструмент сообщает о своём расходе. Инструмент может сам
# обращаться к модели — под-агент именно так и работает, — и платит за это тот
# же кошелёк. Без такого сообщения расход терялся бы: команда его учитывала,
# а вызов моделью — нет.
USAGE_KEY = "usage"


@dataclass
class ToolOutcome:
    """Что инструмент вернул модели и что показать человеку."""

    text: str
    ok: bool = True
    # Подробности для показа: заголовок панели, расход, идентификатор сессии.
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        """Расход инструмента, если он о нём сообщил."""
        spent = self.detail.get(USAGE_KEY)
        return spent if isinstance(spent, dict) else None


@dataclass(frozen=True)
class ToolSpec:
    """Описание инструмента: как его назвать модели и что вызвать в ответ."""

    name: str
    description: str
    # Схема аргументов в том виде, в каком её понимают провайдеры.
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], ToolOutcome]

    def definition(self) -> Dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class Toolbox:
    """Набор инструментов, доступных агенту."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: Dict[str, ToolSpec] = {spec.name: spec for spec in specs}

    def __len__(self) -> int:
        return len(self._specs)

    def __bool__(self) -> bool:
        return bool(self._specs)

    @property
    def names(self) -> List[str]:
        return sorted(self._specs)

    def only(self, allowed: Sequence[str]) -> "Toolbox":
        """Оставить перечисленные в конфиге. Неизвестное имя — ошибка конфига.

        Молчать нельзя: человек написал имя инструмента, ждёт, что он появится,
        а модель о нём не узнает и просто ответит своими словами.
        """
        unknown = [name for name in allowed if name not in self._specs]
        if unknown and not self._specs:
            # Набор пуст — значит инструменты не передал вызывающий, и винить
            # конфиг было бы неверно: имя в нём может быть совершенно правильным.
            raise AgentError(
                "конфиг включает инструменты ({}), но вызывающий их не передал: "
                "агенту нужен toolbox".format(", ".join(unknown)))
        if unknown:
            raise AgentError("нет инструментов {}; доступны: {}".format(
                ", ".join(unknown), ", ".join(self.names)))
        return Toolbox(self._specs[name] for name in allowed)

    def definitions(self) -> Optional[List[Dict[str, Any]]]:
        """Описания для запроса. None — инструментов нет, поле не отправляем."""
        return [spec.definition() for spec in self._specs.values()] or None

    def run(self, name: str, arguments: str) -> ToolOutcome:
        """Выполнить инструмент. Любая беда возвращается моделью, а не летит.

        Модель сочиняет и имя, и аргументы, поэтому ошибиться может в обоих.
        Ей об ошибке надо сказать словами — тогда она поправится сама; падение
        же оборвало бы разговор из-за чужой опечатки.
        """
        spec = self._specs.get(name)
        if spec is None:
            return ToolOutcome(
                "Инструмента «{}» нет. Доступны: {}.".format(
                    name, ", ".join(self.names) or "ни одного"), ok=False)
        try:
            parsed = json.loads(arguments or "{}")
        except ValueError:
            return ToolOutcome(
                "Аргументы не разобраны как JSON. Пришли их заново.", ok=False)
        if not isinstance(parsed, dict):
            return ToolOutcome("Аргументы должны быть объектом.", ok=False)
        try:
            return spec.handler(parsed)
        except Exception as exc:  # noqa: BLE001 — беда инструмента не рушит разговор
            return ToolOutcome("Инструмент не справился: {}".format(exc), ok=False)


def spawn_agent_spec(configs: Sequence[str],
                     handler: Callable[[Dict[str, Any]], ToolOutcome]) -> ToolSpec:
    """Описание инструмента «запустить под-агента».

    Перечень конфигов подставляется в описание, потому что модель выбирает из
    него: без перечня она придумывала бы имена.
    """
    listed = ", ".join(configs) or "ни одного"
    config_field: Dict[str, Any] = {"type": "string",
                                    "description": "имя конфига под-агента"}
    if configs:
        # Перечень допустимых значений держит модель в рамках имеющихся
        # конфигов. Пустой enum схему бы сломал, поэтому его просто нет.
        config_field["enum"] = list(configs)
    return ToolSpec(
        name=SPAWN_AGENT,
        description=(
            "Запустить отдельного под-агента со своим конфигом и задать ему один "
            "вопрос. У под-агента своя память: он не видит этот разговор, поэтому "
            "вопрос должен быть самодостаточным. Вернётся только его ответ. "
            "Годится, когда нужен другой навык или строгая форма ответа. "
            "Доступные конфиги: {}.".format(listed)),
        parameters={
            "type": "object",
            "properties": {
                "config": config_field,
                "question": {"type": "string",
                             "description": "самодостаточный вопрос под-агенту"},
            },
            "required": ["config", "question"],
        },
        handler=handler,
    )
