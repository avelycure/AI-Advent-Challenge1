"""Структурированный профиль пользователя поверх долговременной памяти.

Профиль хранится плоскими ключами, чтобы старые ``profile.json`` продолжали
читаться, но понимает назначение ключей. Общие настройки применяются всегда,
а секции ``task.<имя>`` — когда текущий запрос совпал с их триггерами.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


PROFILE_PROMPT = (
    "Профиль пользователя. Применяй эти устойчивые персональные настройки "
    "автоматически к текущему ответу. Учитывай обращение, стиль, формат и "
    "ограничения, если они не противоречат системной инструкции:\n\n"
)

SECTIONS = ("identity", "style", "format", "constraints")
SECTION_TITLES = {
    "identity": "Обращение и личные настройки",
    "style": "Стиль ответа",
    "format": "Формат ответа",
    "constraints": "Ограничения",
}
TASK_PREFIXES = ("task", "domain")


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


@dataclass(eq=False, init=False)
class UserProfile(MutableMapping[str, str]):
    """Профиль с глобальными и зависящими от задачи предпочтениями.

    Формат ключей намеренно прост и доступен из ``/profile``::

        identity.address = Иван
        style.tone = кратко и по делу
        format.answer = Markdown со списками
        constraints.no_push = не отправлять изменения без команды
        task.feature.match = напиши фичу, реализуй фичу
        task.feature.instructions = сначала план, затем два агента параллельно

    Объект ведёт себя как словарь, сохраняя совместимость с первым форматом
    долговременной памяти, где профиль был обычным ``dict``.
    """

    _items: Dict[str, str] = field(default_factory=dict)

    def __init__(self, items: Optional[Dict[str, str]] = None) -> None:
        self._items = {
            str(key).strip(): _clean(value)
            for key, value in (items or {}).items()
            if str(key).strip() and _clean(value)
        }

    def __getitem__(self, key: str) -> str:
        return self._items[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._items[str(key)] = str(value)

    def __delitem__(self, key: str) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._items)

    @classmethod
    def restore(cls, payload: object) -> "UserProfile":
        if isinstance(payload, cls):
            return cls(payload.to_dict())
        return cls(payload if isinstance(payload, dict) else {})

    def prompt_message(self, request: str = "") -> Optional[Dict[str, str]]:
        blocks: List[str] = []
        used = set()

        for section in SECTIONS:
            values = self._section(section)
            if values:
                blocks.append("{}:\n{}".format(SECTION_TITLES[section], _block(values)))
                used.update("{}.{}".format(section, key) for key in values)

        legacy = {key: value for key, value in self._items.items()
                  if key not in used and not self._is_task_key(key)}
        if legacy:
            blocks.append("Другие нюансы пользователя:\n{}".format(_block(legacy)))

        for name, values in self._active_tasks(request):
            shown = {key: value for key, value in values.items() if key != "match"}
            if shown:
                blocks.append("Настройки для текущей задачи «{}»:\n{}".format(
                    name, _block(shown)))

        if not blocks:
            return None
        return {"role": "user", "content": PROFILE_PROMPT + "\n\n".join(blocks)}

    def active_tasks(self, request: str) -> List[str]:
        """Имена task/domain-секций, автоматически выбранных для запроса."""
        return [name for name, _ in self._active_tasks(request)]

    def _section(self, section: str) -> Dict[str, str]:
        prefix = section + "."
        return {key[len(prefix):]: value for key, value in self._items.items()
                if key.startswith(prefix) and key[len(prefix):]}

    @staticmethod
    def _is_task_key(key: str) -> bool:
        return any(key.startswith(prefix + ".") for prefix in TASK_PREFIXES)

    def _active_tasks(self, request: str):
        grouped: Dict[str, Dict[str, str]] = {}
        for key, value in self._items.items():
            parts = key.split(".", 2)
            if len(parts) != 3 or parts[0] not in TASK_PREFIXES:
                continue
            grouped.setdefault(parts[1], {})[parts[2]] = value

        wanted = request.casefold()
        active = []
        for name, values in grouped.items():
            triggers = _triggers(values.get("match", ""))
            if triggers and any(trigger.casefold() in wanted for trigger in triggers):
                active.append((name, values))
        return active


def _triggers(value: str) -> List[str]:
    normalized = value.replace(";", ",").replace("|", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _block(items: Dict[str, str]) -> str:
    return "\n".join("- {}: {}".format(key, value) for key, value in items.items())
