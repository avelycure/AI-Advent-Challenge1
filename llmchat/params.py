"""Параметры генерации, переключаемые прямо во время диалога.

Это второй слой управления ответом — тот, что работает на стороне провайдера
и не зависит от послушности модели: длина, случайность, стоп-строки и режим
структурированного вывода. Первый слой, инструкцию, задаёт текст запроса.

Значения живут в сессии: изменённые применяются ко всем последующим запросам,
пока их не сбросят.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_TEMPERATURE = 0.7


@dataclass(frozen=True)
class GenerationParams:
    """Набор параметров запроса. Значение None означает «как по умолчанию»."""

    # None — взять резерв под ответ у модели.
    max_tokens: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    response_format: Optional[str] = None   # "json_object" | "text"

    def effective_max_tokens(self, model_reserve: int) -> int:
        return self.max_tokens if self.max_tokens is not None else model_reserve

    @property
    def response_format_arg(self) -> Optional[Dict[str, str]]:
        return {"type": self.response_format} if self.response_format else None

    def changed_fields(self) -> List[Tuple[str, str]]:
        """Что отличается от значений по умолчанию — для показа пользователю."""
        default = GenerationParams()
        changed: List[Tuple[str, str]] = []
        for name in ("max_tokens", "temperature", "top_p", "stop", "response_format"):
            value = getattr(self, name)
            if value != getattr(default, name):
                changed.append((name, format_value(value)))
        return changed

    @property
    def is_default(self) -> bool:
        return not self.changed_fields()


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


# --------------------------------------------------------------------------
# Описание допустимых параметров
# --------------------------------------------------------------------------

class ParamError(ValueError):
    """Пользователь задал недопустимое значение."""


def _int_in(low: int, high: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise ParamError("нужно целое число от {} до {}".format(low, high))
        if not low <= value <= high:
            raise ParamError("допустимо от {} до {}".format(low, high))
        return value
    return parse


def _float_in(low: float, high: float) -> Callable[[str], float]:
    def parse(raw: str) -> float:
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            raise ParamError("нужно число от {} до {}".format(low, high))
        if not low <= value <= high:
            raise ParamError("допустимо от {} до {}".format(low, high))
        return value
    return parse


def _stop_list(raw: str) -> Optional[List[str]]:
    if raw.lower() in ("none", "нет", "-"):
        return None
    items = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        # Перенос строки и табуляцию в командной строке не набрать,
        # поэтому принимаем их привычной записью \n и \t.
        items.append(part.replace("\\n", "\n").replace("\\t", "\t"))
    if not items:
        raise ParamError("укажите строку или несколько через |")
    if len(items) > 4:
        raise ParamError("не больше четырёх стоп-строк")
    return items


def _response_format(raw: str) -> Optional[str]:
    value = raw.lower()
    if value in ("none", "нет", "-", "text"):
        return None if value != "text" else "text"
    if value in ("json", "json_object"):
        return "json_object"
    raise ParamError("допустимо json_object, text или none")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    parse: Callable[[str], Any]
    description: str
    examples: List[str]


SPECS: Dict[str, ParamSpec] = {
    spec.name: spec for spec in (
        ParamSpec("max_tokens", _int_in(1, 32000),
                  "предел длины ответа; на нём генерация обрывается",
                  ["max_tokens=200", "max_tokens=4096"]),
        ParamSpec("temperature", _float_in(0.0, 2.0),
                  "случайность: ниже — предсказуемее, выше — разнообразнее",
                  ["temperature=0", "temperature=0.7", "temperature=1.5"]),
        ParamSpec("top_p", _float_in(0.0, 1.0),
                  "доля самых вероятных продолжений, из которых идёт выбор",
                  ["top_p=0.1 — почти без разброса", "top_p=1 — без отсечения"]),
        ParamSpec("stop", _stop_list,
                  "строки, на которых генерация обрывается",
                  ["stop=###КОНЕЦ###", "stop=Вопрос:|Ответ:",
                   "stop=\\n\\n — оборвать на пустой строке", "stop=none — снять"]),
        ParamSpec("response_format", _response_format,
                  "режим вывода на стороне провайдера",
                  ["response_format=json_object — только валидный JSON",
                   "response_format=text — обычный текст",
                   "response_format=none — не передавать параметр"]),
    )
}

RESET_WORDS = {"reset", "сброс", "default", "defaults", "по-умолчанию"}


def parse_command(argument: str) -> Tuple[Optional[GenerationParams], List[str], bool]:
    """Разобрать аргументы команды.

    Возвращает три значения: словарь изменений (или None, если менять нечего),
    список сообщений об ошибках и признак запроса на сброс.
    """
    try:
        # Удваиваем обратные слэши до разбора: иначе shlex съест их сам,
        # и запись stop=\n превратится в букву n вместо переноса строки.
        # Кавычки при этом продолжают работать как обычно.
        tokens = shlex.split(argument.replace("\\", "\\\\"))
    except ValueError:
        return None, ["не удалось разобрать строку: проверьте кавычки"], False

    if not tokens:
        return None, [], False

    if len(tokens) == 1 and tokens[0].lower() in RESET_WORDS:
        return None, [], True

    updates: Dict[str, Any] = {}
    errors: List[str] = []
    for token in tokens:
        if "=" not in token:
            errors.append("«{}» — нужен вид имя=значение".format(token))
            continue
        name, raw = token.split("=", 1)
        name = name.strip().lower()
        spec = SPECS.get(name)
        if spec is None:
            errors.append("неизвестный параметр «{}»; доступны: {}".format(
                name, ", ".join(sorted(SPECS))))
            continue
        try:
            updates[name] = spec.parse(raw.strip())
        except ParamError as exc:
            errors.append("{}: {}".format(name, exc))

    if not updates:
        return None, errors, False
    return updates, errors, False


def apply(params: GenerationParams, updates: Dict[str, Any]) -> GenerationParams:
    return replace(params, **updates)
