"""Форматы структурного ответа, схема данных и разбор с проверкой.

Модуль отвечает на один вопрос: соблюла ли модель договорённость о форме
ответа. Он ничего не знает ни об агенте, ни о провайдере — на вход текст,
на выходе разобранные данные и перечень проверок.

Схема задаётся снаружи и потому сериализуема: конфиг агента должен уметь
доехать до другого процесса в JSON или YAML, а `type` в поле этого не умеет.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Маркер, по которому провайдер обрывает генерацию (stop sequence).
# Латиница и знаки процента: в живом тексте такая последовательность не
# встречается и устойчиво разбивается на токены, а кириллический «###КОНЕЦ###»
# модель разбивала переносом строки, и стоп-строка переставала срабатывать.
STOP_MARKER = "%%END%%"

KINDS: Dict[str, type] = {"строка": str, "целое число": int, "дробное число": float,
                          "логическое": bool}


@dataclass(frozen=True)
class Field:
    """Одно поле схемы. Тип назван словом, чтобы поле пережило сериализацию."""

    name: str
    kind: str = "строка"
    max_len: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @property
    def python_type(self) -> type:
        return KINDS.get(self.kind, str)


@dataclass(frozen=True)
class Schema:
    """Форма ожидаемых данных: список объектов под ключом ``key``."""

    fields: Tuple[Field, ...] = ()
    key: str = "items"
    item_count: Optional[int] = None
    allow_extra_fields: bool = False

    @property
    def empty(self) -> bool:
        return not self.fields

    def describe(self) -> str:
        return "; ".join("{} — {}".format(f.name, f.kind) for f in self.fields)


@dataclass
class Check:
    name: str            # требование, как оно сформулировано модели
    passed: bool
    detail: str = ""     # пояснение к результату
    violation: str = ""  # что именно нарушено — для замечаний и переспроса
    # Некритичное нарушение снимается при разборе и не мешает использовать данные:
    # например, обрамление в тройные кавычки. Критичное означает, что данные
    # непригодны — не разобрались или разошлись со схемой.
    critical: bool = True


@dataclass
class Validation:
    checks: List[Check] = field(default_factory=list)
    # Данные, полученные разбором. None — разобрать не удалось.
    data: Any = None

    def add(self, name: str, passed: bool, detail: str = "", critical: bool = True,
            violation: str = "") -> None:
        self.checks.append(Check(name, passed, detail, violation, critical))

    @property
    def ok(self) -> bool:
        """Пригодны ли данные: критичные проверки все пройдены."""
        return all(c.passed for c in self.checks if c.critical)

    @property
    def flawless(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def remarks(self) -> List[str]:
        return [c.violation or c.name for c in self.checks
                if not c.passed and not c.critical]

    def failure_summary(self) -> str:
        """Перечень нарушений для переспроса модели."""
        parts = []
        for check in self.checks:
            if check.passed:
                continue
            text = check.violation or check.name
            if check.detail and not check.violation:
                text += " ({})".format(check.detail)
            parts.append(text)
        return "; ".join(parts)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)


# --------------------------------------------------------------------------
# Разбор ответа
# --------------------------------------------------------------------------

FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def marker_pattern(marker: str) -> "re.Pattern[str]":
    """Маркер с допуском на пробелы внутри.

    Модель может разбить его на несколько токенов и вставить перенос строки —
    живой GigaChat выдавал «###\\nКОНЕЦ###». Точное сравнение такой маркер
    не находит, и его хвост ломает разбор ответа.
    """
    return re.compile(r"\s*".join(re.escape(ch) for ch in marker))


def strip_marker(text: str, marker: str = STOP_MARKER) -> str:
    return marker_pattern(marker).sub("", text).strip()


def strip_fence(text: str) -> Tuple[str, bool]:
    """Снять обрамление в тройные кавычки. Второе значение — было ли обрамление."""
    match = FENCE.match(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def parse_json(text: str) -> Any:
    return json.loads(text)


def parse_yaml(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def markdown_parser(schema: Schema) -> Callable[[str], Any]:
    """Разбор таблицы Markdown в ту же структуру, что json и yaml."""

    def parse(text: str) -> Any:
        rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
        if len(rows) < 2:
            raise ValueError("таблица не найдена")
        header = [cell.strip() for cell in rows[0].strip("|").split("|")]
        if not re.match(r"^[\s|:-]+$", rows[1]):
            raise ValueError("нет строки-разделителя под заголовком")
        items = []
        for row in rows[2:]:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) != len(header):
                raise ValueError("в строке {} ячеек вместо {}".format(len(cells), len(header)))
            record: Dict[str, Any] = dict(zip(header, cells))
            # В таблице всё приходит текстом — приводим числовые поля к числам.
            for spec in schema.fields:
                if spec.python_type in (int, float) and spec.name in record:
                    record[spec.name] = spec.python_type(str(record[spec.name]).strip())
            items.append(record)
        return {schema.key: items}

    return parse


@dataclass(frozen=True)
class ResponseFormat:
    key: str
    title: str
    shape: str
    example: str
    json_mode: bool = False   # можно ли попросить провайдера через response_format

    def parser(self, schema: Schema) -> Callable[[str], Any]:
        if self.key == "json":
            return parse_json
        if self.key == "yaml":
            return parse_yaml
        return markdown_parser(schema)


FORMATS: Dict[str, ResponseFormat] = {
    "json": ResponseFormat(
        key="json", title="JSON",
        shape='{"<ключ>": [{<поля>}]}',
        example='{"items": [{"name": "Python", "year": 1991}]}',
        json_mode=True,
    ),
    "yaml": ResponseFormat(
        key="yaml", title="YAML",
        shape="<ключ>:\n  - <поле>: <значение>",
        example="items:\n  - name: Python\n    year: 1991",
    ),
    "md": ResponseFormat(
        key="md", title="таблица Markdown",
        shape="| <поле> | <поле> |\n|---|---|\n| <значение> | <значение> |",
        example="| name | year |\n|---|---|\n| Python | 1991 |",
    ),
}

FREE_FORMAT = "free"


def resolve_format(key: str) -> Optional[ResponseFormat]:
    """None означает свободный ответ: форма не оговаривается и не проверяется."""
    if key == FREE_FORMAT:
        return None
    if key not in FORMATS:
        raise KeyError("неизвестный формат «{}»; доступны: {}, {}".format(
            key, FREE_FORMAT, ", ".join(sorted(FORMATS))))
    return FORMATS[key]


# --------------------------------------------------------------------------
# Инструкция для модели
# --------------------------------------------------------------------------

def build_instruction(fmt: ResponseFormat, schema: Schema,
                      marker: Optional[str] = STOP_MARKER) -> str:
    """Текстовая часть договорённости о форме — самый сильный из рычагов.

    Живая проверка показала, что GigaChat оборачивает JSON в тройные кавычки,
    пока об этом прямо не попросишь, поэтому запрет обрамления стоит в тексте.
    """
    lines = [
        "Ты отвечаешь только данными в формате «{}» и ничем больше.".format(fmt.title),
        "",
        "Структура ответа:",
        shape_for(fmt, schema),
        "",
    ]
    if not schema.empty:
        lines += ["Поля: {}.".format(schema.describe()), ""]

    limits = ["Никаких приветствий, пояснений, выводов, обрамления в тройные кавычки "
              "и слова «{}» перед данными. Ни одного символа до и после данных.".format(fmt.key)]
    if schema.item_count is not None:
        limits.insert(0, "Ровно {} элементов — не больше и не меньше.".format(
            schema.item_count))
    for spec in schema.fields:
        if spec.max_len:
            limits.append("Поле {} — НЕ ДЛИННЕЕ {} символов. Пиши обрубленно, "
                          "лучше короче, чем полнее.".format(spec.name, spec.max_len))
        if spec.minimum is not None or spec.maximum is not None:
            limits.append("Поле {} — в пределах от {} до {}.".format(
                spec.name, spec.minimum, spec.maximum))

    lines.append("ОГРАНИЧЕНИЯ, которые важнее красоты формулировок:")
    lines += ["{}. {}".format(number, text) for number, text in enumerate(limits, 1)]
    lines += ["", "Первый символ ответа — первый символ данных."]
    if marker:
        lines.append("Сразу после последнего символа данных выведи {} и остановись.".format(
            marker))
    return "\n".join(lines)


def shape_for(fmt: ResponseFormat, schema: Schema) -> str:
    """Каркас ответа, собранный по схеме. Без схемы — общий вид формата."""
    if schema.empty:
        return fmt.shape
    if fmt.key == "json":
        body = ", ".join('"{}": <{}>'.format(f.name, f.kind) for f in schema.fields)
        return '{{"{}": [{{{}}}]}}'.format(schema.key, body)
    if fmt.key == "yaml":
        first, *rest = schema.fields
        body = ["{}:".format(schema.key), "  - {}: <{}>".format(first.name, first.kind)]
        body += ["    {}: <{}>".format(f.name, f.kind) for f in rest]
        return "\n".join(body)
    header = "| " + " | ".join(f.name for f in schema.fields) + " |"
    divider = "|" + "|".join("---" for _ in schema.fields) + "|"
    row = "| " + " | ".join("<{}>".format(f.kind) for f in schema.fields) + " |"
    return "\n".join([header, divider, row])


# --------------------------------------------------------------------------
# Проверка ответа
# --------------------------------------------------------------------------

def validate(text: str, fmt: ResponseFormat, schema: Schema, finish_reason: str = "stop",
             marker: Optional[str] = STOP_MARKER, strip_fences: bool = True) -> Validation:
    """Проверить ответ по всем требованиям: формат, схема, длина, завершение."""
    result = Validation()
    body = text.strip()

    # Маркер остановки провайдер обрезает сам, но модель могла успеть его напечатать.
    if marker:
        body = strip_marker(body, marker)
    fenced = False
    if strip_fences:
        body, fenced = strip_fence(body)
        result.add("Ответ не обрамлён в тройные кавычки", not fenced,
                   "" if not fenced else "обрамление снято при разборе", critical=False,
                   violation="модель обернула ответ в тройные кавычки")

    try:
        data = fmt.parser(schema)(body)
        result.data = data
        result.add("Разбирается как {}".format(fmt.title), True)
    except Exception as exc:  # noqa: BLE001
        result.add("Разбирается как {}".format(fmt.title), False, str(exc)[:70],
                   violation="ответ не разбирается как {}".format(fmt.title))
        _finish_check(result, finish_reason)
        return result

    if schema.empty:
        _finish_check(result, finish_reason)
        return result

    items = data.get(schema.key) if isinstance(data, dict) else None
    result.add("Структура — объект с полем {}".format(schema.key), isinstance(items, list),
               "" if isinstance(items, list) else "получено {}".format(type(data).__name__),
               violation="нет списка «{}» на верхнем уровне".format(schema.key))
    if not isinstance(items, list):
        _finish_check(result, finish_reason)
        return result

    if schema.item_count is not None:
        result.add("Ровно {} элементов".format(schema.item_count),
                   len(items) == schema.item_count, "получено {}".format(len(items)),
                   violation="элементов {}, а нужно ровно {}".format(
                       len(items), schema.item_count))

    problems = schema_problems(items, schema)
    result.add("Поля и типы соответствуют схеме", not problems,
               "; ".join(problems[:2]) + ("…" if len(problems) > 2 else ""),
               violation="; ".join(problems[:3]))
    _finish_check(result, finish_reason)
    return result


def schema_problems(items: Sequence[Any], schema: Schema) -> List[str]:
    problems: List[str] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append("элемент {} не объект".format(index))
            continue
        if not schema.allow_extra_fields:
            extra = set(item) - {spec.name for spec in schema.fields}
            if extra:
                problems.append("элемент {}: лишние поля {}".format(
                    index, ", ".join(sorted(extra))))
        for spec in schema.fields:
            problems += field_problems(index, item, spec)
    return problems


def field_problems(index: int, item: Dict[str, Any], spec: Field) -> List[str]:
    if spec.name not in item:
        return ["элемент {}: нет поля {}".format(index, spec.name)]
    value = item[spec.name]
    expected = spec.python_type
    # bool в Python — подкласс int, и без этой проверки True сошло бы за число.
    if expected is not bool and isinstance(value, bool):
        return ["элемент {}: {} не {}".format(index, spec.name, spec.kind)]
    if expected is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected):
        return ["элемент {}: {} — {}, а нужно {}".format(
            index, spec.name, type(value).__name__, spec.kind)]
    if spec.max_len and isinstance(value, str) and len(value) > spec.max_len:
        return ["элемент {}: {} длиннее {} символов".format(index, spec.name, spec.max_len)]
    if isinstance(value, (int, float)):
        if spec.minimum is not None and value < spec.minimum:
            return ["элемент {}: {} меньше {}".format(index, spec.name, spec.minimum)]
        if spec.maximum is not None and value > spec.maximum:
            return ["элемент {}: {} больше {}".format(index, spec.name, spec.maximum)]
    return []


def _finish_check(result: Validation, finish_reason: str) -> None:
    result.add("Ответ завершён самой моделью", finish_reason == "stop",
               "finish_reason={}".format(finish_reason),
               violation="ответ оборван, модель не договорила")
