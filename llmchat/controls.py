"""Управление форматом, длиной и завершением ответа модели.

Контроль строится тремя слоями, потому что ни одного из них по отдельности
не хватает:

1. **Инструкция** — текст, который описывает формат и запрещает всё лишнее.
   Самый сильный рычаг: живая проверка показала, что GigaChat оборачивает JSON
   в тройные кавычки, пока об этом прямо не попросишь.
2. **Параметры запроса** — ``response_format``, ``stop`` и ``max_tokens``.
   Работают на стороне провайдера и не зависят от послушности модели.
3. **Проверка результата** — разбор ответа и сверка со схемой. Только она
   даёт право утверждать, что формат соблюдён, а не надеяться на это.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Маркер, по которому провайдер обрывает генерацию (stop sequence).
STOP_MARKER = "###КОНЕЦ###"
ITEM_COUNT = 5
NOTE_LIMIT = 60
CONSTRAINED_MAX_TOKENS = 260


@dataclass(frozen=True)
class Field:
    name: str
    kind: str          # "строка" | "целое число"
    python_type: type
    max_len: Optional[int] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None


SCHEMA: List[Field] = [
    Field("name", "строка", str, max_len=30),
    Field("year", "целое число", int, minimum=1930, maximum=2030),
    Field("note", "строка", str, max_len=NOTE_LIMIT),
]


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

    def add(self, name: str, passed: bool, detail: str = "", critical: bool = True,
            violation: str = "") -> None:
        self.checks.append(Check(name, passed, detail, violation, critical))

    @property
    def ok(self) -> bool:
        """Пригодны ли данные: критичные проверки все пройдены."""
        return bool(self.checks) and all(c.passed for c in self.checks if c.critical)

    @property
    def flawless(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

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

# Маркер приходится искать с допуском на пробелы внутри: модель может разбить его
# на несколько токенов и вставить перенос строки — живой GigaChat выдавал
# «###\nКОНЕЦ###». Точное сравнение такой маркер не находит, и его хвост ломает
# разбор ответа. По той же причине провайдер не срабатывает по стоп-строке.
MARKER_PATTERN = re.compile(r"\s*".join(re.escape(ch) for ch in STOP_MARKER))


def strip_marker(text: str) -> str:
    return MARKER_PATTERN.sub("", text).strip()


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


def parse_markdown(text: str) -> Any:
    """Разобрать таблицу Markdown в ту же структуру, что json и yaml."""
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
        record = dict(zip(header, cells))
        # В таблице всё приходит текстом — приводим числовые поля к числам.
        for spec in SCHEMA:
            if spec.python_type is int and spec.name in record:
                record[spec.name] = int(str(record[spec.name]).strip())
        items.append(record)
    return {"items": items}


@dataclass(frozen=True)
class ResponseFormat:
    key: str
    title: str
    example: str
    shape: str
    parse: Callable[[str], Any]
    json_mode: bool = False   # можно ли попросить провайдера через response_format


FORMATS: Dict[str, ResponseFormat] = {
    "json": ResponseFormat(
        key="json", title="JSON",
        shape='{"items": [{"name": <строка>, "year": <целое>, "note": <строка>}]}',
        example='{"items": [{"name": "Python", "year": 1991, "note": "читаемый синтаксис"}]}',
        parse=parse_json, json_mode=True,
    ),
    "yaml": ResponseFormat(
        key="yaml", title="YAML",
        shape="items:\n  - name: <строка>\n    year: <целое>\n    note: <строка>",
        example="items:\n  - name: Python\n    year: 1991\n    note: читаемый синтаксис",
        parse=parse_yaml,
    ),
    "md": ResponseFormat(
        key="md", title="таблица Markdown",
        shape="| name | year | note |\n|---|---|---|\n| <строка> | <целое> | <строка> |",
        example="| name | year | note |\n|---|---|---|\n| Python | 1991 | читаемый синтаксис |",
        parse=parse_markdown,
    ),
}


# --------------------------------------------------------------------------
# Инструкция для модели
# --------------------------------------------------------------------------

def build_instruction(fmt: ResponseFormat) -> str:
    fields = "; ".join("{} — {}".format(spec.name, spec.kind) for spec in SCHEMA)
    return (
        "Ты отвечаешь только данными в формате «{title}» и ничем больше.\n\n"
        "Структура ответа:\n{shape}\n\n"
        "Пример правильного ответа:\n{example}\n\n"
        "Поля: {fields}.\n\n"
        "ОГРАНИЧЕНИЯ, которые важнее красоты формулировок:\n"
        "1. Ровно {count} элементов — не больше и не меньше.\n"
        "2. Поле note — НЕ ДЛИННЕЕ {limit} символов. Это примерно шесть слов. "
        "Пиши обрубленно: «читаемый синтаксис», а не развёрнутое предложение. "
        "Лучше короче, чем полнее.\n"
        "3. Никаких приветствий, пояснений, выводов, обрамления в тройные кавычки "
        "и слова «{key}» перед данными. Ни одного символа до и после данных.\n\n"
        "Первый символ ответа — первый символ данных.\n"
        "Сразу после последнего символа данных выведи {marker} и остановись."
    ).format(title=fmt.title, shape=fmt.shape, example=fmt.example, fields=fields,
             count=ITEM_COUNT, limit=NOTE_LIMIT, key=fmt.key, marker=STOP_MARKER)


@dataclass
class Controls:
    """Набор ограничений, который отличает управляемый запрос от свободного."""
    fmt: ResponseFormat
    max_tokens: int = CONSTRAINED_MAX_TOKENS
    stop: List[str] = field(default_factory=lambda: [STOP_MARKER])

    @property
    def instruction(self) -> str:
        return build_instruction(self.fmt)

    @property
    def response_format(self) -> Optional[Dict[str, str]]:
        return {"type": "json_object"} if self.fmt.json_mode else None


# --------------------------------------------------------------------------
# Проверка ответа
# --------------------------------------------------------------------------

def validate(text: str, fmt: ResponseFormat, finish_reason: str) -> Validation:
    """Проверить ответ по всем требованиям: формат, схема, длина, завершение."""
    result = Validation()
    raw = text.strip()

    # Маркер остановки провайдер обрезает сам, но модель могла успеть его напечатать.
    body = strip_marker(raw)
    body, fenced = strip_fence(body)

    result.add("Ответ не обрамлён в тройные кавычки", not fenced,
               "" if not fenced else "обрамление снято при разборе", critical=False,
               violation="модель обернула ответ в тройные кавычки")

    try:
        data = fmt.parse(body)
        result.add("Разбирается как {}".format(fmt.title), True)
    except Exception as exc:  # noqa: BLE001
        result.add("Разбирается как {}".format(fmt.title), False, str(exc)[:70])
        result.add("Структура — объект с полем items", False, "ответ не разобран")
        result.add("Ровно {} элементов".format(ITEM_COUNT), False, "ответ не разобран")
        result.add("Поля и типы соответствуют схеме", False, "ответ не разобран")
        result.add("Ответ завершён самой моделью", finish_reason == "stop",
                   "finish_reason={}".format(finish_reason))
        return result

    items = data.get("items") if isinstance(data, dict) else None
    result.add("Структура — объект с полем items", isinstance(items, list),
               "" if isinstance(items, list) else "получено {}".format(type(data).__name__))
    if not isinstance(items, list):
        result.add("Ровно {} элементов".format(ITEM_COUNT), False, "нет списка items")
        result.add("Поля и типы соответствуют схеме", False, "ответ не разобран")
        result.add("Ответ завершён самой моделью", finish_reason == "stop",
                   "finish_reason={}".format(finish_reason))
        return result

    result.add("Ровно {} элементов".format(ITEM_COUNT), len(items) == ITEM_COUNT,
               "получено {}".format(len(items)))

    problems: List[str] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append("элемент {} не объект".format(index))
            continue
        extra = set(item) - {spec.name for spec in SCHEMA}
        if extra:
            problems.append("элемент {}: лишние поля {}".format(index, ", ".join(sorted(extra))))
        for spec in SCHEMA:
            if spec.name not in item:
                problems.append("элемент {}: нет поля {}".format(index, spec.name))
                continue
            value = item[spec.name]
            if spec.python_type is int and isinstance(value, bool):
                problems.append("элемент {}: {} не число".format(index, spec.name))
            elif not isinstance(value, spec.python_type):
                problems.append("элемент {}: {} — {}, а нужно {}".format(
                    index, spec.name, type(value).__name__, spec.kind))
            elif spec.max_len and isinstance(value, str) and len(value) > spec.max_len:
                problems.append("элемент {}: {} длиннее {} символов".format(
                    index, spec.name, spec.max_len))
            elif spec.minimum is not None and isinstance(value, int) and not (
                    spec.minimum <= value <= spec.maximum):
                problems.append("элемент {}: {} вне диапазона".format(index, spec.name))

    result.add("Поля и типы соответствуют схеме", not problems,
               "; ".join(problems[:2]) + ("…" if len(problems) > 2 else ""))
    result.add("Ответ завершён самой моделью", finish_reason == "stop",
               "finish_reason={}".format(finish_reason))
    return result
