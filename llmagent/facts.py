"""Память фактов: что спросить у модели и как разобрать её ответ.

Здесь нет ни одного обращения к провайдеру — ровно как в ``compression``.
Модуль отвечает на три вопроса: какими словами попросить факты, что считать
фактом и как превратить ответ модели в пары «ключ — значение». Сам запрос
делает агент: у него клиент, счётчик расхода и предел бюджета.

Отличие от пересказа — в форме. Пересказ это проза, и факт в ней теряется
незаметно: второй пересказ пишется по первому, и «бюджет 300 тысяч»
превращается в «обсуждали бюджет». Факт с ключом так не теряется — он либо
есть, либо его нет, и это видно глазом.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence

from .compression import clipped
from .history import Message

FACTS_SYSTEM = (
    "Ты ведёшь картотеку важных сведений о разговоре: коротких пар "
    "«ключ — значение». Ты отвечаешь только объектом JSON и ничем больше."
)

FACTS_REQUEST = (
    "{material}\n\n"
    "Обнови картотеку по этому продолжению разговора.\n"
    "Держи в ней только то, что понадобится дальше: цель и задачу; "
    "ограничения — сроки, деньги, объём; принятые решения и договорённости; "
    "предпочтения и запреты собеседника; имена, числа, названия и версии.\n"
    "Не держи: пересказ ответов, рассуждения, общие сведения о предмете, "
    "вежливые обороты.\n"
    "Ключ — два-три слова на языке разговора, значение — до {limit} слов. "
    "Если сведение изменилось, дай тот же ключ с новым значением. Если "
    "отменено — тот же ключ с пустой строкой. Неизменившееся можно не "
    "повторять.\n"
    "Ответь одним объектом JSON вида {{\"ключ\": \"значение\"}}: без "
    "пояснений, без обрамления в кавычки и без вложенных объектов."
)

PREVIOUS_FACTS = "Картотека на сейчас:\n\n{facts}"
EMPTY_FACTS = "Картотека пока пуста."
NEW_MESSAGES = "Продолжение разговора:"

ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}

# До скольких слов просить значение факта. Предел нужен не ради экономии, а
# ради формы: без него модель пишет в значение абзац пересказа, и картотека
# превращается в тот же пересказ, только с заголовками.
VALUE_WORDS = 12

# Сколько знаков значения хранить. Предел последний и грубый: он страхует от
# модели, которая просьбу про слова прочла по-своему.
VALUE_CHARS = 200


def build_request(previous: Dict[str, str], block: Sequence[Message]) -> List[Dict[str, str]]:
    """Сообщения запроса, которым просят обновить картотеку."""
    return [
        {"role": "system", "content": FACTS_SYSTEM},
        {"role": "user", "content": FACTS_REQUEST.format(
            material=_material(previous, block), limit=VALUE_WORDS)},
    ]


def fitting(block: Sequence[Message], budget: int,
            weigh: Callable[[str], int]) -> List[Message]:
    """Из новых сообщений — самые свежие, которые влезут в запрос за фактами.

    Влезает не всё, когда разбор несколько раз подряд не удался: непросмотренное
    копится, и на очередной попытке весь этот ком уходил бы в один запрос,
    который сам не помещается в окно. Тогда разбираются самые свежие сообщения:
    в них то, о чём говорят сейчас, а остальные и без того остались бы
    неразобранными.
    """
    left = budget
    kept: List[Message] = []
    for message in reversed(list(block)):
        left -= weigh(_line(message))
        if left < 0:
            break
        kept.append(message)
    kept.reverse()
    # Одно сообщение шире всего запроса — не повод не разобрать ничего: оно
    # обрезается ровно так же, как слишком длинное сообщение при пересказе.
    if not kept and block:
        kept = [clipped(block[-1], max(1, budget), weigh)]
    return kept


def parse(text: str) -> Dict[str, str]:
    """Разобрать ответ модели в пары «ключ — значение».

    Разбор терпимый намеренно. Модель, которую попросили ответить одним
    объектом JSON, всё равно то обернёт его в тройные кавычки, то припишет
    «Вот обновлённая картотека:». Отвергать из-за этого всю картотеку значило
    бы терять сведения там, где они на самом деле получены.

    Нетерпим разбор к одному: к значениям, которые фактами не являются.
    Вложенный объект, список и число складываются в строку, а пустое значение
    остаётся пустым — им факт отменяют.
    """
    payload = _json_object(text)
    if payload is None:
        raise ValueError("в ответе модели нет объекта JSON")
    facts: Dict[str, str] = {}
    for key, value in payload.items():
        name = " ".join(str(key).split())
        if name:
            facts[name] = _flat(value)
    return facts


def _material(previous: Dict[str, str], block: Sequence[Message]) -> str:
    parts = [PREVIOUS_FACTS.format(facts=render(previous)) if previous else EMPTY_FACTS]
    parts.append(NEW_MESSAGES)
    parts.append("\n\n".join(_line(message) for message in block))
    return "\n\n".join(parts)


def render(facts: Dict[str, str]) -> str:
    """Картотека строками «— ключ: значение» — так её видит и модель, и человек."""
    return "\n".join("- {}: {}".format(key, value) for key, value in facts.items())


def _line(message: Message) -> str:
    return "{}: {}".format(ROLE_NAMES.get(message.role, message.role),
                           message.content)


def _json_object(text: str) -> Dict[str, object] | None:
    """Первый объект JSON в тексте ответа. None — объекта нет.

    Ищется по границам самых внешних фигурных скобок, а не разбором всего
    текста: пояснение до и после объекта встречается чаще, чем чистый ответ.
    """
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start:end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _flat(value: object) -> str:
    """Значение факта одной строкой, не длиннее отведённого."""
    if value is None or isinstance(value, bool):
        text = "" if value is None else ("да" if value else "нет")
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_flat(item) for item in value)
    elif isinstance(value, dict):
        text = "; ".join("{}: {}".format(key, _flat(item)) for key, item in value.items())
    else:
        text = str(value)
    text = " ".join(text.split())
    return text[:VALUE_CHARS].rstrip()
