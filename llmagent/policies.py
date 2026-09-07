"""Применение входной и выходной политик.

Описание политик лежит в конфиге и потому сериализуемо; здесь — только
поведение. Разделение нужно, чтобы конфиг можно было прислать извне, а
правила его исполнения остались в одном месте и не размножались по вызовам.

Выходная политика строится тремя слоями, потому что ни одного по отдельности
не хватает:

1. **Инструкция** — текст, описывающий форму и запрещающий всё лишнее.
2. **Параметры запроса** — ``response_format``, ``stop`` и ``max_tokens``:
   работают на стороне провайдера и не зависят от послушности модели.
3. **Проверка результата** — разбор и сверка со схемой. Только она даёт право
   утверждать, что форма соблюдена, а не надеяться на это.
"""
from __future__ import annotations

from typing import List, Optional

from . import formats
from .config import InputPolicy, OutputPolicy
from .errors import InputRejected

REPAIR_REQUEST = (
    "Твой предыдущий ответ не подошёл: {violations}.\n"
    "Пришли ответ заново, целиком и с тем же содержанием, исправив только форму. "
    "Никаких извинений и пояснений — только данные."
)


# --------------------------------------------------------------------------
# Вход
# --------------------------------------------------------------------------

def apply_input(policy: InputPolicy, text: str) -> str:
    """Привести запрос к виду, пригодному для отправки, либо отклонить его.

    Отклонение — не ошибка программы, а решение агента: запрос не уходит в
    модель, и деньги за него не тратятся.
    """
    prepared = text.strip() if policy.strip else text

    if policy.forbid_empty and not prepared.strip():
        raise InputRejected("пустой запрос")

    lowered = prepared.lower()
    hit = next((word for word in policy.forbidden if word.lower() in lowered), None)
    if hit is not None:
        raise InputRejected("запрос содержит запрещённое «{}»".format(hit))

    if policy.max_chars and len(prepared) > policy.max_chars:
        if policy.on_too_long == "trim":
            prepared = prepared[:policy.max_chars]
        else:
            raise InputRejected("запрос длиннее {} символов ({})".format(
                policy.max_chars, len(prepared)))

    if policy.template:
        prepared = policy.template.format(input=prepared)
    return prepared


# --------------------------------------------------------------------------
# Выход
# --------------------------------------------------------------------------

def output_system_prompt(policy: OutputPolicy, base: str) -> str:
    """Системный промпт с приклеенной инструкцией о форме, если она задана."""
    instruction = policy.instruction
    if not instruction:
        return base
    return (base + "\n\n" + instruction) if base else instruction


def output_stop(policy: OutputPolicy, generation_stop: Optional[List[str]]) -> Optional[List[str]]:
    """Стоп-строки запроса: договорный маркер плюс заданные пользователем."""
    stops = list(generation_stop or [])
    if policy.stop_marker and policy.stop_marker not in stops:
        stops.append(policy.stop_marker)
    return stops or None


def check_output(policy: OutputPolicy, text: str, finish_reason: str) -> formats.Validation:
    """Проверить ответ. Для свободного формата проверок нет — и это не изъян."""
    fmt = formats.resolve_format(policy.format)
    if fmt is None:
        result = formats.Validation()
        result.data = None
        return result
    return formats.validate(text, fmt, policy.schema, finish_reason,
                            policy.stop_marker, policy.strip_fence)


def clean_output(policy: OutputPolicy, text: str) -> str:
    """Убрать из ответа то, о чём договорились, но что модель всё же напечатала."""
    body = text.strip()
    if policy.stop_marker:
        body = formats.strip_marker(body, policy.stop_marker)
    if policy.strip_fence:
        body, _ = formats.strip_fence(body)
    return body


def repair_request(validation: formats.Validation) -> str:
    return REPAIR_REQUEST.format(violations=validation.failure_summary())
