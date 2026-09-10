"""Пересказ начала разговора: что сжимать и о чём просить модель.

Здесь нет ни одного обращения к провайдеру. Модуль отвечает на два вопроса —
где провести границу между сжимаемым началом и дословным хвостом и какими
словами попросить модель этот пересказ написать. Сам запрос делает агент: у
него клиент, счётчик расхода и предел бюджета.

Пересказ пишется так, чтобы по нему можно было продолжить разговор, а не
чтобы он красиво читался. Поэтому в просьбе прямо перечислено, что терять
нельзя: имена, числа, договорённости и незакрытые вопросы. Всё это уже
терялось при обрезке — ради того сжатие и делается.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence

from .history import Conversation, Message

SUMMARY_SYSTEM = (
    "Ты сжимаешь начало диалога в короткий пересказ, по которому этот диалог "
    "можно продолжить, не видя исходных сообщений."
)

SUMMARY_REQUEST = (
    "{material}\n\n"
    "Составь пересказ этого начала разговора на языке разговора.\n"
    "Обязательно сохрани: имена и то, кого как зовут; числа, даты и названия; "
    "договорённости и решения; поставленные задачи и то, что осталось "
    "невыясненным; предпочтения собеседника.\n"
    "Опусти: вежливые обороты, повторы и подробные объяснения — оставь их "
    "выводы.\n"
    "Пиши сжатым текстом от третьего лица, до {limit} слов. "
    "Ответь только пересказом: без вступления, заголовка и пояснений."
)

PREVIOUS_SUMMARY = "Пересказ ещё более раннего начала разговора:\n\n{summary}"
NEW_MESSAGES = "Продолжение разговора, которое нужно добавить к пересказу:"

ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}

# Во сколько слов укладывать пересказ. Модели проще держать предел в словах,
# чем в токенах, а ``max_tokens`` страхует сверху: без него ответ обрывается
# на середине фразы, и пересказ выходит с оборванным хвостом.
WORDS_PER_TOKEN = 0.6

# Меньше этого куска пересказывать бессмысленно: в него не влезет и одна
# реплика, а обрезать до пары слов каждую — значит пересказать пустоту.
MIN_BLOCK = 200

# Чем помечен обрезанный кусок сообщения, которое само по себе не влезает
# в окно. Такое сообщение не сжать иначе никак: пересказать его целиком
# нельзя, а оставить как есть — значит навсегда запереть разговор.
CLIPPED = "\n…(сообщение обрезано)"


def cut_at(conversation: Conversation, keep_last: int, every: int) -> int:
    """Докуда сжимать переписку. Ноль — сжимать пока нечего.

    Граница ставится так, чтобы дословный хвост начинался с вопроса
    пользователя: ответ без своего вопроса модель прочтёт как ответ неизвестно
    на что. Поэтому граница только отодвигается назад — хвост бывает длиннее
    заказанного, но никогда не рвётся посередине обмена.
    """
    edge = len(conversation.messages) - keep_last
    edge = _back_to_question(conversation.messages, edge, conversation.summarized)
    if edge - conversation.summarized < every:
        return 0
    return edge


def build_request(previous: str, block: Sequence[Message],
                  max_tokens: int) -> List[Dict[str, str]]:
    """Сообщения запроса, которым просят пересказ."""
    material = _material(previous, block)
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": SUMMARY_REQUEST.format(
            material=material, limit=max(20, int(max_tokens * WORDS_PER_TOKEN)))},
    ]


def split_to_fit(block: Sequence[Message], budget: int,
                 weigh: Callable[[str], int]) -> List[List[Message]]:
    """Разбить сжимаемый кусок на части, каждая из которых влезет в запрос.

    Длинный разговор целиком в один запрос не помещается — а пересказать его
    надо, иначе он и дальше будет уходить в модель целиком. Части идут по
    порядку, и каждая следующая пересказывается вместе с итогом предыдущей.
    """
    parts: List[List[Message]] = []
    current: List[Message] = []
    used = 0
    for message in block:
        weight = weigh(_line(message))
        if current and used + weight > budget:
            parts.append(current)
            current, used = [], 0
        current.append(message)
        used += weight
    if current:
        parts.append(current)
    return parts


def weigh_block(block: Sequence[Message], weigh: Callable[[str], int]) -> int:
    """Во сколько токенов обходится кусок переписки, уйди он в модель дословно."""
    return sum(weigh(_line(message)) for message in block)


def clipped(message: Message, budget: int, weigh: Callable[[str], int]) -> Message:
    """Сообщение, укороченное до размера, с которым его можно пересказать.

    Нужно ровно для одного случая: одно сообщение больше целого окна модели.
    Пересказать его нельзя, а оставить как есть — значит остановить разговор
    навсегда: сжатие не сработает ни в этот раз, ни в следующий.
    """
    if weigh(message.content) <= budget:
        return message
    return replace(message, content=_clip_to(message.content, budget, weigh))


def _material(previous: str, block: Sequence[Message]) -> str:
    """Исходный материал пересказа: прежний итог и новые сообщения."""
    parts: List[str] = []
    if previous:
        parts.append(PREVIOUS_SUMMARY.format(summary=previous))
        parts.append(NEW_MESSAGES)
    parts.append("\n\n".join(_line(message) for message in block))
    return "\n\n".join(parts)


def _line(message: Message) -> str:
    return "{}: {}".format(ROLE_NAMES.get(message.role, message.role),
                           message.content)


def _clip_to(text: str, budget: int, weigh: Callable[[str], int]) -> str:
    # Символов на токен считаем по самому тексту: на кириллице и латинице
    # это разные величины, и общий коэффициент промахнулся бы вдвое.
    per_token = max(1.0, len(text) / max(1, weigh(text)))
    kept = text[:max(1, int(budget * per_token))]
    while len(kept) > 1 and weigh(kept) > budget:
        kept = kept[:int(len(kept) * 0.9)]
    return kept + CLIPPED


def _back_to_question(messages: Sequence[Message], edge: int, floor: int) -> int:
    """Отодвинуть границу назад до вопроса пользователя."""
    edge = max(floor, min(edge, len(messages)))
    while edge > floor and not _starts_a_question(messages, edge):
        edge -= 1
    return edge


def _starts_a_question(messages: Sequence[Message], edge: int) -> bool:
    return edge < len(messages) and messages[edge].role == "user"
