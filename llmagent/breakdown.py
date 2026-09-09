"""Раскладка запроса по токенам: из чего сложится ближайшее обращение к модели.

Итоговая цифра «занято 4 812 токенов» не говорит, что с этим делать. Полезно
другое: сколько стоит системный промпт, сколько накопила история и сколько
весит сам вопрос. Промпт правят один раз, историю сбрасывают, вопрос
сокращают — и решение зависит от того, что именно занимает место.

Здесь же живёт сверка оценки с фактом. Локально размер считается токенизатором
ChatGPT, а отвечает Qwen, GLM или YandexGPT со своим — расхождение неизбежно, и
единственная честная реакция на это: показать его величину, а не делать вид,
что оценка точна.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .history import Conversation
from .transport import MESSAGE_OVERHEAD, count_message_tokens, count_text_tokens


@dataclass
class RequestBreakdown:
    """Что уйдёт в модель прямо сейчас и влезет ли это в окно."""

    system: int
    # Отвеченная часть переписки: то, за что уже платили и заплатят снова.
    history: int
    # Вопрос, на который ответа ещё нет. Обычно один, но после `/new` в память
    # сессии могут лечь и заметки от под-агента.
    pending: int
    overhead: int
    reserve: int
    window: int
    # Точный размер по данным API и своя оценка того же куска переписки.
    # Ноль в первом означает «ответов ещё не было, сверять не с чем».
    measured: int = 0
    measured_estimate: int = 0
    # Поправка к локальной оценке, выведенная из этой сверки.
    scale: float = 1.0

    @property
    def estimated_input(self) -> int:
        """Размер запроса по локальной оценке — без единого обращения к API."""
        return self.system + self.history + self.pending + self.overhead

    @property
    def input_tokens(self) -> int:
        """Размер запроса, каким его считает агент: оценка с поправкой.

        Одно число, а не «факт плюс оценка хвоста»: по нему решают и здесь, и
        перед самой отправкой, и разойтись эти два решения не должны. Однажды
        разошлись — обрезка сочла, что диалог влез, а застава перед отправкой
        его отвергла, и разрешённый конфигом обрезающий агент всё равно вставал.
        """
        return round(self.estimated_input * self.scale)

    @property
    def planned(self) -> int:
        """Худший случай запроса: весь вход плюс всё, что разрешено ответу."""
        return self.input_tokens + self.reserve

    @property
    def free(self) -> int:
        return max(0, self.window - self.planned)

    @property
    def fits(self) -> bool:
        return self.planned <= self.window

    @property
    def excess(self) -> int:
        """На сколько токенов запрос не помещается в окно."""
        return max(0, self.planned - self.window)

    @property
    def drift(self) -> Optional[float]:
        """Во сколько раз оценка разошлась с фактом. None — сверять не с чем."""
        if not self.measured or not self.measured_estimate:
            return None
        return self.measured / self.measured_estimate

    def parts(self) -> List["Part"]:
        """Части запроса в порядке, в котором они уходят в модель.

        Числа здесь — своя оценка без поправки: она складывается в
        ``estimated_input``, и в отчёте столбик обязан сходиться с итогом.
        Поправка применяется один раз, к сумме.
        """
        named = [
            Part("Системный промпт", self.system),
            Part("История диалога", self.history),
            Part("Новый вопрос", self.pending),
            Part("Служебная разметка", self.overhead),
        ]
        return [part.of(self.estimated_input) for part in named if part.tokens]


@dataclass
class Part:
    """Одна часть запроса и её доля в нём."""

    title: str
    tokens: int
    share: float = 0.0

    def of(self, total: int) -> "Part":
        return Part(self.title, self.tokens, self.tokens / total if total else 0.0)


@dataclass
class Step:
    """Один обмен «вопрос — ответ» и во что он обошёлся.

    Ключевое здесь — ``prompt``: в него каждый раз уходит вся переписка целиком.
    Именно поэтому десятый вопрос дороже первого, хотя набран теми же словами.
    """

    number: int
    question: str
    prompt_tokens: int
    completion_tokens: int
    cost: Optional[float] = None
    currency: str = "USD"
    # Накопленным итогом по всем обменам с начала диалога.
    total_tokens: int = 0
    total_cost: Optional[float] = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def request_breakdown(conversation: Conversation, system_prompt: str,
                      reserve: int, window: int) -> RequestBreakdown:
    """Разложить ближайший запрос на части.

    Отвеченным считается всё до последнего вопроса без ответа: именно эта
    граница отделяет то, за что уже платили, от того, за что заплатят впервые.

    Считается по всем сообщениям подряд, а из подряд идущих ответов в запрос
    уходит только последний. Значит, раскладка бывает щедрее настоящего
    запроса, но никогда не скупее, — и «влезает» здесь означает «влезет и там».
    """
    edge = _pending_from(conversation)
    return RequestBreakdown(
        system=(count_text_tokens(system_prompt) + MESSAGE_OVERHEAD) if system_prompt else 0,
        history=_weigh_span(conversation, 0, edge),
        pending=_weigh_span(conversation, edge, len(conversation.messages)),
        overhead=count_message_tokens([]),
        reserve=reserve,
        window=window,
        measured=conversation.exact_context,
        measured_estimate=conversation.exact_estimate,
        scale=conversation.scale,
    )


def growth(conversation: Conversation) -> List[Step]:
    """Рост расхода по обменам: сколько стоил каждый и сколько набежало."""
    steps: List[Step] = []
    total_tokens = 0
    total_cost: Optional[float] = None
    for index, message in enumerate(conversation.messages):
        if message.role != "assistant":
            continue
        total_tokens += message.prompt_tokens + message.completion_tokens
        if message.cost is not None:
            total_cost = (total_cost or 0.0) + message.cost
        steps.append(Step(
            number=len(steps) + 1,
            question=_question_before(conversation, index),
            prompt_tokens=message.prompt_tokens,
            completion_tokens=message.completion_tokens,
            cost=message.cost,
            currency=message.cost_currency,
            total_tokens=total_tokens,
            total_cost=total_cost,
        ))
    return steps


def _pending_from(conversation: Conversation) -> int:
    """Номер первого неотвеченного сообщения — граница «оплачено / впервые»."""
    for index in range(len(conversation.messages) - 1, -1, -1):
        if conversation.messages[index].role == "assistant":
            return index + 1
    return 0


def _weigh_span(conversation: Conversation, start: int, stop: int) -> int:
    return sum(count_text_tokens(message.content) + MESSAGE_OVERHEAD
               for message in conversation.messages[start:stop])


def _question_before(conversation: Conversation, index: int) -> str:
    for step in range(index - 1, -1, -1):
        message = conversation.messages[step]
        if message.role == "user":
            return message.content
    return ""
