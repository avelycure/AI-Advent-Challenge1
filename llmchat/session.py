"""Состояние экрана поверх агента.

Диалог, счётчики и параметры живут в агенте — здесь только то, что относится
к показу: тема в шапке и удобные для отрисовки имена. Класс намеренно оставлен
на месте: интерфейс обращался к нему десятками мест, и оборачивать агента
дешевле, чем переписывать каждую панель.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from llmagent import Agent, GenerationParams, Message
from llmagent.transport import ModelInfo, ProviderInfo

DEFAULT_TOPIC = "Новый диалог"

__all__ = ["Session", "Message", "DEFAULT_TOPIC"]


class Session:
    """Обёртка агента для терминального интерфейса."""

    def __init__(self, agent: Agent, topic: str = DEFAULT_TOPIC,
                 title: str = "") -> None:
        self.agent = agent
        self.topic = topic
        # Имя сессии — то, по которому человек её узнаёт в списке. Пустое
        # означает «придумай сам»: имя выводится из темы или первого вопроса.
        self.title = title
        # Время последней известной записи. По нему видно, не изменил ли эту
        # сессию кто-то ещё, пока мы разговаривали.
        self.seen_at: float = 0.0

    @property
    def label(self) -> str:
        """Как называть эту сессию человеку."""
        return self.title or self.topic or self.first_question or DEFAULT_TOPIC

    @property
    def first_question(self) -> str:
        for message in self.messages:
            if message.role == "user" and not message.note:
                return " ".join(message.content.split())
        return ""

    def suggest_title(self) -> str:
        """Имя, придуманное само: из темы, иначе из первого вопроса.

        Тему придумывает модель в три-пять слов — она и есть лучшее имя. Пока
        темы нет, годится начало первого вопроса: по нему разговор узнаётся.
        """
        if self.topic and self.topic != DEFAULT_TOPIC:
            return self.topic
        question = self.first_question
        if not question:
            return ""
        return question[:44].rstrip(" ,.:;—-") + ("…" if len(question) > 44 else "")

    # --- конфиг агента, как его видит экран -----------------------------
    @property
    def provider(self) -> ProviderInfo:
        return self.agent.config.provider_info

    @property
    def model(self) -> ModelInfo:
        return self.agent.config.model_info

    @property
    def model_ref(self) -> str:
        return self.agent.model_ref

    @property
    def params(self) -> GenerationParams:
        return self.agent.config.generation

    @params.setter
    def params(self, value: GenerationParams) -> None:
        self.agent.reconfigure(generation=value)

    def switch_to(self, provider: ProviderInfo, model: ModelInfo,
                  api_extra: Optional[str] = None) -> None:
        """Сменить модель, сохранив историю, тему и общий счёт токенов."""
        self.agent.reconfigure(provider=provider.key, model=model.id, api_extra=api_extra)

    # --- история --------------------------------------------------------
    @property
    def messages(self) -> List[Message]:
        return self.agent.conversation.messages

    def add_user(self, content: str) -> None:
        self.agent.conversation.add_user(content)

    def drop_last_user(self) -> None:
        self.agent.conversation.drop_last_user()

    def last_user_index(self) -> int:
        return self.agent.conversation.last_user_index()

    def reset(self) -> None:
        self.agent.reset()
        self.topic = DEFAULT_TOPIC

    @property
    def exchanges(self) -> int:
        return self.agent.conversation.exchanges

    # --- окно контекста --------------------------------------------------
    @property
    def context_limit(self) -> int:
        return self.agent.context_limit

    @property
    def output_reserve(self) -> int:
        return self.agent.output_reserve

    @property
    def input_budget(self) -> int:
        return self.agent.input_budget

    def context_used(self) -> int:
        return self.agent.context_used()

    def free_tokens(self) -> int:
        return self.agent.free_tokens()

    def fill_ratio(self) -> float:
        return self.agent.fill_ratio()

    def window_ratio(self) -> float:
        return self.agent.window_ratio()

    def avg_exchange_tokens(self) -> int:
        return self.agent.avg_exchange_tokens()

    def remaining_exchanges(self) -> int:
        return self.agent.remaining_exchanges()

    def is_full(self) -> bool:
        return self.agent.is_full()

    def breakdown(self):
        return self.agent.breakdown()

    def growth(self):
        return self.agent.growth()

    # --- расход ----------------------------------------------------------
    @property
    def requests(self) -> int:
        return self.agent.usage.requests

    @property
    def total_prompt_tokens(self) -> int:
        return self.agent.usage.prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.agent.usage.completion_tokens

    @property
    def total_reasoning_tokens(self) -> int:
        return self.agent.usage.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        return self.agent.usage.total_tokens

    @property
    def total_seconds(self) -> float:
        return self.agent.usage.seconds

    @property
    def avg_seconds(self) -> float:
        return self.agent.usage.avg_seconds

    @property
    def total_costs(self) -> Dict[str, float]:
        return self.agent.usage.costs

    @property
    def unpriced_requests(self) -> int:
        return self.agent.usage.unpriced_requests
