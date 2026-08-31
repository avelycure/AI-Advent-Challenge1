"""Состояние диалога: история, тема и учёт токенов."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .providers import ModelInfo, ProviderInfo
from .tokens import MESSAGE_OVERHEAD, count_message_tokens, count_text_tokens

SYSTEM_PROMPT = (
    "Ты — полезный ассистент, который общается с пользователем в терминале. "
    "Отвечай на языке пользователя, по существу и без лишней воды. "
    "Форматирование Markdown допустимо: списки, заголовки, блоки кода."
)

DEFAULT_TOPIC = "Новый диалог"


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Session:
    provider: ProviderInfo
    model: ModelInfo
    # Строка, которая уходит в поле model запроса. У Яндекса это длинный URI
    # gpt://<каталог>/<модель>/latest, поэтому на экране показываем model.id.
    model_ref: str = ""
    system_prompt: str = SYSTEM_PROMPT
    messages: List[Message] = field(default_factory=list)
    topic: str = DEFAULT_TOPIC

    exchanges: int = 0
    requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Точный размер диалога по данным API и число сообщений, которое он покрывает.
    exact_context: int = 0
    exact_upto: int = 0

    def __post_init__(self) -> None:
        if not self.model_ref:
            self.model_ref = self.model.id

    # --- история -------------------------------------------------------
    def api_messages(self) -> List[Dict[str, str]]:
        payload = [{"role": "system", "content": self.system_prompt}]
        payload += [{"role": m.role, "content": m.content} for m in self.messages]
        return payload

    def add_user(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message("assistant", content))
        self.exchanges += 1

    def drop_last_user(self) -> None:
        """Убрать неотвеченное сообщение, чтобы история не осталась битой."""
        if self.messages and self.messages[-1].role == "user":
            self.messages.pop()

    def reset(self) -> None:
        self.messages.clear()
        self.topic = DEFAULT_TOPIC
        self.exchanges = 0
        self.exact_context = 0
        self.exact_upto = 0

    # --- учёт токенов --------------------------------------------------
    def record_main_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Ответ основного диалога: и в общий счёт, и в точный размер контекста."""
        self.record_side_usage(prompt_tokens, completion_tokens)
        self.exact_context = prompt_tokens + completion_tokens
        self.exact_upto = len(self.messages)

    def record_side_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Служебный запрос (например, генерация темы): только в общий счёт."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.requests += 1

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def context_used(self) -> int:
        """Размер диалога: точные данные API плюс оценка неотправленного хвоста."""
        if self.exact_upto == 0:
            return count_message_tokens(self.api_messages())
        pending = self.messages[self.exact_upto:]
        return self.exact_context + sum(
            count_text_tokens(m.content) + MESSAGE_OVERHEAD for m in pending
        )

    @property
    def context_limit(self) -> int:
        return self.model.context_window

    @property
    def input_budget(self) -> int:
        """Сколько контекста реально доступно под историю с учётом места на ответ."""
        return max(1, self.model.context_window - self.model.output_reserve)

    def free_tokens(self) -> int:
        return max(0, self.input_budget - self.context_used())

    def fill_ratio(self) -> float:
        """Давление на бюджет истории: 1.0 — новые сообщения уже не примутся."""
        return min(1.0, self.context_used() / self.input_budget)

    def window_ratio(self) -> float:
        """Доля физического окна модели, занятая диалогом."""
        return min(1.0, self.context_used() / self.context_limit)

    def avg_exchange_tokens(self) -> int:
        """Средний прирост контекста за один обмен «вопрос — ответ»."""
        if self.exchanges == 0:
            return 0
        return max(1, round(self.context_used() / self.exchanges))

    def remaining_exchanges(self) -> int:
        average = self.avg_exchange_tokens()
        if average == 0:
            return 0
        return self.free_tokens() // average

    def is_full(self) -> bool:
        return self.free_tokens() <= 0
