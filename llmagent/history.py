"""История диалога и измерение её размера в токенах.

История — часть агента, а не интерфейса: от неё зависит, что уйдёт в запрос
и сколько останется места под ответ. Интерфейс её только показывает.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from .transport import MESSAGE_OVERHEAD, count_message_tokens, count_text_tokens


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    # Чей это ответ. Подпись хранится в самом сообщении, а не берётся из конфига:
    # иначе после смены модели прежние ответы переклеились бы её именем.
    model: str = ""
    accent: str = ""
    # Что стоил этот ответ. Метрики держатся при сообщении, чтобы после смены
    # модели можно было сравнить ответы между собой.
    elapsed: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Optional[float] = None
    cost_currency: str = "USD"
    # Заметка — сообщение, которое не спрашивал пользователь и не писала
    # модель: например, ответ под-агента, принятый в память сессии.
    note: bool = False


@dataclass
class Conversation:
    """Переписка агента с пользователем и точный размер отправленного."""

    keep_last_answer: bool = True
    messages: List[Message] = field(default_factory=list)
    # Точный размер диалога по данным API и число сообщений, которое он покрывает.
    exact_context: int = 0
    exact_upto: int = 0

    def __len__(self) -> int:
        return len(self.messages)

    # --- изменение ------------------------------------------------------
    def add_user(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_assistant(self, content: str, model: str = "", accent: str = "") -> Message:
        message = Message("assistant", content, model=model, accent=accent)
        self.messages.append(message)
        return message

    def add_note(self, content: str, model: str = "", accent: str = "") -> Message:
        """Внешнее сведение, попадающее в память сессии.

        Роль ``user``, а не ``assistant``: это не ответ модели, а материал,
        который ей дали. Провайдеры принимают два сообщения пользователя
        подряд, и подделывать под ответ ассистента было бы неправдой.
        """
        message = Message("user", content, model=model, accent=accent, note=True)
        self.messages.append(message)
        return message

    def drop_last_user(self) -> None:
        """Убрать неотвеченное сообщение, чтобы история не осталась битой."""
        if self.messages and self.messages[-1].role == "user":
            self.messages.pop()

    def reset(self) -> None:
        self.messages.clear()
        self.exact_context = 0
        self.exact_upto = 0

    def forget_exact(self) -> None:
        """Забыть точный размер: после смены модели он измерен чужим токенизатором."""
        self.exact_context = 0
        self.exact_upto = 0

    def note_exchange(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.exact_context = prompt_tokens + completion_tokens
        self.exact_upto = len(self.messages)

    # --- чтение ---------------------------------------------------------
    def last_user_index(self) -> int:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].role == "user":
                return index
        return -1

    def api_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        return self.api_messages_upto(len(self.messages) - 1, system_prompt)

    def api_messages_upto(self, index: int, system_prompt: str) -> List[Dict[str, str]]:
        """История по указанное сообщение включительно — для повторного запроса."""
        payload = [{"role": "system", "content": system_prompt}] if system_prompt else []
        chosen = self.messages[:index + 1]
        if self.keep_last_answer:
            chosen = keep_last_answer(chosen)
        payload += [{"role": m.role, "content": m.content} for m in chosen]
        return payload

    def context_used(self, system_prompt: str) -> int:
        """Размер диалога: точные данные API плюс оценка неотправленного хвоста."""
        if self.exact_upto == 0:
            return count_message_tokens(self.api_messages(system_prompt))
        pending = self.messages[self.exact_upto:]
        return self.exact_context + sum(
            count_text_tokens(m.content) + MESSAGE_OVERHEAD for m in pending
        )

    @property
    def exchanges(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    # --- сохранение -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_last_answer": self.keep_last_answer,
            "exact_context": self.exact_context,
            "exact_upto": self.exact_upto,
            "messages": [asdict(message) for message in self.messages],
        }

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "Conversation":
        """Собрать переписку из сохранённого вида.

        Незнакомые поля сообщений пропускаются: файл мог быть записан прежней
        версией программы, и терять из-за этого весь диалог не стоит.
        """
        known = {f.name for f in fields(Message)}
        conversation = cls(keep_last_answer=bool(payload.get("keep_last_answer", True)))
        conversation.exact_context = int(payload.get("exact_context", 0) or 0)
        conversation.exact_upto = int(payload.get("exact_upto", 0) or 0)
        for item in payload.get("messages", []):
            conversation.messages.append(
                Message(**{k: v for k, v in item.items() if k in known}))
        return conversation


def keep_last_answer(messages: List[Message]) -> List[Message]:
    """Из подряд идущих ответов оставить последний.

    Ответы копятся, когда один вопрос переспрашивают на разных моделях: в истории
    их видно все, но в запрос уходит только свежий — иначе модель получила бы
    несколько ответов на один свой вопрос.
    """
    kept: List[Message] = []
    for message in messages:
        if kept and message.role == "assistant" and kept[-1].role == "assistant":
            kept[-1] = message
            continue
        kept.append(message)
    return kept
