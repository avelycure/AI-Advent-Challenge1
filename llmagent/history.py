"""История диалога и измерение её размера в токенах.

История — часть агента, а не интерфейса: от неё зависит, что уйдёт в запрос
и сколько останется места под ответ. Интерфейс её только показывает.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from .transport import MESSAGE_OVERHEAD, count_message_tokens, count_text_tokens


# Насколько далеко поправке позволено уводить оценку от локального счёта.
MIN_SCALE = 0.5
MAX_SCALE = 2.0

# Чем подписан пересказ в запросе. Роль у него ``user``, а не ``system``:
# системных сообщений некоторые провайдеры принимают ровно одно, а два
# сообщения пользователя подряд принимают все.
SUMMARY_PREFIX = "Краткий пересказ более раннего начала этого разговора:\n\n"


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
    # Пересказ начала разговора и граница, до которой он его заменяет.
    # Сами сообщения остаются на месте: на экране разговор виден целиком, а в
    # модель вместо начала уходит пересказ. Граница — номер первого сообщения,
    # которое в пересказ ещё не вошло и потому отправляется дословно.
    summary: str = ""
    summarized: int = 0
    # Точный размер диалога по данным API — на момент последнего ответа.
    exact_context: int = 0
    # Своя оценка того же куска переписки. Хранится ради сверки: без неё нельзя
    # сказать, насколько локальный токенизатор расходится с чужой моделью.
    exact_estimate: int = 0
    # Поправка к локальной оценке, выведенная из этой сверки. Отдельным полем,
    # а не свойством: сверка относится к куску переписки, который обрезка может
    # выбросить, а поправка описывает токенизатор модели и переживает обрезку.
    scale: float = 1.0

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

    def absorb_summary(self, text: str, upto: int) -> None:
        """Заменить начало разговора пересказом по сообщение ``upto`` не включая.

        Сами сообщения остаются: разговор виден на экране целиком и целиком же
        ложится в файл сессии. Меняется только то, что уходит в модель.
        """
        self.summary = text.strip()
        self.summarized = max(0, min(upto, len(self.messages)))
        # Точный размер измерен для несжатой переписки: с этой минуты он
        # описывает не тот запрос, который уйдёт следующим.
        self.forget_exact()

    def forget_summary(self) -> None:
        self.summary = ""
        self.summarized = 0

    def drop_last_user(self) -> None:
        """Убрать неотвеченное сообщение, чтобы история не осталась битой."""
        if self.messages and self.messages[-1].role == "user":
            self.messages.pop()

    def drop_oldest_exchange(self) -> int:
        """Забыть самый старый обмен «вопрос — ответ». Возвращает число сообщений.

        Забывается самый старый из тех, что действительно уходят в модель:
        сжатое начало выбрасывать бессмысленно — в запросе его и так нет,
        а с экрана и из пересказа оно бы при этом пропало.

        Ноль означает, что забывать больше нечего: в переписке не осталось ни
        одного отвеченного вопроса, а неотвеченный — то, ради чего идёт запрос.
        """
        first_answer = next((index for index in range(self.summarized, len(self.messages))
                             if self.messages[index].role == "assistant"), -1)
        if first_answer < 0:
            return 0
        removed = first_answer + 1 - self.summarized
        del self.messages[self.summarized:first_answer + 1]
        # Точный размер измерен для прежней, более длинной переписки: оставить
        # его значило бы считать по нему обрезанный диалог.
        self.forget_exact()
        return removed

    def reset(self) -> None:
        self.messages.clear()
        self.forget_summary()
        self.forget_model()

    def forget_model(self) -> None:
        """Забыть всё, что измерено прежней моделью, — вместе с поправкой.

        После смены модели чужой токенизатор больше ничего не описывает: и
        точный размер, и выведенная из него поправка относятся к прежней.
        """
        self.forget_exact()
        self.scale = 1.0

    def forget_exact(self) -> None:
        """Забыть точный размер: описанный им кусок переписки больше не тот.

        Поправка при этом остаётся. Она про токенизатор модели, а не про эти
        конкретные сообщения, и терять её на каждой обрезке значило бы после
        каждой заново промахиваться на треть.
        """
        self.exact_context = 0
        self.exact_estimate = 0

    def note_exchange(self, prompt_tokens: int, completion_tokens: int,
                      estimate: int = 0) -> None:
        """Запомнить, во что провайдер оценил переписку, и поправить себя.

        Локально считает токенизатор ChatGPT, а отвечает Qwen, GLM или
        YandexGPT со своим: на русском тексте расхождение доходит до трети.
        Провайдер называет точное число с каждым ответом — поправка выводится
        из него и делает следующую оценку заметно ближе к правде.

        Границы поправки намеренные. Одно измерение бывает нетипичным —
        короткий первый обмен, ответ из одного слова, — и без них случайная
        цифра перекосила бы весь дальнейший счёт.
        """
        self.exact_context = prompt_tokens + completion_tokens
        self.exact_estimate = estimate
        # Молчащий провайдер существует: он не вернул usage, и его ноль — не
        # измерение. Вывести из него поправку значило бы вдвое занизить счёт
        # на пустом месте.
        if estimate and self.exact_context:
            self.scale = min(MAX_SCALE, max(MIN_SCALE, self.exact_context / estimate))

    # --- чтение ---------------------------------------------------------
    def last_user_index(self) -> int:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].role == "user":
                return index
        return -1

    def api_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        return self.api_messages_upto(len(self.messages) - 1, system_prompt)

    def api_messages_upto(self, index: int, system_prompt: str) -> List[Dict[str, str]]:
        """История по указанное сообщение включительно — для повторного запроса.

        Сжатое начало заменяется пересказом. Исключение — повтор вопроса,
        который сам попал в пересказ: пересказ ответа на него не заменит, и
        такая история собирается из настоящих сообщений, с самого начала.
        """
        payload = [{"role": "system", "content": system_prompt}] if system_prompt else []
        folded = bool(self.summary) and index >= self.summarized
        if folded:
            payload.append(self.summary_message())
        chosen = self.messages[self.summarized if folded else 0:index + 1]
        if self.keep_last_answer:
            chosen = keep_last_answer(chosen)
        payload += [{"role": m.role, "content": m.content} for m in chosen]
        return payload

    def summary_message(self) -> Dict[str, str]:
        """Пересказ в том виде, в каком он уходит в модель."""
        return {"role": "user", "content": SUMMARY_PREFIX + self.summary}

    def context_used(self, system_prompt: str) -> int:
        """Размер диалога: точные данные API плюс оценка неотправленного хвоста."""
        return self.weigh(self.api_messages(system_prompt))

    def weigh(self, messages: List[Dict[str, str]]) -> int:
        """Размер готовых сообщений с поправкой на токенизатор модели."""
        return round(count_message_tokens(messages) * self.scale)

    @property
    def exchanges(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    # --- сохранение -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_last_answer": self.keep_last_answer,
            "exact_context": self.exact_context,
            "exact_estimate": self.exact_estimate,
            "scale": self.scale,
            "summary": self.summary,
            "summarized": self.summarized,
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
        conversation.exact_estimate = int(payload.get("exact_estimate", 0) or 0)
        conversation.scale = float(payload.get("scale", 1.0) or 1.0)
        conversation.summary = str(payload.get("summary", "") or "")
        for item in payload.get("messages", []):
            conversation.messages.append(
                Message(**{k: v for k, v in item.items() if k in known}))
        # Границу ставим после сообщений и не дальше их конца: файл мог быть
        # записан другой версией, а граница за краем списка отправила бы в
        # модель один пересказ без единого сообщения.
        conversation.summarized = max(0, min(int(payload.get("summarized", 0) or 0),
                                             len(conversation.messages)))
        if not conversation.summary:
            conversation.summarized = 0
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
