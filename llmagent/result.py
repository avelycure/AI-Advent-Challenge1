"""Что агент возвращает на один запрос.

Возвращается не строка, а результат со следами работы: сколько было попыток,
прошёл ли ответ проверку, что срезал провайдер, во что обошёлся запрос. Строка
не даёт вызывающему ни одного способа отличить годный ответ от негодного.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .formats import Validation


@dataclass
class AgentResult:
    agent: str = ""
    provider: str = ""
    model: str = ""
    text: str = ""
    # Разобранные данные, если выходная политика задавала форму.
    data: Any = None
    validation: Optional[Validation] = None
    # Сколько раз пришлось спросить, считая первый раз.
    attempts: int = 1
    scores: Dict[str, int] = field(default_factory=dict)
    judge_note: str = ""
    # Инструменты, которые модель вызвала сама по ходу ответа.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    # Параметры, которые провайдер не принял и которые пришлось убрать.
    dropped_params: List[str] = field(default_factory=list)
    # Что программа изменила в запросе сама, чтобы он прошёл.
    notes: List[str] = field(default_factory=list)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    cost: Optional[float] = None
    currency: str = "USD"

    # Заполняется, когда запрос не удался: в массовом прогоне падение одного
    # агента не должно ронять остальных, поэтому ошибка возвращается, а не летит.
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        if self.error is not None:
            return False
        return self.validation is None or self.validation.ok

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def mean_score(self) -> Optional[float]:
        return sum(self.scores.values()) / len(self.scores) if self.scores else None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "agent": self.agent,
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "text": self.text,
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
            "cost": self.cost,
            "currency": self.currency,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.validation is not None and self.validation.checks:
            payload["checks"] = [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.validation.checks
            ]
        if self.scores:
            payload["scores"] = self.scores
        if self.tool_calls:
            payload["tool_calls"] = [
                {"name": call.get("name"), "ok": call.get("ok"),
                 "arguments": call.get("arguments")} for call in self.tool_calls]
        if self.dropped_params:
            payload["dropped_params"] = self.dropped_params
        if self.notes:
            payload["notes"] = self.notes
        if self.error:
            payload["error"] = self.error
        return payload
