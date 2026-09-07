"""Общие заготовки тестов.

Ни один тест не ходит в сеть и не тратит денег: вместо провайдера подставляется
либо ``ScriptedClient`` с заранее заданными ответами, либо штатная заглушка
``DemoClient`` с нулевой паузой.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import pytest

from llmagent import AgentConfig, Transport
from llmagent.transport import Completion, count_message_tokens, count_text_tokens


class ScriptedClient:
    """Клиент, отвечающий по написанному. Помнит, о чём его спрашивали."""

    def __init__(self, replies, finish_reason: str = "stop") -> None:
        self.replies = list(replies) if isinstance(replies, (list, tuple)) else [replies]
        self.finish_reason = finish_reason
        self.calls: List[List[Dict[str, str]]] = []
        self.kwargs: List[dict] = []

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref: str, messages, max_tokens: int, temperature: float = 0.7,
                 top_p: Optional[float] = None, stop=None, response_format=None) -> Completion:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append({"model_ref": model_ref, "max_tokens": max_tokens,
                            "temperature": temperature, "top_p": top_p, "stop": stop,
                            "response_format": response_format})
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        text = self.replies[index]
        return Completion(text, count_message_tokens(messages), count_text_tokens(text),
                          self.finish_reason, elapsed=0.001)


class FailingClient:
    """Клиент, который всегда отказывает: нужен проверке изоляции сбоев."""

    def __init__(self, error) -> None:
        self.error = error

    def validate_key(self) -> None:
        raise self.error

    def complete(self, *args, **kwargs):
        raise self.error


@pytest.fixture
def demo_config() -> AgentConfig:
    """Конфиг на заглушке без паузы: сто таких прогоняются за доли секунды."""
    return AgentConfig(name="test", provider="openai", model="gpt-5.4-mini",
                       transport=Transport(demo=True, demo_delay=0.0))


@pytest.fixture
def scripted():
    return ScriptedClient
