"""Общие заготовки тестов.

Ни один тест не ходит в сеть и не тратит денег: вместо провайдера подставляется
либо ``ScriptedClient`` с заранее заданными ответами, либо штатная заглушка
``DemoClient`` с нулевой паузой.
"""
from __future__ import annotations

import pathlib
import time
from typing import Dict, List, Optional

import pytest

from llmagent import AgentConfig, Transport
from llmagent.transport import Completion, count_message_tokens, count_text_tokens

ROOT = pathlib.Path(__file__).resolve().parent.parent


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
                 top_p: Optional[float] = None, stop=None, response_format=None,
                 tools=None) -> Completion:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append({"model_ref": model_ref, "max_tokens": max_tokens,
                            "temperature": temperature, "top_p": top_p, "stop": stop,
                            "response_format": response_format, "tools": tools})
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


class ToolCallingClient:
    """Модель, которая зовёт инструмент, пока он предложен, и потом отвечает.

    Настоящий провайдер не может попросить вызов, когда инструменты ему не
    предложены, — заглушка ведёт себя так же, иначе проверка предела кругов
    мерила бы небывалое.
    """

    def __init__(self, name: str, arguments: str, answer: str = "Готово.",
                 rounds: int = 99) -> None:
        self.name = name
        self.arguments = arguments
        self.answer = answer
        self.rounds = rounds
        self.offered: List[bool] = []
        self.calls: List[List[Dict[str, object]]] = []

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref: str, messages, max_tokens: int, temperature: float = 0.7,
                 top_p: Optional[float] = None, stop=None, response_format=None,
                 tools=None) -> Completion:
        from llmagent.transport import ToolCall

        self.offered.append(bool(tools))
        self.calls.append([dict(m) for m in messages])
        asked = sum(1 for flag in self.offered if flag)
        if tools and asked <= self.rounds:
            return Completion("", count_message_tokens(messages), 5, elapsed=0.001,
                              tool_calls=[ToolCall("call-1", self.name, self.arguments)])
        return Completion(self.answer, count_message_tokens(messages),
                          count_text_tokens(self.answer), elapsed=0.001)


# --------------------------------------------------------------------------
# Дочерние процессы
# --------------------------------------------------------------------------

REAL_SESSIONS = pathlib.Path("~/.llm-agent").expanduser()


def child_env(home) -> Dict[str, str]:
    """Окружение дочернего процесса тестов. Домашний каталог обязателен.

    Обязателен он потому, что этот параметр забыли дважды и дважды поплатились.
    Окружение задаётся с нуля, поэтому переменные вида ``*_API_KEY`` в него не
    попадают — но ``HOME`` Python восстанавливает из системной записи
    пользователя, и без подмены агент находит настоящий ключ в ``~/.openai-key``
    и уходит в платный запрос, а сессии пишет в настоящий ``~/.llm-agent``.
    """
    if home is None:
        raise AssertionError("дочернему процессу нужен свой домашний каталог: "
                             "передайте tmp_path")
    return {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "TERM": "dumb",
            "COLUMNS": "120", "HOME": str(home)}


@pytest.fixture(autouse=True)
def no_provider_keys_in_the_environment(monkeypatch):
    """Убрать реквизиты провайдеров из окружения на время теста.

    Ни один тест не должен полагаться на то, есть ли ключ у запускающего.
    Оболочка легко передаёт его дальше — так один тест начал находить
    настоящий ключ и перестал проверять то, ради чего написан. А в худшем
    случае найденный ключ уводит проверку в платный запрос.
    """
    from llmagent.registry import EXTRA_ENV
    from llmagent.transport import PROVIDERS

    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider.api_key_env, raising=False)
    monkeypatch.delenv(EXTRA_ENV, raising=False)


@pytest.fixture(scope="session", autouse=True)
def real_home_stays_clean():
    """Караул у настоящего каталога сессий.

    Ни один тест не должен оставить в нём следа. Проверка сделана сторожем, а
    не памяткой: забыть подменить домашний каталог легко, а заметить чужой
    файл в своём каталоге сессий — трудно.
    """
    def snapshot():
        if not REAL_SESSIONS.is_dir():
            return None
        return sorted(path.name for path in REAL_SESSIONS.rglob("*"))

    before = snapshot()
    yield
    after = snapshot()
    assert after == before, (
        "тесты изменили настоящий каталог сессий {}: было {}, стало {}".format(
            REAL_SESSIONS, before, after))


@pytest.fixture
def demo_config() -> AgentConfig:
    """Конфиг на заглушке без паузы: сто таких прогоняются за доли секунды."""
    return AgentConfig(name="test", provider="openai", model="gpt-5.4-mini",
                       transport=Transport(demo=True, demo_delay=0.0))


@pytest.fixture
def scripted():
    return ScriptedClient
