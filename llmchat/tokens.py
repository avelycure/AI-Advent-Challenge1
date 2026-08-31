"""Подсчёт токенов.

Точные цифры расхода приходят от API в поле ``usage`` каждого ответа — они и
используются как источник истины. Этот модуль нужен только для оценки ещё не
отправленных сообщений: без него нельзя было бы показать заполнение контекста
до первого запроса и сразу после ввода пользователя.
"""
from __future__ import annotations

from typing import Dict, List

# Служебная разметка ролей, которую провайдер добавляет к каждому сообщению.
MESSAGE_OVERHEAD = 4
CONVERSATION_OVERHEAD = 3

_encoder = None
_encoder_resolved = False


def _get_encoder():
    """Ленивая загрузка tiktoken; при любой проблеме молча уходим в эвристику."""
    global _encoder, _encoder_resolved
    if _encoder_resolved:
        return _encoder
    _encoder_resolved = True
    try:
        import warnings

        # На старом системном Python tiktoken тянет urllib3, который шумит
        # предупреждением про LibreSSL — в красивом TUI это лишнее.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = None
    return _encoder


def _heuristic(text: str) -> int:
    """Грубая оценка: латиница ~4 символа на токен, кириллица ~2."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return int(ascii_chars / 4 + other_chars / 2) + 1


def count_text_tokens(text: str) -> int:
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return _heuristic(text)


def count_message_tokens(messages: List[Dict[str, str]]) -> int:
    total = CONVERSATION_OVERHEAD
    for message in messages:
        total += count_text_tokens(message.get("content", "")) + MESSAGE_OVERHEAD
    return total


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _get_encoder() is not None else "эвристика"
