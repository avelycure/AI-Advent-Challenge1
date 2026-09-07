"""Транспорт: HTTP-обращение к провайдерам, их каталог, реквизиты и токены.

Слой ничего не знает ни об агенте, ни о интерфейсе: он умеет отправить готовые
сообщения выбранной модели и вернуть ответ с измерениями.
"""
from .client import Completion, LLMClient, LLMError, make_client
from .providers import (
    PROVIDER_ORDER,
    PROVIDERS,
    ModelInfo,
    ProviderInfo,
    request_cost,
)
from .secrets import SecretSource, find_sources, mask, shorten
from .tokens import (
    CONVERSATION_OVERHEAD,
    MESSAGE_OVERHEAD,
    count_message_tokens,
    count_text_tokens,
    tokenizer_name,
)

__all__ = [
    "Completion", "LLMClient", "LLMError", "make_client",
    "PROVIDERS", "PROVIDER_ORDER", "ModelInfo", "ProviderInfo", "request_cost",
    "SecretSource", "find_sources", "mask", "shorten",
    "count_message_tokens", "count_text_tokens", "tokenizer_name",
    "MESSAGE_OVERHEAD", "CONVERSATION_OVERHEAD",
]
