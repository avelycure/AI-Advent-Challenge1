"""Реквизиты и клиенты — без единого вопроса пользователю.

Ровно это место делало невозможным запуск агента из кода: ключ спрашивался
через ``Console``. Здесь он ищется молча — в конфиге, в переменной окружения,
в файле рядом, — а если не нашёлся, поднимается ошибка. Спрашивать человека
может интерфейс, и его дело — положить найденное в ``AgentConfig.api_key``.

Клиенты кэшируются: сто агентов на одном ключе не должны открывать сто
соединений. Кэш общий на процесс и берётся из разных потоков, поэтому за ним
стоит замок.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import AgentConfig
from .errors import MissingCredentials
from .transport import ProviderInfo, find_sources, make_client


@dataclass(frozen=True)
class Credentials:
    key: str
    extra: Optional[str] = None
    # Откуда взято — для интерфейса и отчётов; в кэше не участвует.
    source: str = ""


def resolve_credentials(config: AgentConfig) -> Credentials:
    """Найти ключ и второй реквизит для конфига."""
    provider = config.provider_info

    if config.transport.demo:
        # Заглушке реквизиты не нужны, но пусть они будут заданными:
        # иначе кэш клиентов пришлось бы учить особому случаю.
        return Credentials("demo", config.api_extra or "demo", "демонстрационный режим")

    key, source = config.api_key, "конфиг"
    if not key:
        sources = find_sources(provider.api_key_env, provider.key_files)
        if not sources:
            raise MissingCredentials(
                "не найден {} для {}. Задайте config.api_key, переменную {} "
                "или положите ключ в {}".format(
                    provider.key_phrase, provider.name, provider.api_key_env,
                    " либо ".join(provider.key_files) or "файл рядом"))
        key, source = sources[0].value, sources[0].label

    extra = config.api_extra
    if provider.extra_field is not None and not extra:
        found = find_sources(None, provider.extra_field.files)
        if not found:
            raise MissingCredentials(
                "не найден реквизит «{}» для {}. Задайте config.api_extra "
                "или положите его в {}".format(
                    provider.extra_field.title, provider.name,
                    " либо ".join(provider.extra_field.files) or "файл рядом"))
        extra = found[0].value

    return Credentials(key, extra, source)


class ClientRegistry:
    """Кэш HTTP-клиентов по паре «провайдер + реквизиты»."""

    def __init__(self) -> None:
        self._clients: Dict[Tuple, object] = {}
        self._lock = threading.Lock()

    def client_for(self, config: AgentConfig, credentials: Credentials):
        provider = config.provider_info
        token = (provider.key, credentials.key, credentials.extra,
                 config.transport.demo, config.transport.demo_delay)
        with self._lock:
            client = self._clients.get(token)
            if client is None:
                client = make_client(provider, credentials.key, config.transport.demo,
                                     config.transport.demo_delay)
                self._clients[token] = client
            return client

    def __len__(self) -> int:
        return len(self._clients)

    def clear(self) -> None:
        with self._lock:
            self._clients.clear()


# Общий реестр процесса: его хватает и чату, и массовому прогону.
SHARED = ClientRegistry()


def model_ref(provider: ProviderInfo, config: AgentConfig, extra: Optional[str]) -> str:
    """Строка, которая уйдёт в поле ``model`` запроса."""
    return provider.model_ref(config.model_info, extra)
