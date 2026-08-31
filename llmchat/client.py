"""Работа с LLM по HTTP.

Все провайдеры принимают тело запроса в формате OpenAI, поэтому запросы делает
один SDK ``openai``. Отличия провайдеров — база адреса, формат имени модели и
способ авторизации — спрятаны здесь же.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from .providers import OAuthInfo, ProviderInfo
from .tokens import count_message_tokens, count_text_tokens

REQUEST_TIMEOUT = 120.0
DEFAULT_TOKEN_TTL = 1800.0  # 30 минут — столько живёт access-токен GigaChat


class LLMError(Exception):
    """Ошибка обращения к API, уже переведённая в понятный текст."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


# --------------------------------------------------------------------------
# Разбор ошибок
# --------------------------------------------------------------------------

def error_chain_text(exc: BaseException) -> str:
    """Собрать текст исключения вместе с его причинами.

    SDK прячет настоящую причину (например, отказ проверки TLS-сертификата)
    в ``__cause__``, а сам показывает лаконичное «Connection error».
    """
    parts: List[str] = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append("{}: {}".format(type(current).__name__, current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _compact(text: str, limit: int = 200) -> str:
    """Схлопнуть переносы и пробелы: в панель ошибки должна лечь одна строка."""
    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def describe_error(exc: Exception) -> str:
    """Перевести исключение SDK в короткое человеческое объяснение."""
    name = type(exc).__name__
    text = str(exc)
    chain = error_chain_text(exc)
    lowered = chain.lower()

    if isinstance(exc, UnicodeEncodeError) or "UnicodeEncodeError" in chain:
        return ("В ключе есть символы вне латиницы — HTTP-заголовок такое не принимает. "
                "Скорее всего при копировании попала кириллическая буква "
                "(например, «с» вместо латинской «c») или лишний символ.")
    if "certificate_verify_failed" in lowered or "certificate verify failed" in lowered:
        return ("Не пройдена проверка TLS-сертификата. У GigaChat сертификат НУЦ Минцифры — "
                "установите его: curl -k "
                "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt "
                ">> $(python -m certifi)")
    # Корпоративный прокси часто отвечает HTML-страницей вместо JSON провайдера.
    if "<html" in lowered or "<!doctype html" in lowered:
        return ("Вместо ответа API пришла HTML-страница — запрос перехватил прокси "
                "или фильтр сети. Добавьте адрес провайдера в NO_PROXY либо "
                "отключите прокси для него.")
    if name == "AuthenticationError" or "401" in text or "invalid_api_key" in text:
        return "Ключ отклонён провайдером (401). Проверьте, что он актуален и скопирован целиком."
    if name == "PermissionDeniedError" or "403" in text:
        return "Доступ запрещён (403): у ключа нет прав на эту модель или заблокирован регион."
    if name == "NotFoundError" or "404" in text:
        return "Модель не найдена (404): проверьте, доступна ли она вашему аккаунту."
    if name == "RateLimitError" or "429" in text:
        return "Слишком много запросов или закончился баланс (429). Попробуйте позже."
    if "402" in text or "insufficient" in text.lower() or "quota" in text.lower():
        return ("Недостаточно средств на аккаунте провайдера (402). Ключ рабочий, "
                "но баланс нулевой — пополните его или запустите ./run.sh --demo.")
    if name in {"APIConnectionError", "APITimeoutError"} or "Connection" in name:
        return "Не удалось связаться с API: проверьте сеть, прокси или VPN."
    if name == "BadRequestError" or "400" in text:
        return "Провайдер отклонил запрос (400): {}".format(_compact(text))
    if "context" in text.lower() and "length" in text.lower():
        return "Контекст переполнен — начните новый диалог командой /new."
    return "{}: {}".format(name, _compact(text) or "неизвестная ошибка")


def _endpoint_unsupported(exc: Exception) -> bool:
    """Отличить «у провайдера нет списка моделей» от настоящей проблемы.

    Пропускать пользователя дальше можно только в первом случае. Раньше здесь был
    обратный список — «что считать блокирующим», — и любая неожиданная ошибка молча
    пропускалась, из-за чего негодный ключ доходил до диалога.
    """
    name = type(exc).__name__
    if name in {"NotFoundError", "UnprocessableEntityError"}:
        return True
    chain = error_chain_text(exc)
    return any(marker in chain for marker in ("404", "405", "Not Found", "Method Not Allowed"))


# --------------------------------------------------------------------------
# Клиенты
# --------------------------------------------------------------------------

class LLMClient:
    """Провайдеры, принимающие ключ в заголовке как есть: OpenAI, DeepSeek, YandexGPT."""

    def __init__(self, provider: ProviderInfo, api_key: str) -> None:
        from openai import OpenAI

        self.provider = provider
        self._client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=2,
        )

    def _prepare(self) -> None:
        """Точка расширения: у GigaChat здесь обновляется access-токен."""

    def validate_key(self) -> None:
        """Дешёвая проверка ключа до начала диалога: список моделей."""
        self._prepare()
        try:
            self._client.models.list()
        except Exception as exc:  # noqa: BLE001 — переводим в свой тип
            if not _endpoint_unsupported(exc):
                raise LLMError(describe_error(exc)) from exc
            # Эндпоинта /models у провайдера нет — проверку отложим до первого запроса.

    def complete(
        self,
        model_ref: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> Completion:
        self._prepare()
        kwargs = {
            "model": model_ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Новые модели OpenAI принимают max_completion_tokens вместо max_tokens.
            if "max_tokens" in str(exc) and "max_completion_tokens" in str(exc):
                kwargs.pop("max_tokens")
                kwargs["max_completion_tokens"] = max_tokens
                try:
                    response = self._client.chat.completions.create(**kwargs)
                except Exception as retry_exc:  # noqa: BLE001
                    raise LLMError(describe_error(retry_exc)) from retry_exc
            else:
                raise LLMError(describe_error(exc)) from exc

        if not response.choices:
            raise LLMError("Модель вернула пустой ответ без вариантов.")

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("Модель вернула пустой текст ответа.")

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            # Некоторые прокси не возвращают usage — оцениваем сами.
            prompt_tokens = count_message_tokens(messages)
            completion_tokens = count_text_tokens(text)
        return Completion(text, prompt_tokens, completion_tokens)


class GigaChatAuth:
    """Обмен ключа авторизации на access-токен с кэшированием.

    Токен живёт 30 минут, а частота обращений к серверу авторизации ограничена,
    поэтому запрашивать его перед каждым сообщением нельзя.
    """

    def __init__(self, oauth: OAuthInfo, authorization_key: str) -> None:
        self._oauth = oauth
        # Из личного кабинета ключ иногда копируют вместе со словом Basic.
        self._key = authorization_key.strip()
        if self._key.lower().startswith("basic "):
            self._key = self._key[6:].strip()
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def access_token(self) -> str:
        if self._token and time.time() < self._expires_at - self._oauth.refresh_margin_seconds:
            return self._token
        return self._fetch()

    def _fetch(self) -> str:
        headers = {
            "Authorization": "Basic " + self._key,
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            response = httpx.post(
                self._oauth.url,
                headers=headers,
                data={"scope": self._oauth.scope},
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(describe_error(exc)) from exc

        if response.status_code != 200:
            raise LLMError(self._describe_failure(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError("Сервер авторизации вернул не JSON.") from exc

        token = payload.get("access_token")
        if not token:
            raise LLMError("Сервер авторизации не вернул access_token.")

        # expires_at приходит в миллисекундах Unix-времени, а не в секундах.
        expires_at = payload.get("expires_at")
        try:
            self._expires_at = float(expires_at) / 1000.0
        except (TypeError, ValueError):
            self._expires_at = time.time() + DEFAULT_TOKEN_TTL
        self._token = token
        return token

    @staticmethod
    def _describe_failure(response: "httpx.Response") -> str:
        body = _compact(response.text, 160)
        if response.status_code == 401:
            return "Ключ авторизации отклонён (401): проверьте его в личном кабинете."
        if response.status_code == 400:
            return ("Сервер не смог разобрать ключ авторизации (400). Нужна вся строка "
                    "Base64 из личного кабинета целиком. Ответ сервера: {}".format(body))
        if response.status_code == 429:
            return "Слишком частые обращения к серверу авторизации (429), попробуйте позже."
        return "Авторизация не удалась ({}): {}".format(response.status_code, body)


class GigaChatClient(LLMClient):
    """GigaChat: перед каждым запросом подставляем свежий access-токен."""

    def __init__(self, provider: ProviderInfo, authorization_key: str) -> None:
        super().__init__(provider, api_key="ожидается-oauth")
        assert provider.oauth is not None
        self._auth = GigaChatAuth(provider.oauth, authorization_key)

    def _prepare(self) -> None:
        self._client.api_key = self._auth.access_token()


DEMO_REPLIES = [
    "Это демонстрационный режим: настоящий запрос в API не отправлялся.\n\n"
    "Так выглядит обычный ответ модели — с **разметкой**, списком:\n\n"
    "1. история диалога сохраняется целиком;\n"
    "2. счётчики токенов внизу обновляются после каждого ответа;\n"
    "3. тема диалога в шапке подставляется автоматически.",
    "Продолжаю демонстрацию. Проверить можно так:\n\n"
    "```bash\n./run.sh --demo\n```\n\n"
    "Каждый следующий ответ увеличивает контекст, поэтому полоса заполнения растёт, "
    "а оценка «хватит ещё на N сообщений» уменьшается.",
    "Ещё один ответ демо-режима. Реальные цифры расхода приходят от API в поле "
    "`usage`, здесь они смоделированы по длине текста.",
]


class DemoClient:
    """Заглушка для проверки интерфейса и логики без обращения к сети."""

    def __init__(self, provider: ProviderInfo, api_key: str) -> None:
        self.provider = provider
        self._counter = 0

    def validate_key(self) -> None:
        time.sleep(0.6)

    def complete(
        self,
        model_ref: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> Completion:
        time.sleep(random.uniform(1.2, 2.2))
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if any("три-пять слов" in m.get("content", "") for m in messages):
            text = "Демонстрация работы клиента"
        else:
            text = DEMO_REPLIES[self._counter % len(DEMO_REPLIES)]
            text = "Вы спросили: «{}».\n\n{}".format(last_user[:120], text)
            self._counter += 1
        return Completion(
            text=text,
            prompt_tokens=count_message_tokens(messages),
            completion_tokens=count_text_tokens(text),
        )


def make_client(provider: ProviderInfo, secret: str, demo: bool):
    """Собрать клиент, подходящий выбранному провайдеру."""
    if demo:
        return DemoClient(provider, secret)
    if provider.oauth is not None:
        return GigaChatClient(provider, secret)
    return LLMClient(provider, secret)
