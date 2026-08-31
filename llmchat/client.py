"""Работа с LLM по HTTP.

Оба провайдера говорят на OpenAI-совместимом протоколе, поэтому используется
один SDK ``openai`` — DeepSeek подключается через собственный base_url.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List

from .providers import ProviderInfo
from .tokens import count_message_tokens, count_text_tokens

REQUEST_TIMEOUT = 120.0


class LLMError(Exception):
    """Ошибка обращения к API, уже переведённая в понятный текст."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    def __init__(self, provider: ProviderInfo, api_key: str) -> None:
        from openai import OpenAI

        self.provider = provider
        self._client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=2,
        )

    def validate_key(self) -> None:
        """Дешёвая проверка ключа до начала диалога: список моделей."""
        try:
            self._client.models.list()
        except Exception as exc:  # noqa: BLE001 — переводим в свой тип
            raise LLMError(describe_error(exc)) from exc

    def complete(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> Completion:
        kwargs = {
            "model": model_id,
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
        model_id: str,
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


def _compact(text: str, limit: int = 200) -> str:
    """Схлопнуть переносы и пробелы: в панель ошибки должна лечь одна строка."""
    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def describe_error(exc: Exception) -> str:
    """Перевести исключение SDK в короткое человеческое объяснение."""
    name = type(exc).__name__
    text = str(exc)

    # Корпоративный прокси часто отвечает HTML-страницей вместо JSON провайдера.
    lowered = text.lower()
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
