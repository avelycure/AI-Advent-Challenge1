"""Работа с LLM по HTTP.

Все провайдеры принимают тело запроса в формате OpenAI, поэтому запросы делает
один SDK ``openai``. Отличия провайдеров — база адреса, формат имени модели и
способ авторизации — спрятаны здесь же.
"""
from __future__ import annotations

import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from .providers import OAuthInfo, ProviderInfo
from .tokens import count_message_tokens, count_text_tokens

REQUEST_TIMEOUT = 120.0
DEFAULT_TOKEN_TTL = 1800.0  # 30 минут — столько живёт access-токен GigaChat


class LLMError(Exception):
    """Ошибка обращения к API, уже переведённая в понятный текст."""


class HttpProblem(Exception):
    """Неуспешный ответ, полученный не через SDK.

    Существует ради того, чтобы такой ответ проходил через тот же разбор
    ошибок, что и исключения SDK, и не заводить вторую копию правил.
    """


@dataclass
class ToolCall:
    """Просьба модели вызвать инструмент."""

    identifier: str
    name: str
    # Аргументы приходят строкой JSON: модель их сочиняет, и они могут быть
    # неразбираемы. Разбирать их — дело того, кто вызывает инструмент.
    arguments: str = "{}"


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # Почему генерация закончилась: "stop" — модель договорила сама,
    # "length" — упёрлась в max_tokens и ответ обрезан на полуслове.
    finish_reason: str = "stop"
    # Параметры, которые провайдер не принял и которые пришлось убрать.
    # Без этого списка отключение параметра выглядело бы как его применение.
    dropped_params: List[str] = field(default_factory=list)
    # Что программа изменила в запросе сама, чтобы он прошёл.
    notes: List[str] = field(default_factory=list)
    # Сколько секунд заняли отправка, ожидание и получение ответа. Повторы
    # после отказа провайдера входят сюда же: столько запрос и правда занял.
    elapsed: float = 0.0
    # Скрытое размышление рассуждающих моделей. Уже входит в completion_tokens
    # и тарифицируется как выход, но без отдельной строки непонятно, за что
    # заплачено: ответ в пять символов может стоить как страница текста.
    reasoning_tokens: int = 0
    # Часть входа, зачтённая провайдером по кешу и потому более дешёвая.
    cached_tokens: int = 0
    # Стоимость запроса. Заполняет счётчик агента: цена живёт в каталоге
    # моделей, а не в транспорте, и до записи в счётчик она неизвестна.
    cost: Optional[float] = None
    # Инструменты, которые модель просит вызвать. Пока список не пуст,
    # ответа как такового ещё нет.
    tool_calls: List[ToolCall] = field(default_factory=list)


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
    if "tool calling" in lowered and "not supported" in lowered:
        # Инструменты умеют не все модели, и сырой отказ провайдера этого не
        # объясняет. Молча убрать их нельзя: агент перестал бы делать то, что
        # записано в его конфиге, а человек не узнал бы почему.
        return ("Эта модель не умеет вызывать инструменты. Выберите другую модель "
                "или уберите их из конфига: --set tools.enabled=[]")
    if "incorrect api key" in lowered:
        # xAI отвечает на неверный ключ кодом 400, а не 401, как остальные.
        return "Ключ отклонён провайдером. Проверьте его в консоли console.x.ai."
    if "credits or licenses" in lowered or "no credits" in lowered:
        return ("У аккаунта нет кредитов: xAI не даёт бесплатного доступа к моделям. "
                "Пополните баланс в консоли console.x.ai или выберите другого провайдера.")
    if ("api key not valid" in lowered or "pass a valid api key" in lowered
            or "invalid authorization header" in lowered):
        # Google отвечает на неверный ключ кодом 400, а не 401, как большинство.
        return "Ключ отклонён провайдером. Проверьте его в Google AI Studio."
    if name == "AuthenticationError" or "401" in text or "invalid_api_key" in text:
        return "Ключ отклонён провайдером (401). Проверьте, что он актуален и скопирован целиком."
    if name == "PermissionDeniedError" or "403" in text:
        return "Доступ запрещён (403): у ключа нет прав на эту модель или заблокирован регион."
    if name == "NotFoundError" or "404" in text:
        return "Модель не найдена (404): проверьте, доступна ли она вашему аккаунту."
    if "rate-limited upstream" in lowered:
        # Бесплатные модели OpenRouter делят общую очередь: отказ означает, что
        # сейчас занята сама модель, а не что исчерпан ваш лимит или баланс.
        return ("Бесплатная модель сейчас перегружена на стороне провайдера. "
                "Это не про ваш ключ и не про баланс — попробуйте позже "
                "или выберите другую модель.")
    if name == "RateLimitError" or "429" in text:
        return "Слишком много запросов или закончился баланс (429). Попробуйте позже."
    if "402" in text or "insufficient" in text.lower() or "quota" in text.lower():
        return ("Недостаточно средств на аккаунте провайдера (402). Ключ рабочий, "
                "но баланс нулевой — пополните его или запустите ./run.sh --demo.")
    if name in {"APIConnectionError", "APITimeoutError"} or "Connection" in name:
        return "Не удалось связаться с API: проверьте сеть, прокси или VPN."
    # Раньше общей ветки про 400: переполнение контекста приходит именно
    # с этим кодом, и, стоя ниже, объяснение не срабатывало никогда — человек
    # вместо совета получал сырой JSON провайдера.
    if _about_context_length(lowered):
        # Слова провайдера идут следом за советом, а не вместо него: только он
        # знает точный размер запроса по своему токенизатору, и эти числа
        # человеку нужнее нашей оценки — по ним видно, насколько сокращать.
        return _with_provider_words(
            "Контекст переполнен: диалог не помещается в окно модели. "
            "Начните новый диалог командой /new или задайте вопрос короче.", text)
    if name == "BadRequestError" or "400" in text:
        return "Провайдер отклонил запрос (400): {}".format(_compact(text))
    return "{}: {}".format(name, _compact(text) or "неизвестная ошибка")


# Как разные провайдеры называют одно и то же переполнение. Проверено живьём
# на OpenRouter: «This endpoint's maximum context length is 65536 tokens.
# However, you requested about 141100 tokens».
CONTEXT_LENGTH_PHRASES = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "too many tokens",
    "reduce the length of the messages",
)


def _about_context_length(lowered: str) -> bool:
    return any(phrase in lowered for phrase in CONTEXT_LENGTH_PHRASES)


def _with_provider_words(advice: str, text: str) -> str:
    words = _provider_message(text)
    return "{}\nПровайдер: {}".format(advice, words) if words else advice


# Провайдеры заворачивают человеческое объяснение в поле message своего тела
# ответа, а SDK показывает это тело как есть: «Error code: 400 - {'error':
# {'message': "…"}}». Читать такое человеку незачем — нужна только строка.
PROVIDER_MESSAGE = re.compile(r"""["']message["']\s*:\s*(["'])(.+?)\1""", re.DOTALL)


def _provider_message(text: str) -> str:
    """Вынуть объяснение провайдера из тела ответа. Пусто — вынимать нечего."""
    found = PROVIDER_MESSAGE.search(text)
    if found is None:
        return ""
    words = found.group(2).replace("\\n", " ").replace('\\"', '"').replace("\\'", "'")
    return _compact(words, limit=300)


# Сколько раз запрос можно переписать под отказ провайдера, прежде чем сдаться.
ADAPT_ATTEMPTS = 4

# Отказ вида «в запросе должно встречаться слово json».
JSON_WORD_REQUIRED = re.compile(r"must contain the word .?json", re.IGNORECASE)
JSON_INSTRUCTION = "Ответ верни строго в формате JSON и ничем больше."
JSON_HINT_NOTE = ("провайдер требует, чтобы формат был назван и в самом запросе — "
                  "в него добавлено требование ответить в формате JSON")


def _adapt_request(exc: Exception, kwargs: Dict[str, object], reported: List[str],
                   notes: List[str], max_tokens: int) -> bool:
    """Подстроить запрос под отказ провайдера. False — подстраивать нечего.

    Провайдер называет в ошибке одно поле за раз, поэтому подстройка вызывается
    в цикле: например, groq/compound отвергает сначала reasoning_format,
    а на следующем запросе — reasoning_effort.
    """
    text = str(exc)
    if "max_tokens" in kwargs and "max_tokens" in text and "max_completion_tokens" in text:
        # Новые модели OpenAI принимают max_completion_tokens вместо max_tokens.
        kwargs.pop("max_tokens")
        kwargs["max_completion_tokens"] = max_tokens
        return True

    if JSON_WORD_REQUIRED.search(text) and _require_json_in_prompt(kwargs):
        notes.append(JSON_HINT_NOTE)
        return True

    changed, names = _drop_unsupported(exc, kwargs)
    reported.extend(names)
    return changed


def _is_json_mode(response_format: Optional[Dict[str, str]]) -> bool:
    return bool(response_format) and response_format.get("type") == "json_object"


def _require_json_in_prompt(kwargs: Dict[str, object]) -> bool:
    """Дописать в запрос требование JSON. False — оно там уже есть.

    OpenAI, DeepSeek и Groq принимают response_format=json_object только если
    слово «json» встречается в самом запросе: иначе модель не знает, чего от неё
    хотят, и провайдер отказывается угадывать за неё.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return False
    if any("json" in str(item.get("content", "")).lower() for item in messages):
        return False
    kwargs["messages"] = [{"role": "system", "content": JSON_INSTRUCTION}] + list(messages)
    return True


def _drop_unsupported(exc: Exception, kwargs: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Убрать из запроса параметры, которые провайдер не принимает.

    Возвращает признак того, что запрос изменился и его стоит повторить, и
    список имён для показа пользователю. Списки разные: параметры генерации
    пользователь задавал сам и должен узнать, что они не применились, а
    служебные поля провайдера он не выбирал, и сообщать о них незачем.
    """
    text = error_chain_text(exc).lower()
    if not any(marker in text for marker in ("400", "unsupported", "unknown", "invalid")):
        return False, []

    reported: List[str] = []
    for name in ("response_format", "stop", "top_p", "temperature", "tools",
                 "tool_choice"):
        if name in kwargs and _mentioned(name, text):
            kwargs.pop(name)
            reported.append(name)

    changed = bool(reported)
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict):
        for name in [key for key in extra if _mentioned(key, text)]:
            extra.pop(name)
            changed = True
        if not extra:
            kwargs.pop("extra_body")
    return changed, reported


def _tool_calls_of(message: object) -> List[ToolCall]:
    """Вытащить просьбы вызвать инструмент из ответа модели.

    SDK отдаёт их объектами, часть прокси — словарями, поэтому читаем оба вида:
    иначе на прокси инструменты молча перестали бы работать.
    """
    raw = getattr(message, "tool_calls", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("tool_calls")
    calls: List[ToolCall] = []
    for item in raw or []:
        function = item.get("function") if isinstance(item, dict) else getattr(
            item, "function", None)
        if function is None:
            continue
        name = (function.get("name") if isinstance(function, dict)
                else getattr(function, "name", "")) or ""
        arguments = (function.get("arguments") if isinstance(function, dict)
                     else getattr(function, "arguments", "")) or "{}"
        identifier = (item.get("id") if isinstance(item, dict)
                      else getattr(item, "id", "")) or name
        if name:
            calls.append(ToolCall(identifier, name, arguments))
    return calls


def _usage_detail(usage: object, group: str, name: str) -> int:
    """Поле из вложенной детализации usage.

    SDK отдаёт её объектом, часть прокси — словарём, поэтому читаем обоими
    способами: иначе на прокси счётчик молча становится нулём.
    """
    holder = getattr(usage, group, None)
    if holder is None and isinstance(usage, dict):
        holder = usage.get(group)
    if holder is None:
        return 0
    value = holder.get(name) if isinstance(holder, dict) else getattr(holder, name, 0)
    return int(value or 0)


def _mentioned(name: str, error_text: str) -> bool:
    """Назван ли параметр в тексте ошибки: подчёркивания в нём непостоянны."""
    return name.replace("_", "") in error_text.replace("_", "")


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
        # Провайдер сообщает об этом только отказом на первый запрос, поэтому
        # запоминаем: иначе каждое следующее сообщение стоило бы лишнего отказа.
        self._json_word_required = False
        self._client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=2,
        )

    def _prepare(self) -> None:
        """Точка расширения: у GigaChat здесь обновляется access-токен."""

    def validate_key(self) -> None:
        """Дешёвая проверка ключа до начала диалога."""
        self._prepare()

        if self.provider.validate_url:
            # У части провайдеров список моделей открыт без авторизации и потому
            # ключ не проверяет — для них задан отдельный адрес проверки.
            try:
                response = httpx.get(
                    self.provider.validate_url,
                    headers={"Authorization": "Bearer " + str(self._client.api_key)},
                    timeout=REQUEST_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                raise LLMError(describe_error(exc)) from exc
            if response.status_code >= 400:
                problem = HttpProblem("Error code: {} - {}".format(
                    response.status_code, _compact(response.text, 200)))
                raise LLMError(describe_error(problem))
            return

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
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Completion:
        self._prepare()
        kwargs = {
            "model": model_ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop:
            kwargs["stop"] = stop
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        # SDK не пропускает незнакомые ему поля как обычные аргументы,
        # поэтому специфичные для провайдера кладём в тело запроса напрямую.
        extra_body = self.provider.extra_body_for(model_ref)
        if extra_body:
            kwargs["extra_body"] = extra_body

        dropped: List[str] = []
        notes: List[str] = []
        if self._json_word_required and _is_json_mode(response_format):
            _require_json_in_prompt(kwargs)

        started = time.perf_counter()
        for attempt in range(ADAPT_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last = attempt == ADAPT_ATTEMPTS - 1
                if last or not _adapt_request(exc, kwargs, dropped, notes, max_tokens):
                    raise LLMError(describe_error(exc)) from exc
                if notes and JSON_HINT_NOTE in notes:
                    self._json_word_required = True

        if not response.choices:
            raise LLMError("Модель вернула пустой ответ без вариантов.")

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        tool_calls = _tool_calls_of(choice.message)
        # Пустой текст — не ошибка, когда модель просит вызвать инструмент:
        # ответа ещё нет, и он появится после того, как инструмент отработает.
        if not text and not tool_calls:
            raise LLMError("Модель вернула пустой текст ответа.")
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        elapsed = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        reasoning_tokens = _usage_detail(usage, "completion_tokens_details", "reasoning_tokens")
        cached_tokens = _usage_detail(usage, "prompt_tokens_details", "cached_tokens")
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            # GigaChat кэширует префикс промпта и НЕ включает его в prompt_tokens:
            # без этого слагаемого размер диалога занижается и даже уменьшается
            # по мере роста истории. У OpenAI и DeepSeek семантика обратная —
            # там кэшированные токены уже входят в prompt_tokens, поэтому
            # складываем только специфичное поле Сбера.
            precached = int(getattr(usage, "precached_prompt_tokens", 0) or 0)
            prompt_tokens += precached
        else:
            # Некоторые прокси не возвращают usage — оцениваем сами.
            prompt_tokens = count_message_tokens(messages)
            completion_tokens = count_text_tokens(text)
        return Completion(text, prompt_tokens, completion_tokens, finish_reason, dropped,
                          notes, elapsed, reasoning_tokens, cached_tokens,
                          tool_calls=tool_calls)


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
            # Сбер отвечает 400 и на испорченный ключ, и на нераспознанный —
            # различить их по ответу нельзя, поэтому называем обе причины.
            return ("Ключ авторизации не принят (400). Либо скопирована не вся строка "
                    "Base64 из личного кабинета, либо ключ недействителен или выдан "
                    "под другой scope. Ответ сервера: {}".format(body))
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

    def __init__(self, provider: ProviderInfo, api_key: str,
                 delay: Optional[float] = None) -> None:
        self.provider = provider
        self._counter = 0
        # None — обычная «живая» пауза, чтобы в интерфейсе было видно ожидание.
        # 0.0 ставят тесты и массовый прогон: сотня агентов не должна ждать зря.
        self._delay = delay

    def _pause(self, low: float, high: float) -> None:
        if self._delay is None:
            time.sleep(random.uniform(low, high))
        elif self._delay > 0:
            time.sleep(self._delay)

    def validate_key(self) -> None:
        self._pause(0.6, 0.6)

    def complete(
        self,
        model_ref: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Completion:
        started = time.perf_counter()
        self._pause(1.2, 2.2)
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
            elapsed=time.perf_counter() - started,
        )


def make_client(provider: ProviderInfo, secret: str, demo: bool,
                demo_delay: Optional[float] = None):
    """Собрать клиент, подходящий выбранному провайдеру."""
    if demo:
        return DemoClient(provider, secret, demo_delay)
    if provider.oauth is not None:
        return GigaChatClient(provider, secret)
    return LLMClient(provider, secret)
