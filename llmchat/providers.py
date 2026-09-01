"""Описание поддерживаемых LLM-провайдеров и их моделей.

Все четыре провайдера принимают тело запроса в формате OpenAI, поэтому запросы
выполняет один SDK. Различия вынесены в описание провайдера:

* DeepSeek и OpenAI — ключ уходит в заголовок как есть;
* YandexGPT — модель задаётся URI ``gpt://<каталог>/<модель>/latest``,
  поэтому кроме ключа нужен идентификатор каталога;
* GigaChat — ключ авторизации сначала меняется на access-токен по OAuth,
  токен живёт 30 минут и обновляется автоматически.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Сколько токенов резервируем под ответ модели: на столько же выставляется
# max_tokens запроса, и на столько же уменьшается полезный объём контекста.
OUTPUT_RESERVE_CAP = 4096


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    context_window: int
    max_output: int

    @property
    def output_reserve(self) -> int:
        return min(self.max_output, OUTPUT_RESERVE_CAP)


@dataclass(frozen=True)
class ExtraField:
    """Дополнительный реквизит, который нужен провайдеру помимо ключа."""

    title: str
    hint: str
    help_text: str
    # Файлы, откуда значение можно взять, чтобы не вводить его каждый раз.
    files: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OAuthInfo:
    """Обмен долгоживущего ключа авторизации на короткий access-токен."""

    url: str
    scope: str
    # Токен живёт 30 минут; обновляем заранее, чтобы не словить 401 в середине запроса.
    refresh_margin_seconds: int = 120


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    name: str
    base_url: Optional[str]
    api_key_env: str
    token_url: str
    accent: str
    key_hint: str
    models: List[ModelInfo]
    # key_title — для заголовка панели, key_phrase — для середины предложения,
    # чтобы не приходилось менять регистр на лету и не получать «api-ключ».
    key_title: str = "API-ключ"
    key_phrase: str = "API-ключ"
    # Локальные файлы с ключом, в порядке предпочтения. Программа их только
    # читает: сама она ключи никуда не записывает.
    key_files: List[str] = field(default_factory=list)
    # Адрес для проверки ключа. Нужен там, где список моделей открыт без
    # авторизации и потому ключ не проверяет: у OpenRouter это именно так.
    validate_url: Optional[str] = None
    extra_field: Optional[ExtraField] = None
    model_uri_template: Optional[str] = None
    oauth: Optional[OAuthInfo] = None
    notes: List[str] = field(default_factory=list)

    @property
    def default_model(self) -> ModelInfo:
        return self.models[0]

    def model_ref(self, model: ModelInfo, extra: Optional[str]) -> str:
        """Строка, которая уйдёт в поле ``model`` запроса."""
        if self.model_uri_template and extra:
            return self.model_uri_template.format(extra=extra, model=model.id)
        return model.id


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        key="openai",
        name="ChatGPT (OpenAI)",
        base_url=None,  # дефолтный https://api.openai.com/v1
        api_key_env="OPENAI_API_KEY",
        token_url="https://platform.openai.com/api-keys",
        accent="green",
        key_hint="ключ начинается с sk-",
        key_files=["~/.openai-key", "~/.config/llm-chat/openai.key"],
        models=[
            ModelInfo("gpt-4o-mini", "GPT-4o mini — быстрый и дешёвый", 128_000, 16_384),
            ModelInfo("gpt-4o", "GPT-4o — самый сильный из классических", 128_000, 16_384),
            ModelInfo("gpt-4.1-mini", "GPT-4.1 mini — окно на 1 млн токенов", 1_047_576, 32_768),
        ],
    ),
    "deepseek": ProviderInfo(
        key="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        token_url="https://platform.deepseek.com/api_keys",
        accent="magenta",
        key_hint="ключ начинается с sk-",
        key_files=["~/.deepseek-key", "~/.llm-test-key", "~/.config/llm-chat/deepseek.key"],
        models=[
            ModelInfo("deepseek-chat", "DeepSeek Chat — универсальная модель", 64_000, 8_192),
            ModelInfo("deepseek-reasoner", "DeepSeek Reasoner — с цепочкой рассуждений", 64_000, 8_192),
        ],
    ),
    "yandex": ProviderInfo(
        key="yandex",
        name="YandexGPT",
        base_url="https://llm.api.cloud.yandex.net/v1",
        api_key_env="YANDEX_API_KEY",
        token_url="https://yandex.cloud/ru/docs/iam/operations/api-key/create",
        accent="red",
        key_hint="API-ключ сервисного аккаунта, начинается с AQVN",
        key_title="API-ключ",
        key_phrase="API-ключ",
        key_files=["~/.yandex-key", "~/.config/llm-chat/yandex.key"],
        extra_field=ExtraField(
            title="Идентификатор каталога (folder ID)",
            hint="выглядит как b1g…, виден в консоли Yandex Cloud",
            help_text="Модель у Яндекса задаётся адресом gpt://<каталог>/<модель>/latest, "
                      "поэтому кроме ключа нужен идентификатор каталога.",
            files=["~/.yandex-folder", "~/.config/llm-chat/yandex.folder"],
        ),
        model_uri_template="gpt://{extra}/{model}/latest",
        models=[
            ModelInfo("yandexgpt-lite", "YandexGPT Lite — быстрая и дешёвая", 32_000, 2_000),
            ModelInfo("yandexgpt", "YandexGPT Pro — сильнее и дороже", 32_000, 2_000),
        ],
        notes=["Ключ и каталог берутся в консоли Yandex Cloud: нужен сервисный аккаунт "
               "с ролью ai.languageModels.user."],
    ),
    "openrouter": ProviderInfo(
        key="openrouter",
        name="OpenRouter (бесплатные модели)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        token_url="https://openrouter.ai/keys",
        accent="bright_magenta",
        key_hint="ключ начинается с sk-or-v1-",
        key_files=["~/.openrouter-key", "~/.config/llm-chat/openrouter.key"],
        validate_url="https://openrouter.ai/api/v1/key",
        # Модели и размеры окон взяты из открытого каталога openrouter.ai/api/v1/models:
        # там указана цена, и у перечисленных ниже она равна нулю за ввод и за вывод.
        models=[
            ModelInfo("minimax/minimax-m3:free", "MiniMax M3 — окно на миллион токенов, отвечает быстро",
                      1_048_576, 943_718),
            ModelInfo("z-ai/glm-5.2:free", "GLM 5.2 от Z.ai — рассуждающая модель",
                      256_000, 230_400),
        ],
        notes=["Один ключ открывает модели разных разработчиков. Обе перечисленные "
               "бесплатны: в каталоге провайдера у них нулевая цена. Ключ выдаётся "
               "без привязки карты.",
               "Бесплатные модели делят общую очередь и временами отвечают отказом "
               "«перегружена». По проверке MiniMax отвечает стабильнее и быстрее, "
               "GLM — рассуждающая модель, ей нужен max_tokens побольше."],
    ),
    "gemini": ProviderInfo(
        key="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        token_url="https://aistudio.google.com/apikey",
        accent="blue",
        key_hint="ключ из Google AI Studio, начинается с AIza",
        key_files=["~/.gemini-key", "~/.config/llm-chat/gemini.key"],
        models=[
            ModelInfo("gemini-3.7-flash", "Gemini 3.7 Flash — самая свежая", 1_048_576, 65_536),
            ModelInfo("gemini-3.5-flash", "Gemini 3.5 Flash", 1_048_576, 65_536),
            ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash — проверенная временем",
                      1_048_576, 65_536),
        ],
        notes=["У Google AI Studio есть бесплатный тариф с ограничением по числу "
               "запросов в сутки; карта для получения ключа не нужна.",
               "Список моделей по OpenAI-совместимому пути не отдаётся, поэтому "
               "проверка ключа откладывается до первого запроса."],
    ),
    "gigachat": ProviderInfo(
        key="gigachat",
        name="GigaChat (Сбер)",
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key_env="GIGACHAT_CREDENTIALS",
        token_url="https://developers.sber.ru/studio/workspaces",
        accent="bright_green",
        key_hint="ключ авторизации из личного кабинета, длинная строка Base64",
        key_title="Ключ авторизации",
        key_phrase="ключ авторизации",
        key_files=["~/.gigachat-key", "~/.config/llm-chat/gigachat.key"],
        oauth=OAuthInfo(
            url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            scope="GIGACHAT_API_PERS",  # тариф для физических лиц
        ),
        models=[
            ModelInfo("GigaChat-2", "GigaChat 2 Lite — быстрая, входит в бесплатный лимит", 128_000, 4_096),
            ModelInfo("GigaChat-2-Pro", "GigaChat 2 Pro — для сложных задач", 128_000, 4_096),
            ModelInfo("GigaChat-2-Max", "GigaChat 2 Max — самая мощная", 128_000, 4_096),
        ],
        notes=["Ключ авторизации меняется на access-токен по OAuth; токен живёт 30 минут "
               "и обновляется программой автоматически.",
               "Сервер Сбера использует сертификат НУЦ Минцифры. Если увидите ошибку "
               "проверки сертификата, установите его командой: "
               "curl -k https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt "
               ">> $(python -m certifi)"],
    ),
}

PROVIDER_ORDER: List[str] = ["deepseek", "openai", "yandex", "gigachat",
                             "openrouter", "gemini"]
