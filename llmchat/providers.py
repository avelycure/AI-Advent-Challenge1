"""Описание поддерживаемых LLM-провайдеров и их моделей.

DeepSeek предоставляет OpenAI-совместимый API, поэтому оба провайдера
обслуживаются одним и тем же SDK — различаются только base_url и список моделей.
"""
from __future__ import annotations

from dataclasses import dataclass
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
class ProviderInfo:
    key: str
    name: str
    base_url: Optional[str]
    api_key_env: str
    token_url: str
    accent: str
    key_hint: str
    models: List[ModelInfo]

    @property
    def default_model(self) -> ModelInfo:
        return self.models[0]


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        key="openai",
        name="ChatGPT (OpenAI)",
        base_url=None,  # дефолтный https://api.openai.com/v1
        api_key_env="OPENAI_API_KEY",
        token_url="https://platform.openai.com/api-keys",
        accent="green",
        key_hint="ключ начинается с sk-",
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
        models=[
            ModelInfo("deepseek-chat", "DeepSeek Chat — универсальная модель", 64_000, 8_192),
            ModelInfo("deepseek-reasoner", "DeepSeek Reasoner — с цепочкой рассуждений", 64_000, 8_192),
        ],
    ),
}

PROVIDER_ORDER: List[str] = ["deepseek", "openai"]
