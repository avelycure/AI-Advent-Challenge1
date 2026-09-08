"""Поиск модели по имени.

Имя модели уникально во всём каталоге, поэтому провайдера можно не называть.
Проверка на неоднозначность всё равно нужна: каталог растёт, и в день, когда
одно имя появится у двух провайдеров, программа обязана спросить, а не выбрать
первого попавшегося.
"""
from __future__ import annotations

import difflib
from typing import List, Tuple

from .errors import ConfigError
from .transport import PROVIDER_ORDER, PROVIDERS


def model_ids() -> List[str]:
    return [model.id for key in PROVIDER_ORDER for model in PROVIDERS[key].models]


def find_model(spec: str) -> Tuple[str, str]:
    """Вернуть пару «провайдер, модель» по записи вида ``gpt-5.4`` или
    ``openai/gpt-5.4``.

    Сначала пробуем имя целиком, и только потом читаем часть до косой черты
    как провайдера. Порядок именно такой, потому что у Groq косая черта входит
    в само имя модели: ``qwen/qwen3.6-27b`` и ``groq/compound`` — это имена,
    а не запись «провайдер/модель».
    """
    wanted = spec.strip()
    if not wanted:
        raise ConfigError("не названа модель")

    exact = [(key, model.id) for key in PROVIDER_ORDER
             for model in PROVIDERS[key].models if model.id == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ConfigError("модель «{}» есть у нескольких провайдеров ({}) — "
                          "уточните записью провайдер/модель".format(
                              wanted, ", ".join(key for key, _ in exact)))

    if "/" in wanted:
        provider_key, _, model_id = wanted.partition("/")
        provider = PROVIDERS.get(provider_key.strip())
        if provider is None:
            raise ConfigError("неизвестный провайдер «{}»; доступны: {}".format(
                provider_key, ", ".join(PROVIDER_ORDER)))
        for model in provider.models:
            if model.id == model_id.strip():
                return provider.key, model.id
        raise ConfigError("у провайдера {} нет модели «{}»; доступны: {}".format(
            provider.name, model_id, ", ".join(m.id for m in provider.models)))

    close = difflib.get_close_matches(wanted, model_ids(), n=3, cutoff=0.4)
    raise ConfigError("нет модели «{}»{}".format(
        wanted, ". Похожие: " + ", ".join(close) if close else
        "; полный список моделей показывает /change_model"))
