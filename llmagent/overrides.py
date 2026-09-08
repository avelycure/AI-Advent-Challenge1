"""Правки конфига, заданные при запуске.

Конфиг из файла или из умолчаний — это основа, а правка накрывает её сверху:
названное поле меняется, остальные остаются как были. Правка задаётся полным
путём к полю, например ``generation.temperature`` или ``judge.agent.model``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import AgentConfig, default_section, field_type
from .errors import ConfigError


# Короткие имена частых полей. Ровно те же, что у именованных флагов: человек,
# набравший «--temperature», не должен вспоминать «generation.temperature»,
# когда пишет то же самое через --set или в правке под-агенту.
ALIASES: Dict[str, str] = {
    "temperature": "generation.temperature",
    "max_tokens": "generation.max_tokens",
    "top_p": "generation.top_p",
    "stop": "generation.stop",
    "format": "output.format",
    "attempts": "output.max_attempts",
    "max_cost": "budget.max_cost",
    "max_requests": "budget.max_requests",
    "max_chars": "input.max_chars",
    "template": "input.template",
    "demo": "transport.demo",
}


def expand(path: str) -> str:
    """Развернуть короткое имя в полный путь. Полный путь остаётся как есть."""
    return ALIASES.get(path, path)


def parse_assignment(text: str) -> Tuple[str, str]:
    """Разобрать запись «путь=значение»."""
    path, sign, raw = text.partition("=")
    if not sign:
        raise ConfigError("«{}» — нужен вид поле=значение, например "
                          "generation.temperature=0.2".format(text))
    path = path.strip()
    if not path:
        raise ConfigError("«{}» — не назван путь к полю".format(text))
    return expand(path), raw.strip()


def coerce(path: str, current: Any, raw: str) -> Any:
    """Привести значение к типу поля.

    Тип берётся из объявления конфига. Строку оставляем строкой: иначе
    ``model=gpt-5.4`` превратилось бы в число, имя вида ``2024`` — в целое,
    а шаблон ``Переведи на английский: {input}`` — в словарь YAML. Всё прочее
    разбираем как значение YAML, поэтому ``false`` становится логическим,
    ``300`` — числом, а ``null`` — пустым значением.
    """
    declared = field_type(path)
    if declared is str or (declared is None and isinstance(current, str)):
        return raw
    import yaml

    try:
        return yaml.safe_load(raw)
    except Exception:  # noqa: BLE001 — непонятное значение остаётся строкой
        return raw


def set_path(payload: Dict[str, Any], path: str, raw: str) -> None:
    """Положить значение по пути внутрь словаря конфига."""
    segments = [part for part in path.split(".") if part]
    if not segments:
        raise ConfigError("не назван путь к полю")

    holder: Dict[str, Any] = payload
    walked: List[str] = []
    for segment in segments[:-1]:
        _require(holder, segment, walked)
        walked.append(segment)
        nested = holder[segment]
        if nested is None:
            # Раздела ещё нет — создаём его из умолчаний, иначе правку
            # вида judge.min_mean=4 положить было бы некуда.
            nested = default_section(segment)
            if nested is None:
                raise ConfigError("поле «{}» не раздел, внутрь него нельзя".format(
                    ".".join(walked)))
            holder[segment] = nested
        if not isinstance(nested, dict):
            raise ConfigError("поле «{}» не раздел, внутрь него нельзя".format(
                ".".join(walked)))
        holder = nested

    leaf = segments[-1]
    _require(holder, leaf, walked)
    holder[leaf] = coerce(path, holder[leaf], raw)


def _require(holder: Dict[str, Any], segment: str, walked: Sequence[str]) -> None:
    """Проверить, что поле существует. Опечатка — ошибка, а не тихое создание."""
    if segment in holder:
        return
    where = ".".join(walked)
    available = sorted(holder)
    if not where:
        available += sorted(ALIASES)
    raise ConfigError("нет поля «{}»{}. Доступны: {}".format(
        segment, " внутри «{}»".format(where) if where else "",
        ", ".join(available)))


def apply(config: AgentConfig, assignments: Iterable[str]) -> AgentConfig:
    """Применить правки вида «путь=значение» к конфигу."""
    return apply_pairs(config, [parse_assignment(item) for item in assignments])


def apply_pairs(config: AgentConfig, pairs: Iterable[Tuple[str, str]]) -> AgentConfig:
    """То же для уже разобранных пар — так задаются именованные флаги."""
    pairs = list(pairs)
    if not pairs:
        return config
    # Ключ забираем с собой: иначе он потерялся бы при сборке конфига обратно,
    # и агенту пришлось бы искать реквизиты заново.
    payload = config.to_dict(with_secrets=True)
    for path, raw in pairs:
        set_path(payload, path, raw)
    return AgentConfig.from_dict(payload)


def append_system_prompt(config: AgentConfig, addition: str) -> AgentConfig:
    """Дописать к действующему системному промпту, а не заменить его."""
    if not addition:
        return config
    base = config.system_prompt.rstrip()
    return config.with_changes(
        system_prompt=(base + "\n\n" + addition) if base else addition)


def describe(config: AgentConfig) -> Dict[str, Any]:
    """Конфиг для показа человеку: без ключа, но целиком."""
    return config.to_dict()


def flatten(payload: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Плоский перечень «путь — значение»: так конфиг удобно печатать."""
    rows: List[Tuple[str, Any]] = []
    for key, value in payload.items():
        path = prefix + key
        if isinstance(value, dict):
            rows += flatten(value, path + ".")
        else:
            rows.append((path, value))
    return rows


def paths(config: Optional[AgentConfig] = None) -> List[str]:
    """Все пути, которые можно задать. Нужен справке и сообщениям об ошибках."""
    return [path for path, _ in flatten((config or AgentConfig()).to_dict())]
