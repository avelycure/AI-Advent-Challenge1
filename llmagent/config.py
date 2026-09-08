"""Конфиг агента: всё поведение коробки, выраженное данными.

Одно правило определяет состав: если что-то отличает одного агента от другого,
это поле конфига, а не ветка в коде. Поэтому здесь и модель, и системный
промпт, и параметры генерации, и обе политики, и судья, и бюджет.

Конфиг заморожен: агент не правит его на ходу, а получает новый через
``Agent.reconfigure`` — так его можно без опаски держать в словаре, слать
между процессами и сравнивать с эталоном.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints

from . import formats
from .errors import ConfigError
from .params import GenerationParams
from .transport import PROVIDERS, ModelInfo, ProviderInfo

SYSTEM_PROMPT = (
    "Ты — полезный ассистент, который общается с пользователем в терминале. "
    "Отвечай на языке пользователя, по существу и без лишней воды. "
    "Форматирование Markdown допустимо: списки, заголовки, блоки кода."
)

# Значение, которым подменяется ключ при сериализации без секретов.
SECRET_PLACEHOLDER = "***"


# --------------------------------------------------------------------------
# Части конфига
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InputPolicy:
    """Что происходит с запросом до отправки в модель.

    Политика намеренно скучная: обрезать, отклонить, обернуть в шаблон. Всё,
    что сложнее, — уже задача самой модели, и её место в системном промпте.
    """

    max_chars: int = 8000
    forbid_empty: bool = True
    strip: bool = True
    # Шаблон с местом под запрос: "Переведи на английский: {input}".
    template: Optional[str] = None
    # Подстроки, при которых запрос отклоняется, не доходя до провайдера.
    forbidden: Tuple[str, ...] = ()
    # "reject" — отклонить длинный запрос, "trim" — обрезать до max_chars.
    on_too_long: str = "reject"

    def __post_init__(self) -> None:
        if self.on_too_long not in ("reject", "trim"):
            raise ConfigError("input.on_too_long: допустимо reject или trim, "
                              "получено «{}»".format(self.on_too_long))
        if self.template is not None and "{input}" not in self.template:
            raise ConfigError("input.template: в шаблоне нет места {input}")


@dataclass(frozen=True)
class OutputPolicy:
    """Какой ответ считается годным и что делать, если он не такой."""

    # "free" — форма не оговаривается; иначе json, yaml или md.
    format: str = formats.FREE_FORMAT
    schema: formats.Schema = field(default_factory=formats.Schema)
    # Договорный маркер конца ответа. None — не просить и не ставить stop.
    stop_marker: Optional[str] = None
    strip_fence: bool = True
    # Сколько раз переспросить модель, назвав нарушения. 1 — не переспрашивать.
    max_attempts: int = 1
    # True — если так и не сошлось, поднять OutputRejected вместо возврата текста.
    require_valid: bool = False

    def __post_init__(self) -> None:
        try:
            formats.resolve_format(self.format)
        except KeyError as exc:
            raise ConfigError("output.format: {}".format(exc.args[0])) from exc
        if self.max_attempts < 1:
            raise ConfigError("output.max_attempts: минимум 1")

    @property
    def structured(self) -> bool:
        return self.format != formats.FREE_FORMAT

    @property
    def response_format(self) -> Optional[Dict[str, str]]:
        """Параметр запроса — второй рычаг, работающий на стороне провайдера."""
        fmt = formats.resolve_format(self.format)
        return {"type": "json_object"} if fmt is not None and fmt.json_mode else None

    @property
    def instruction(self) -> Optional[str]:
        fmt = formats.resolve_format(self.format)
        if fmt is None:
            return None
        return formats.build_instruction(fmt, self.schema, self.stop_marker)


@dataclass(frozen=True)
class JudgeConfig:
    """Оценка ответа отдельной моделью по заданным признакам.

    Своя оценка своему же ответу — слабый довод, поэтому судье можно задать
    отдельный конфиг с другим провайдером. Без него судит та же модель.
    """

    criteria: Tuple[Tuple[str, str], ...] = (
        ("полнота", "разобрано ли условие, доведено ли решение до конца"),
        ("ясность", "легко ли проследить ход мысли, есть ли структура"),
        ("обоснованность", "объяснено ли, почему способ верен, есть ли проверка"),
    )
    scale: int = 5
    max_tokens: int = 200
    # Порог годности: средняя оценка ниже — ответ помечается как слабый.
    min_mean: Optional[float] = None
    agent: Optional["AgentConfig"] = None

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ConfigError("judge.criteria: нужен хотя бы один признак")
        for item in self.criteria:
            if len(item) != 2:
                raise ConfigError(
                    "judge.criteria: признак — пара «имя, что оценивать», "
                    "а получено {} значений: {}. В YAML описание с запятыми "
                    "нужно взять в кавычки".format(len(item), list(item)[:3]))
        if self.scale < 2:
            raise ConfigError("judge.scale: шкала меньше двух баллов бессмысленна")


@dataclass(frozen=True)
class ToolPolicy:
    """Инструменты, которые модель вправе вызывать сама.

    Пустой список означает, что модель об инструментах не узнает: поле в
    запрос не уйдёт, и она ответит своими словами.
    """

    enabled: Tuple[str, ...] = ()
    # Сколько кругов «модель просит вызов — мы отвечаем результатом» допустимо
    # за один ответ. Предел нужен: модель может звать инструмент без конца.
    max_calls: int = 4

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ConfigError("tools.max_calls: минимум 1")

    @property
    def on(self) -> bool:
        return bool(self.enabled)


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = True
    # Из подряд идущих ответов в запрос уходит только последний.
    keep_last_answer: bool = True


@dataclass(frozen=True)
class Budget:
    """Предел расхода на всё время жизни агента. None — без предела."""

    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_requests: Optional[int] = None


@dataclass(frozen=True)
class Transport:
    demo: bool = False
    # Пауза заглушки в секундах. None — «живая» случайная, 0 — без паузы.
    demo_delay: Optional[float] = None


# --------------------------------------------------------------------------
# Конфиг целиком
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    name: str = "default"
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    # Ключ и второй реквизит (у Яндекса — каталог). None означает «взять из
    # окружения или из файла»: так конфиг можно хранить в репозитории.
    api_key: Optional[str] = None
    api_extra: Optional[str] = None
    system_prompt: str = SYSTEM_PROMPT
    generation: GenerationParams = field(default_factory=GenerationParams)
    input: InputPolicy = field(default_factory=InputPolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)
    judge: Optional[JudgeConfig] = None
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    budget: Budget = field(default_factory=Budget)
    transport: Transport = field(default_factory=Transport)

    # --- разрешение имён в объекты каталога ----------------------------
    def resolve(self) -> Tuple[ProviderInfo, ModelInfo]:
        """Найти провайдера и модель по именам. Ошибка — сразу и с перечнем."""
        provider = PROVIDERS.get(self.provider)
        if provider is None:
            raise ConfigError("неизвестный провайдер «{}»; доступны: {}".format(
                self.provider, ", ".join(sorted(PROVIDERS))))
        for model in provider.models:
            if model.id == self.model:
                return provider, model
        raise ConfigError("у провайдера {} нет модели «{}»; доступны: {}".format(
            provider.name, self.model, ", ".join(m.id for m in provider.models)))

    @property
    def provider_info(self) -> ProviderInfo:
        return self.resolve()[0]

    @property
    def model_info(self) -> ModelInfo:
        return self.resolve()[1]

    def with_changes(self, **changes: Any) -> "AgentConfig":
        return replace(self, **changes)

    # --- сериализация --------------------------------------------------
    def to_dict(self, with_secrets: bool = False) -> Dict[str, Any]:
        """Конфиг словарём. Ключ по умолчанию не выносится наружу.

        Умолчание именно такое, потому что этот словарь уходит в файлы, логи
        и тело HTTP-ответа, а ключ провайдера там не нужен ни разу.
        """
        payload = _to_plain(self)
        if not with_secrets:
            for name in ("api_key", "api_extra"):
                if payload.get(name):
                    payload[name] = SECRET_PLACEHOLDER
            judge = payload.get("judge")
            if isinstance(judge, dict) and isinstance(judge.get("agent"), dict):
                for name in ("api_key", "api_extra"):
                    if judge["agent"].get(name):
                        judge["agent"][name] = SECRET_PLACEHOLDER
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentConfig":
        try:
            return _from_plain(cls, payload)
        except ConfigError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError("конфиг не разобран: {}".format(exc)) from exc

    @classmethod
    def from_file(cls, path: str) -> "AgentConfig":
        """Прочитать конфиг из JSON или YAML — по расширению файла."""
        expanded = os.path.expanduser(path)
        try:
            with open(expanded, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise ConfigError("конфиг не прочитан: {}".format(exc)) from exc

        if expanded.endswith((".yaml", ".yml")):
            import yaml

            payload = yaml.safe_load(raw)
        else:
            payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ConfigError("в файле {} ожидался объект с полями конфига".format(path))
        return cls.from_dict(payload)

    def to_file(self, path: str, with_secrets: bool = False) -> None:
        payload = self.to_dict(with_secrets=with_secrets)
        expanded = os.path.expanduser(path)
        with open(expanded, "w", encoding="utf-8") as handle:
            if expanded.endswith((".yaml", ".yml")):
                import yaml

                yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
            else:
                json.dump(payload, handle, ensure_ascii=False, indent=2)


# Конфиг по умолчанию: ChatGPT gpt-5.4-mini — штатная модель провайдера,
# самая дешёвая из пригодных и, в отличие от gpt-5.5 и gpt-6-astra,
# принимающая temperature. Ключ берётся из OPENAI_API_KEY или ~/.openai-key.
DEFAULT_CONFIG = AgentConfig()


# --------------------------------------------------------------------------
# Превращение вложенных dataclass в словарь и обратно
# --------------------------------------------------------------------------

def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _from_plain(target: type, payload: Any) -> Any:
    """Собрать dataclass из словаря, восстановив типы вложенных полей."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ConfigError("ожидался объект для {}, получено {}".format(
            target.__name__, type(payload).__name__))

    known = {f.name: f for f in fields(target)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ConfigError("{}: неизвестные поля {}; доступны: {}".format(
            target.__name__, ", ".join(sorted(unknown)), ", ".join(sorted(known))))

    kwargs: Dict[str, Any] = {}
    for name, spec in known.items():
        if name not in payload:
            continue
        kwargs[name] = _restore(target, name, payload[name])
    return target(**kwargs)


# Поля, у которых из аннотации тип не вытащить простым способом: аннотации
# в модуле строковые (from __future__ import annotations), а разбирать их
# на лету дороже и хрупче, чем назвать восстановление явно.
_NESTED: Dict[str, Any] = {
    "generation": GenerationParams,
    "input": InputPolicy,
    "output": OutputPolicy,
    "judge": JudgeConfig,
    "tools": ToolPolicy,
    "history": HistoryConfig,
    "budget": Budget,
    "transport": Transport,
    "schema": formats.Schema,
    "agent": "AgentConfig",
}


# Поля-кортежи, опознаваемые парой «класс, имя».
_TUPLE_FIELDS = {
    (InputPolicy, "forbidden"),
    (ToolPolicy, "enabled"),
}


def field_type(path: str) -> Optional[type]:
    """Объявленный тип поля по пути внутрь конфига. None — путь неизвестен.

    Тип нужен разбору правок из командной строки. Смотреть на текущее значение
    нельзя: у строковых полей вроде ``input.template`` по умолчанию лежит
    пустое значение, и по нему поле не отличить от числового.
    """
    holder: Any = AgentConfig
    segments = [part for part in path.split(".") if part]
    for index, segment in enumerate(segments):
        if not (is_dataclass(holder) and isinstance(holder, type)):
            return None
        try:
            hints = get_type_hints(holder)
        except Exception:  # noqa: BLE001 — незнакомая аннотация не повод падать
            return None
        if segment not in hints:
            return None
        annotation = _unwrap_optional(hints[segment])
        if index == len(segments) - 1:
            return annotation if isinstance(annotation, type) else None
        holder = annotation
    return None


def _unwrap_optional(annotation: Any) -> Any:
    """Снять Optional: для разбора значения важен тип, а не его допустимая пустота."""
    if get_origin(annotation) is Union:
        actual = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(actual) == 1:
            return actual[0]
    return annotation


def section_names() -> Tuple[str, ...]:
    """Имена вложенных разделов конфига — те, внутрь которых можно углубляться."""
    return tuple(_NESTED)


def default_section(name: str) -> Optional[Dict[str, Any]]:
    """Раздел конфига со значениями по умолчанию, словарём.

    Нужен для правок вида ``judge.min_mean=4``, когда судьи ещё нет: раздел
    создаётся целиком из умолчаний, и правка ложится в него.
    """
    target = _NESTED.get(name)
    if target is None:
        return None
    if target == "AgentConfig":
        target = AgentConfig
    return _to_plain(target())


def _restore(owner: type, name: str, value: Any) -> Any:
    """Восстановить значение поля из простого вида.

    Поле опознаётся парой «класс, имя», а не одним именем. По имени было
    нельзя: ``enabled`` у истории — признак, а у инструментов — список имён,
    и разбор по имени превращал один в другой.
    """
    if (owner, name) in _TUPLE_FIELDS:
        return tuple(value or ())
    if (owner, name) == (JudgeConfig, "criteria"):
        return tuple(tuple(item) for item in value or ())
    if (owner, name) == (formats.Schema, "fields"):
        return tuple(_from_plain(formats.Field, item) for item in value or ())
    target = _NESTED.get(name)
    if target is not None:
        return _from_plain(AgentConfig if target == "AgentConfig" else target, value)
    return value