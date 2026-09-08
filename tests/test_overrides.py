"""Правки конфига при запуске: пути, типы, короткие имена, ошибки."""
from __future__ import annotations

import pytest

from llmagent import AgentConfig, ConfigError, overrides
from llmagent.catalog import find_model

BASE = AgentConfig()


def test_named_field_changes_and_the_rest_survives():
    changed = overrides.apply(BASE, ["generation.temperature=0.2"])
    assert changed.generation.temperature == 0.2
    # Всё прочее должно остаться ровно как было — в этом весь смысл правки.
    assert changed.generation.max_tokens == BASE.generation.max_tokens
    assert changed.input.max_chars == BASE.input.max_chars
    assert changed.output.format == BASE.output.format
    assert changed.provider == BASE.provider


@pytest.mark.parametrize("assignment,path,expected", [
    ("model=gpt-5.4", "model", "gpt-5.4"),                       # строка остаётся строкой
    ("name=2024", "name", "2024"),                               # даже если похожа на число
    ("generation.max_tokens=300", "generation.max_tokens", 300),  # число
    ("generation.temperature=0.35", "generation.temperature", 0.35),  # дробь
    ("history.enabled=false", "history.enabled", False),         # логическое
    ("output.require_valid=true", "output.require_valid", True),
    ("generation.max_tokens=null", "generation.max_tokens", None),  # пустое значение
    ("budget.max_cost=0.05", "budget.max_cost", 0.05),
])
def test_value_types(assignment, path, expected):
    config = overrides.apply(BASE, [assignment])
    holder = config
    for segment in path.split("."):
        holder = getattr(holder, segment)
    assert holder == expected and type(holder) is type(expected)


def test_list_value():
    config = overrides.apply(BASE, ["generation.stop=[КОНЕЦ, СТОП]"])
    assert config.generation.stop == ["КОНЕЦ", "СТОП"]


def test_tuple_field_stays_a_tuple():
    """Часть полей внутри конфига — кортежи, и правка не должна это ломать."""
    config = overrides.apply(BASE, ["input.forbidden=[пароль, токен]"])
    assert config.input.forbidden == ("пароль", "токен")


def test_deep_path_inside_the_judge():
    config = overrides.apply(BASE, ["judge.agent.model=gpt-5.4-nano",
                                    "judge.agent.name=свой-судья"])
    assert config.judge.agent.model == "gpt-5.4-nano"
    assert config.judge.agent.name == "свой-судья"


def test_missing_section_is_created_from_defaults():
    """Судьи в конфиге нет, а правка внутрь него должна работать."""
    assert BASE.judge is None
    config = overrides.apply(BASE, ["judge.min_mean=4"])
    assert config.judge is not None
    assert config.judge.min_mean == 4
    assert config.judge.scale == 5           # остальное из умолчаний


def test_short_names_work_like_flags():
    config = overrides.apply(BASE, ["temperature=0.1", "max_tokens=80", "format=json",
                                    "max_cost=0.02", "attempts=3", "max_chars=500"])
    assert config.generation.temperature == 0.1
    assert config.generation.max_tokens == 80
    assert config.output.format == "json"
    assert config.output.max_attempts == 3
    assert config.budget.max_cost == 0.02
    assert config.input.max_chars == 500


def test_secret_survives_the_round_trip():
    """Правка не должна терять ключ: агенту иначе пришлось бы искать его снова."""
    config = overrides.apply(BASE.with_changes(api_key="sk-secret"), ["temperature=0.1"])
    assert config.api_key == "sk-secret"


@pytest.mark.parametrize("bad,expect", [
    ("outupt.format=json", "нет поля «outupt»"),
    ("generation.tempreture=1", "внутри «generation»"),
    ("model", "нужен вид поле=значение"),
    ("=0.2", "не назван путь"),
    ("model.deeper=x", "не раздел"),
])
def test_mistakes_are_reported(bad, expect):
    with pytest.raises(ConfigError) as info:
        overrides.apply(BASE, [bad])
    assert expect in str(info.value)


def test_unknown_top_level_field_lists_short_names_too():
    """Подсказка должна включать и короткие имена, иначе она вводит в заблуждение."""
    with pytest.raises(ConfigError) as info:
        overrides.apply(BASE, ["temperatura=0.2"])
    assert "temperature" in str(info.value)


def test_later_assignment_wins():
    config = overrides.apply(BASE, ["temperature=0.2", "temperature=0.9"])
    assert config.generation.temperature == 0.9


def test_append_system_prompt_adds_and_does_not_replace():
    config = overrides.append_system_prompt(BASE, "Отвечай только числами.")
    assert BASE.system_prompt in config.system_prompt
    assert config.system_prompt.endswith("Отвечай только числами.")


def test_append_to_empty_prompt_does_not_leave_blank_lines():
    config = overrides.append_system_prompt(BASE.with_changes(system_prompt=""), "Кратко.")
    assert config.system_prompt == "Кратко."


def test_invalid_value_is_caught_by_the_config_itself():
    with pytest.raises(ConfigError):
        overrides.apply(BASE, ["output.format=xml"])
    with pytest.raises(ConfigError):
        overrides.apply(BASE, ["input.on_too_long=обрезать"])


def test_every_field_is_reachable():
    """Обещано «любое поле» — значит путь должен вести к каждому."""
    paths = overrides.paths()
    assert len(paths) >= 33
    for path in paths:
        if path in ("api_key", "api_extra"):
            continue                      # значение маскируется, правку не сверить
        overrides.set_path(BASE.to_dict(with_secrets=True), path, "null")


# --------------------------------------------------------------------------
# Поиск модели
# --------------------------------------------------------------------------

def test_model_found_without_naming_the_provider():
    assert find_model("gpt-5.4") == ("openai", "gpt-5.4")
    assert find_model("openai/gpt-5.4-nano") == ("openai", "gpt-5.4-nano")


@pytest.mark.parametrize("bad,expect", [
    ("gpt-5.9", "Похожие"),
    ("нетакого/x", "неизвестный провайдер"),
    ("openai/нет", "нет модели"),
    ("", "не названа модель"),
])
def test_model_mistakes_are_reported(bad, expect):
    with pytest.raises(ConfigError) as info:
        find_model(bad)
    assert expect in str(info.value)


@pytest.mark.parametrize("spec,expected", [
    ("qwen/qwen3.6-27b", ("groq", "qwen/qwen3.6-27b")),
    ("groq/compound", ("groq", "groq/compound")),
    ("openai/gpt-oss-120b", ("groq", "openai/gpt-oss-120b")),
])
def test_model_names_may_contain_a_slash(spec, expected):
    """У Groq косая черта входит в само имя модели, а не разделяет провайдера."""
    assert find_model(spec) == expected


def test_provider_prefix_still_works_for_plain_names():
    assert find_model("openai/gpt-5.4") == ("openai", "gpt-5.4")


@pytest.mark.parametrize("path", ["input.template", "output.stop_marker", "system_prompt"])
def test_string_fields_are_not_parsed_as_yaml(path):
    """Строковое поле с пустым значением по умолчанию тоже должно оставаться строкой."""
    value = "Переведи на английский: {input}" if path == "input.template" else "а: б"
    config = overrides.apply(BASE, ["{}={}".format(path, value)])
    holder = config
    for segment in path.split("."):
        holder = getattr(holder, segment)
    assert holder == value


@pytest.mark.parametrize("bad,expect", [
    ("generation.max_tokens=true", "получено логическое"),
    ("output.require_valid=да", "true или false"),
    ("budget.max_requests=3.7", "целое число"),
    ("budget.max_cost=пять", "нужно число"),
    ("generation.temperature=жарко", "нужно число"),
])
def test_value_must_match_the_declared_type(bad, expect):
    """Негодное значение отвергается сразу, а не падает много позже.

    Проверка появилась после того, как --max-cost 0.0000001 уронил агента
    внутри проверки бюджета: YAML не считает числом запись 1e-07, значение
    доехало до конфига строкой и сравнилось с числом.
    """
    with pytest.raises(ConfigError) as info:
        overrides.apply(BASE, [bad])
    assert expect in str(info.value)


@pytest.mark.parametrize("assignment,path,expected", [
    ("budget.max_cost=1e-07", "budget.max_cost", 1e-07),   # запись без точки
    ("generation.temperature=1", "generation.temperature", 1.0),   # целое в дробное
    ("budget.max_requests=3.0", "budget.max_requests", 3),         # ровное дробное
])
def test_numbers_are_brought_to_the_declared_type(assignment, path, expected):
    config = overrides.apply(BASE, [assignment])
    holder = config
    for segment in path.split("."):
        holder = getattr(holder, segment)
    assert holder == expected and type(holder) is type(expected)
