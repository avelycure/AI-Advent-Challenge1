"""Конфиг: значения по умолчанию, проверка имён и перенос через файл."""
from __future__ import annotations

import json
import pathlib

import pytest

from llmagent import (
    DEFAULT_CONFIG,
    AgentConfig,
    Budget,
    ConfigError,
    GenerationParams,
    InputPolicy,
    JudgeConfig,
    OutputPolicy,
    Schema,
)
from llmagent.formats import Field

CONFIGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs"


def test_default_is_chatgpt():
    """Дефолт задан ментором: ChatGPT, и модель должна существовать."""
    assert DEFAULT_CONFIG.provider == "openai"
    provider, model = DEFAULT_CONFIG.resolve()
    assert provider.name.startswith("ChatGPT")
    assert model.id == "gpt-5.4-mini"
    # Модель по умолчанию обязана принимать temperature: иначе значение
    # из конфига молча срезалось бы провайдером на первом же запросе.
    assert "temperature не принимает" not in model.label


def test_default_key_comes_from_environment():
    """Ключа в конфиге нет: он берётся снаружи, и конфиг остаётся переносимым."""
    assert DEFAULT_CONFIG.api_key is None
    assert DEFAULT_CONFIG.provider_info.api_key_env == "OPENAI_API_KEY"


def test_unknown_provider_names_alternatives():
    with pytest.raises(ConfigError) as info:
        AgentConfig(provider="нетакого").resolve()
    assert "openai" in str(info.value)


def test_unknown_model_names_alternatives():
    with pytest.raises(ConfigError) as info:
        AgentConfig(provider="openai", model="gpt-1").resolve()
    assert "gpt-5.4-mini" in str(info.value)


def test_unknown_output_format_rejected():
    with pytest.raises(ConfigError):
        OutputPolicy(format="xml")


def test_template_without_place_for_input_rejected():
    with pytest.raises(ConfigError):
        InputPolicy(template="переведи текст")


def test_unknown_field_rejected():
    with pytest.raises(ConfigError) as info:
        AgentConfig.from_dict({"провайдер": "openai"})
    assert "провайдер" in str(info.value)


def rich_config() -> AgentConfig:
    """Конфиг, в котором задействована каждая часть — для проверки переноса."""
    return AgentConfig(
        name="rich", provider="openai", model="gpt-5.4", api_key="sk-secret",
        system_prompt="Отвечай коротко.",
        generation=GenerationParams(max_tokens=300, temperature=0.1, top_p=0.9,
                                    stop=["КОНЕЦ"], response_format="json_object"),
        input=InputPolicy(max_chars=100, template="Вопрос: {input}",
                          forbidden=("пароль",), on_too_long="trim"),
        output=OutputPolicy(format="json", max_attempts=3, stop_marker="%%END%%",
                            schema=Schema(fields=(Field("name", "строка", max_len=30),
                                                  Field("year", "целое число",
                                                        minimum=1900, maximum=2030)),
                                          item_count=3)),
        judge=JudgeConfig(criteria=(("точность", "верно ли"),), scale=10, min_mean=6.0,
                          agent=AgentConfig(name="judge", model="gpt-5.4-nano")),
        budget=Budget(max_tokens=1000, max_cost=0.01, max_requests=5),
    )


def test_round_trip_through_dict():
    config = rich_config()
    assert AgentConfig.from_dict(config.to_dict(with_secrets=True)) == config


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_round_trip_through_file(tmp_path, suffix):
    config = rich_config()
    path = tmp_path / ("config" + suffix)
    config.to_file(str(path), with_secrets=True)
    assert AgentConfig.from_file(str(path)) == config


def test_secrets_are_not_written_by_default(tmp_path):
    """Конфиг уходит в файлы и логи, и ключу там делать нечего."""
    config = rich_config()
    path = tmp_path / "config.json"
    config.to_file(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "***"
    assert "sk-secret" not in path.read_text(encoding="utf-8")


def test_judge_secret_is_masked_too():
    config = rich_config().with_changes(
        judge=JudgeConfig(agent=AgentConfig(name="judge", api_key="sk-judge")))
    assert config.to_dict()["judge"]["agent"]["api_key"] == "***"


def test_reconfigure_keeps_config_frozen():
    config = rich_config()
    changed = config.with_changes(model="gpt-5.4-mini")
    assert config.model == "gpt-5.4"
    assert changed.model == "gpt-5.4-mini"


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")),
                         ids=lambda p: p.name)
def test_shipped_configs_load_and_resolve(path):
    """Примеры в configs/ должны быть годными, а не просто лежать в репозитории."""
    config = AgentConfig.from_file(str(path))
    provider, model = config.resolve()
    assert provider.key == config.provider
    assert model.id == config.model


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")),
                         ids=lambda p: p.name)
def test_shipped_configs_actually_answer(path):
    """Загрузиться мало — конфиг должен доводить запрос до ответа.

    Проверка появилась после того, как configs/reviewer.yaml пролежал битым:
    YAML разбивал описания признаков судьи по запятым, конфиг при этом
    прекрасно загружался, а падал только при первом ответе.
    """
    from llmagent import Agent, Transport
    from llmchat.subagent import toolbox_for

    from conftest import ScriptedClient

    config = AgentConfig.from_file(str(path)).with_changes(
        transport=Transport(demo=True, demo_delay=0.0))
    # Второй ответ — на случай судьи или переспроса по выходной политике.
    # Набор инструментов настоящий: конфиг может их включать, и тогда без
    # набора он бы не собрался — а проверять надо годность конфига.
    agent = Agent(config, toolbox=toolbox_for(config), client=ScriptedClient(
        ['{"items": []}', "полнота: 4\nясность: 4\nобоснованность: 4"]))
    result = agent.ask("Проверочный вопрос")
    assert result.text
    assert agent.usage.requests >= 1
    if config.judge is not None:
        assert result.scores or result.judge_note
