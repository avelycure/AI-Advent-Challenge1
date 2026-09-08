"""Инструменты: модель вызывает под-агента сама."""
from __future__ import annotations

import pytest

from llmagent import (
    DEFAULT_CONFIG,
    Agent,
    AgentConfig,
    AgentError,
    LLMError,
    ToolOutcome,
    Toolbox,
    ToolPolicy,
    Transport,
    spawn_agent_spec,
)
from llmagent.tools import USAGE_KEY
from llmagent.usage import MAIN, SUB, TOOL

from conftest import ScriptedClient, ToolCallingClient

FAST_DEMO = Transport(demo=True, demo_delay=0.0)
ARGUMENTS = '{"config": "books-json", "question": "Год выхода Python?"}'


def toolbox(handler=None, configs=("books-json", "frugal")) -> Toolbox:
    return Toolbox([spawn_agent_spec(
        list(configs), handler or (lambda arguments: ToolOutcome("1991")))])


def agent_with(client, box=None, **changes) -> Agent:
    changes.setdefault("tools", ToolPolicy(enabled=("spawn_agent",)))
    config = AgentConfig(transport=FAST_DEMO, **changes)
    return Agent(config, client=client, toolbox=box or toolbox())


# --------------------------------------------------------------------------
# Согласие конфига и набора
# --------------------------------------------------------------------------

def test_tools_are_off_by_default():
    """Пока в конфиге ничего не включено, модель об инструментах не узнает."""
    assert DEFAULT_CONFIG.tools.enabled == ()
    agent = Agent(DEFAULT_CONFIG.with_changes(transport=FAST_DEMO),
                  client=ScriptedClient("ответ"), toolbox=toolbox())
    assert not agent.toolbox
    agent.ask("вопрос")
    assert agent._client.kwargs[0]["tools"] is None


def test_config_names_a_tool_that_does_not_exist():
    """Человек написал имя и ждёт инструмента — промолчать нельзя."""
    with pytest.raises(AgentError) as info:
        Agent(AgentConfig(tools=ToolPolicy(enabled=("нетакого",)), transport=FAST_DEMO),
              client=ScriptedClient("ответ"), toolbox=toolbox())
    assert "нетакого" in str(info.value) and "spawn_agent" in str(info.value)


def test_tools_can_be_switched_on_during_the_dialogue():
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=ScriptedClient("ответ"),
                  toolbox=toolbox())
    assert not agent.toolbox
    agent.reconfigure(tools=ToolPolicy(enabled=("spawn_agent",)))
    assert agent.toolbox.names == ["spawn_agent"]


def test_available_configs_are_named_in_the_description():
    """Без перечня модель придумывала бы имена конфигов."""
    definition = toolbox().definitions()[0]["function"]
    assert "books-json" in definition["description"]
    assert definition["parameters"]["properties"]["config"]["enum"] == \
        ["books-json", "frugal"]


# --------------------------------------------------------------------------
# Круг вызова
# --------------------------------------------------------------------------

def test_model_calls_the_tool_and_then_answers():
    asked = []

    def handler(arguments):
        asked.append(arguments)
        return ToolOutcome("Python появился в 1991 году")

    client = ToolCallingClient("spawn_agent", ARGUMENTS, "В 1991 году.", rounds=1)
    agent = agent_with(client, toolbox(handler))
    result = agent.ask("Когда появился Python?")

    assert result.ok and result.text == "В 1991 году."
    assert asked == [{"config": "books-json", "question": "Год выхода Python?"}]
    assert [(call["name"], call["ok"]) for call in result.tool_calls] == \
        [("spawn_agent", True)]
    # Результат инструмента ушёл модели своим сообщением, а не подмешался в вопрос.
    roles = [message["role"] for message in client.calls[1]]
    assert roles == ["system", "user", "assistant", "tool"]


def test_tool_round_is_counted_separately():
    client = ToolCallingClient("spawn_agent", ARGUMENTS, "готово", rounds=1)
    agent = agent_with(client)
    agent.ask("вопрос")
    assert agent.usage.by_kind[MAIN].requests == 1
    assert agent.usage.by_kind[TOOL].requests == 1


def test_tool_result_does_not_land_in_the_dialogue():
    """В памяти остаётся ответ, а не служебная переписка с инструментом."""
    agent = agent_with(ToolCallingClient("spawn_agent", ARGUMENTS, "ответ", rounds=1))
    agent.ask("вопрос")
    assert [(m.role, m.content) for m in agent.conversation.messages] == \
        [("user", "вопрос"), ("assistant", "ответ")]


# --------------------------------------------------------------------------
# Ошибки модели
# --------------------------------------------------------------------------

def test_unknown_tool_name_is_explained_to_the_model():
    """Имя придумывает модель, значит ошибиться может — и должна узнать об этом."""
    client = ToolCallingClient("нетакой_инструмент", "{}", "понял", rounds=1)
    agent = agent_with(client)
    result = agent.ask("вопрос")
    assert result.text == "понял"
    assert result.tool_calls[0]["ok"] is False
    assert "нет" in result.tool_calls[0]["text"].lower()
    # Объяснение ушло модели сообщением роли tool.
    assert any(m["role"] == "tool" and "spawn_agent" in str(m["content"])
               for m in client.calls[1])


@pytest.mark.parametrize("arguments,expect", [
    ("{не json", "не разобраны"),
    ('"строка"', "объектом"),
    ('{"config": "books-json"}', "Нужны оба поля"),
    ('{"question": "а"}', "Нужны оба поля"),
])
def test_bad_arguments_are_explained_to_the_model(arguments, expect):
    from llmchat.subagent import toolbox_for

    client = ToolCallingClient("spawn_agent", arguments, "ладно", rounds=1)
    agent = agent_with(client, toolbox_for(AgentConfig(transport=FAST_DEMO)))
    result = agent.ask("вопрос")
    assert result.text == "ладно"
    assert result.tool_calls[0]["ok"] is False
    assert expect in result.tool_calls[0]["text"]


def test_a_broken_tool_does_not_break_the_dialogue():
    def handler(arguments):
        raise RuntimeError("всё сломалось")

    agent = agent_with(ToolCallingClient("spawn_agent", ARGUMENTS, "отвечаю сам",
                                         rounds=1), toolbox(handler))
    result = agent.ask("вопрос")
    assert result.text == "отвечаю сам"
    assert result.tool_calls[0]["ok"] is False


# --------------------------------------------------------------------------
# Предел кругов
# --------------------------------------------------------------------------

def test_rounds_are_capped_and_tools_are_withdrawn_on_the_last_one():
    """Предлагать то, чем уже нельзя пользоваться, значит выпрашивать пустоту."""
    client = ToolCallingClient("spawn_agent", ARGUMENTS, "больше не могу")
    agent = agent_with(client, tools=ToolPolicy(enabled=("spawn_agent",), max_calls=2))
    result = agent.ask("зови без конца")

    assert len(result.tool_calls) == 2
    assert client.offered == [True, True, False]
    assert result.text == "больше не могу"
    # Модели прямо сказано, почему инструментов больше нет.
    assert any("Предел вызовов" in str(m.get("content")) for m in client.calls[-1])


def test_a_model_that_never_answers_gets_a_clear_error():
    class Stubborn(ToolCallingClient):
        def complete(self, *args, **kwargs):
            from llmagent.transport import Completion, ToolCall
            from llmagent.transport import count_message_tokens

            self.offered.append(bool(kwargs.get("tools")))
            return Completion("", count_message_tokens(args[1] if len(args) > 1
                                                       else kwargs["messages"]), 5,
                              elapsed=0.001,
                              tool_calls=[ToolCall("c", "spawn_agent", ARGUMENTS)])

    agent = agent_with(Stubborn("spawn_agent", ARGUMENTS),
                       tools=ToolPolicy(enabled=("spawn_agent",), max_calls=1))
    with pytest.raises(LLMError) as info:
        agent.ask("вопрос")
    assert "не дала текстового ответа" in str(info.value)
    assert len(agent.conversation) == 0


# --------------------------------------------------------------------------
# Расход инструмента
# --------------------------------------------------------------------------

def test_tool_spending_lands_in_the_common_count():
    """Под-агент платит из того же кошелька — и когда его зовёт модель тоже."""
    def handler(arguments):
        return ToolOutcome("1991", detail={USAGE_KEY: {
            "prompt_tokens": 100, "completion_tokens": 40, "seconds": 1.2,
            "cost": 0.0004, "currency": "USD"}})

    agent = agent_with(ToolCallingClient("spawn_agent", ARGUMENTS, "в 1991", rounds=1),
                       toolbox(handler))
    agent.ask("когда?")
    assert agent.usage.by_kind[SUB].requests == 1
    assert agent.usage.by_kind[SUB].total_tokens == 140
    assert agent.usage.cost("USD") > 0.0004


def test_a_tool_that_says_nothing_about_spending_is_not_counted():
    agent = agent_with(ToolCallingClient("spawn_agent", ARGUMENTS, "ответ", rounds=1))
    agent.ask("вопрос")
    assert agent.usage.by_kind[SUB].requests == 0


# --------------------------------------------------------------------------
# Настоящий под-агент отдельным процессом
# --------------------------------------------------------------------------

def test_the_tool_really_launches_a_subagent_process():
    from llmchat.subagent import toolbox_for

    parent = AgentConfig(name="parent", transport=FAST_DEMO)
    outcome = toolbox_for(parent).run(
        "spawn_agent", '{"config": "frugal", "question": "Что такое хороший код"}')
    assert outcome.ok, outcome.text
    assert outcome.text
    delegation = outcome.detail["delegation"]
    assert delegation.session_id and delegation.model == "gpt-5.4-nano"
    assert outcome.usage["prompt_tokens"] > 0


def test_forgotten_toolbox_is_not_blamed_on_the_config():
    """Имя в конфиге может быть верным, а инструменты просто не переданы."""
    with pytest.raises(AgentError) as info:
        Agent(AgentConfig(tools=ToolPolicy(enabled=("spawn_agent",)),
                          transport=FAST_DEMO), client=ScriptedClient("ответ"))
    assert "вызывающий их не передал" in str(info.value)
