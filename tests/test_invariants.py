"""Инварианты: отдельное хранение, явная проверка и объяснённый отказ."""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    ConfigError,
    Invariant,
    InvariantCheckError,
    InvariantConfig,
    ToolOutcome,
    ToolPolicy,
    ToolSpec,
    Toolbox,
    Transport,
)
from llmagent.transport import (
    Completion,
    ToolCall,
    count_message_tokens,
    count_text_tokens,
)

from conftest import ScriptedClient

FAST = Transport(demo=True, demo_delay=0.0)
RULE = Invariant(
    id="stack.python",
    category="stack",
    rule="Серверная часть пишется только на Python",
    reason="Команда поддерживает единый Python-стек",
)


def guarded(client, *rules: Invariant) -> Agent:
    config = AgentConfig(
        name="guarded",
        transport=FAST,
        invariants=InvariantConfig(rules=tuple(rules or (RULE,))),
    )
    return Agent(config, client=client)


def verdict(allowed: bool, identifier: str = "", conflict: str = "") -> str:
    if allowed:
        return '{"allowed":true,"violations":[]}'
    return (
        '{"allowed":false,"violations":[{"id":"' + identifier
        + '","conflict":"' + conflict + '"}]}'
    )


def discovery(*items, declaration_only: bool = False) -> str:
    import json

    return json.dumps({
        "declaration_only": declaration_only,
        "invariants": list(items),
    }, ensure_ascii=False)


def test_invariants_round_trip_as_config_not_conversation_messages():
    config = guarded(ScriptedClient("unused")).config
    restored = AgentConfig.from_dict(config.to_dict())

    assert restored.invariants.rules == (RULE,)
    assert restored.invariants is not restored.history
    assert "invariants" not in restored.system_prompt.lower()


def test_compatible_request_is_checked_and_answer_sees_invariants():
    client = ScriptedClient([
        verdict(True),
        "Предлагаю FastAPI.",
        verdict(True),
    ])
    agent = guarded(client)

    result = agent.ask("Предложи API для сервиса")

    assert result.text == "Предлагаю FastAPI."
    assert not result.refused
    assert result.invariant_checks == [
        {"phase": "request", "status": "allowed"},
        {"phase": "response", "status": "allowed"},
    ]
    assert len(client.calls) == 3
    main_system = client.calls[1][0]["content"]
    assert "Перед выбором решения явно сверь" in main_system
    assert RULE.id in main_system and RULE.rule in main_system
    # Служебные проверки не становятся репликами диалога.
    assert [message.content for message in agent.conversation.messages] == [
        "Предложи API для сервиса", "Предлагаю FastAPI."]
    assert agent.usage.by_kind["инварианты"].requests == 2
    assert result.total_tokens == agent.usage.total_tokens


def test_conflicting_request_is_refused_before_main_generation():
    client = ScriptedClient([
        verdict(False, RULE.id, "Пользователь требует заменить Python на Go"),
    ])
    agent = guarded(client)
    agent.enable_task_state_automation()

    result = agent.ask("Перепиши сервер с Python на Go")

    assert result.refused and result.ok
    assert len(client.calls) == 1
    assert RULE.id in result.text
    assert RULE.rule in result.text
    assert "заменить Python на Go" in result.text
    assert "подобрать вариант" in result.text
    assert result.invariant_violations == [{
        "id": RULE.id,
        "category": RULE.category,
        "rule": RULE.rule,
        "conflict": "Пользователь требует заменить Python на Go",
    }]
    assert [message.content for message in agent.conversation.messages] == [
        "Перепиши сервер с Python на Go", result.text]
    assert not agent.task_state.active


def test_answer_that_breaks_an_invariant_is_replaced_by_refusal():
    client = ScriptedClient([
        verdict(True),
        "Перепишем сервис на Go: так будет быстрее.",
        verdict(False, RULE.id, "Ответ предлагает реализацию на Go"),
    ])
    agent = guarded(client)

    result = agent.ask("Как ускорить сервис?")

    assert result.refused
    assert "реализацию на Go" in result.text
    assert "Перепишем сервис" not in result.text
    assert agent.conversation.messages[-1].content == result.text
    assert result.invariant_checks[-1] == {
        "phase": "response", "status": "refused"}


def test_conflicting_tool_call_is_blocked_before_handler_runs():
    handled = []

    def handler(arguments):
        handled.append(arguments)
        return ToolOutcome("переписано")

    class ToolClient:
        def __init__(self):
            self.calls = []

        def validate_key(self):
            return None

        def complete(self, model_ref, messages, max_tokens, temperature=0.7,
                     top_p=None, stop=None, response_format=None, tools=None):
            self.calls.append([dict(message) for message in messages])
            step = len(self.calls)
            if step in (1, 5):
                text = verdict(True)
                return Completion(text, count_message_tokens(messages),
                                  count_text_tokens(text), elapsed=0.001)
            if step == 2:
                return Completion(
                    "", count_message_tokens(messages), 5, elapsed=0.001,
                    tool_calls=[ToolCall(
                        "call-1", "rewrite_server", '{"language":"go"}')])
            if step == 3:
                text = verdict(False, RULE.id,
                               "Инструмент переписывает сервер на Go")
                return Completion(text, count_message_tokens(messages),
                                  count_text_tokens(text), elapsed=0.001)
            text = "Оставляем сервер на Python."
            return Completion(text, count_message_tokens(messages),
                              count_text_tokens(text), elapsed=0.001)

    config = AgentConfig(
        transport=FAST,
        invariants=InvariantConfig(rules=(RULE,)),
        tools=ToolPolicy(enabled=("rewrite_server",)),
    )
    box = Toolbox([ToolSpec(
        "rewrite_server", "Переписать сервер", {"type": "object"}, handler)])
    agent = Agent(config, client=ToolClient(), toolbox=box)

    result = agent.ask("Ускорь сервер совместимым способом")

    assert handled == []
    assert result.text == "Оставляем сервер на Python."
    assert result.tool_calls[0]["ok"] is False
    assert result.invariant_checks == [
        {"phase": "request", "status": "allowed"},
        {"phase": "tool", "tool": "rewrite_server", "status": "refused"},
        {"phase": "response", "status": "allowed"},
    ]


def test_unknown_rule_from_controller_fails_closed():
    client = ScriptedClient([
        verdict(False, "invented-rule", "выдуманный конфликт"),
    ])
    agent = guarded(client)

    with pytest.raises(InvariantCheckError, match="неизвестный инвариант"):
        agent.ask("Любой запрос")

    assert agent.conversation.messages == []


def test_duplicate_ids_are_rejected_in_config():
    with pytest.raises(ConfigError, match="повторяются id"):
        InvariantConfig(rules=(RULE, RULE))


def test_runtime_invariants_can_be_added_replaced_and_removed():
    agent = Agent(AgentConfig(transport=FAST), client=ScriptedClient("unused"))
    agent.conversation.exact_context = 123

    assert agent.set_invariant(RULE.id, RULE.category, RULE.rule, RULE.reason)
    assert agent.config.invariants.rules == (RULE,)
    assert agent.conversation.exact_context == 0
    assert not agent.set_invariant(RULE.id, RULE.category, RULE.rule, RULE.reason)

    assert agent.set_invariant(
        RULE.id, RULE.category, "Сервер остаётся на Python", "Новое решение")
    assert len(agent.config.invariants.rules) == 1
    assert agent.config.invariants.rules[0].reason == "Новое решение"
    assert agent.remove_invariant(RULE.id)
    assert not agent.config.invariants.rules
    assert not agent.remove_invariant(RULE.id)


def test_runtime_invariants_survive_new_session():
    agent = Agent(AgentConfig(transport=FAST), client=ScriptedClient("unused"))
    agent.set_invariant(RULE.id, RULE.category, RULE.rule, RULE.reason)
    agent.new_session()
    assert agent.config.invariants.rules == (RULE,)


def test_explicit_rule_in_an_ordinary_message_becomes_an_invariant():
    kotlin = {
        "id": "stack.kotlin-only",
        "category": "stack",
        "rule": "В проекте используется только Kotlin; Java запрещена",
        "reason": "Явно установлено пользователем",
    }
    client = ScriptedClient([
        discovery(kotlin, declaration_only=True),
        verdict(False, "stack.kotlin-only",
                "Запрос требует использовать запрещённую Java"),
    ])
    agent = Agent(AgentConfig(transport=FAST), client=client)
    agent.enable_task_state_automation()

    saved = agent.ask(
        "В проекте мы используем Kotlin и только. Java использовать нельзя.")

    assert agent.config.invariants.rules[0].id == "stack.kotlin-only"
    assert saved.detected_invariants == [kotlin]
    assert saved.invariant_checks == [
        {"phase": "discovery", "status": "saved"},
    ]
    assert "Зафиксировал инвариант" in saved.text
    assert not agent.task_state.active
    assert len(client.calls) == 1
    assert saved.total_tokens == agent.usage.total_tokens

    refused = agent.ask("Напиши новый сервис на Java")
    assert refused.refused
    assert "stack.kotlin-only" in refused.text
    assert len(client.calls) == 2


def test_one_off_only_instruction_is_not_saved_as_an_invariant():
    client = ScriptedClient("JVM — виртуальная машина Java.")
    agent = Agent(AgentConfig(transport=FAST), client=client)

    result = agent.ask("Ответь только кратко: что такое JVM?")

    assert not agent.config.invariants.rules
    assert result.detected_invariants == []
    assert result.invariant_checks == []
    assert len(client.calls) == 1
    assert result.total_tokens == agent.usage.total_tokens


def test_invariant_and_task_in_one_message_use_the_new_rule_immediately():
    kotlin = {
        "id": "stack.kotlin-only",
        "category": "stack",
        "rule": "В проекте используется только Kotlin; Java запрещена",
        "reason": "Явно установлено пользователем",
    }
    client = ScriptedClient([
        discovery(kotlin, declaration_only=False),
        verdict(True),
        "Предлагаю реализацию на Kotlin.",
        verdict(True),
    ])
    agent = Agent(AgentConfig(transport=FAST), client=client)

    result = agent.ask(
        "Используем только Kotlin, Java запрещена. Спроектируй новый сервис.")

    assert result.text == "Предлагаю реализацию на Kotlin."
    assert result.detected_invariants == [kotlin]
    assert "stack.kotlin-only" in client.calls[2][0]["content"]
    assert len(client.calls) == 4


def test_agent_without_invariants_keeps_the_single_call_pipeline():
    client = ScriptedClient("обычный ответ")
    result = Agent(AgentConfig(transport=FAST), client=client).ask("вопрос")
    assert result.text == "обычный ответ"
    assert len(client.calls) == 1
    assert result.invariant_checks == []
