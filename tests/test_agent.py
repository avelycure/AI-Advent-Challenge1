"""Агент целиком: конвейер, история, переспрос, судья, бюджет."""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    Budget,
    BudgetExceeded,
    GenerationParams,
    HistoryConfig,
    InputPolicy,
    InputRejected,
    JudgeConfig,
    OutputPolicy,
    OutputRejected,
    Schema,
    Transport,
)
from llmagent.errors import LLMError
from llmagent.formats import Field

from conftest import FailingClient, ScriptedClient

SCHEMA = Schema(fields=(Field("name", "строка", max_len=30),), item_count=1)
GOOD = '{"items": [{"name": "Python"}]}'
BAD = '{"items": [{"name": "Python"}, {"name": "Rust"}]}'


def agent_with(client, **changes) -> Agent:
    config = AgentConfig(name="t", transport=Transport(demo=True, demo_delay=0.0),
                         **changes)
    return Agent(config, client=client)


# --------------------------------------------------------------------------
# Основной путь
# --------------------------------------------------------------------------

def test_answer_carries_text_and_measurements():
    agent = agent_with(ScriptedClient("Привет!"))
    result = agent.ask("Здравствуй")
    assert result.ok and result.text == "Привет!"
    assert result.total_tokens > 0
    assert result.cost is not None and result.cost > 0
    assert result.agent == "t" and result.model == "gpt-5.4-mini"


def test_system_prompt_goes_first_and_history_accumulates():
    client = ScriptedClient(["раз", "два"])
    agent = agent_with(client, system_prompt="Ты краток.")
    agent.ask("первый")
    agent.ask("второй")

    assert client.calls[0][0] == {"role": "system", "content": "Ты краток."}
    assert [m["role"] for m in client.calls[1]] == ["system", "user", "assistant", "user"]
    assert len(agent.conversation) == 4
    assert agent.conversation.exchanges == 2


def test_history_can_be_switched_off():
    client = ScriptedClient(["раз", "два"])
    agent = agent_with(client, history=HistoryConfig(enabled=False))
    agent.ask("первый")
    agent.ask("второй")
    assert [m["role"] for m in client.calls[1]] == ["system", "user"]
    assert len(agent.conversation) == 0


def test_input_policy_runs_before_the_provider():
    client = ScriptedClient("не должно случиться")
    agent = agent_with(client, input=InputPolicy(max_chars=5))
    with pytest.raises(InputRejected):
        agent.ask("слишком длинный запрос")
    assert client.calls == []          # денег не потрачено
    assert len(agent.conversation) == 0  # история цела


def test_template_from_input_policy_reaches_the_provider():
    client = ScriptedClient("ok")
    agent = agent_with(client, input=InputPolicy(template="Кратко: {input}"))
    agent.ask("почему небо синее")
    assert client.calls[0][-1]["content"] == "Кратко: почему небо синее"


def test_failed_request_leaves_history_intact():
    agent = agent_with(FailingClient(LLMError("провайдер лёг")))
    with pytest.raises(LLMError):
        agent.ask("вопрос")
    assert len(agent.conversation) == 0


# --------------------------------------------------------------------------
# Выходная политика и переспрос
# --------------------------------------------------------------------------

def test_bad_form_is_reasked_and_fixed():
    client = ScriptedClient([BAD, GOOD])
    agent = agent_with(client, output=OutputPolicy(format="json", schema=SCHEMA,
                                                   max_attempts=3))
    result = agent.ask("список")
    assert result.ok and result.attempts == 2
    assert result.data == {"items": [{"name": "Python"}]}
    # Замечание ушло модели отдельным сообщением, а не подмешалось в вопрос.
    assert "не подошёл" in client.calls[1][-1]["content"]


def test_repair_does_not_pollute_the_dialogue():
    """Негодный ответ и замечание к нему в историю попадать не должны."""
    agent = agent_with(ScriptedClient([BAD, GOOD]),
                       output=OutputPolicy(format="json", schema=SCHEMA, max_attempts=3))
    agent.ask("список")
    contents = [m.content for m in agent.conversation.messages]
    assert contents == ["список", GOOD]


def test_repair_attempts_are_capped_and_counted():
    client = ScriptedClient([BAD])
    agent = agent_with(client, output=OutputPolicy(format="json", schema=SCHEMA,
                                                   max_attempts=3))
    result = agent.ask("список")
    assert not result.ok and result.attempts == 3
    assert len(client.calls) == 3
    # Переспросы стоят денег и обязаны быть видны в счёте отдельной строкой.
    assert agent.usage.by_kind["переспрос"].requests == 2


def test_require_valid_turns_bad_form_into_an_error():
    agent = agent_with(ScriptedClient([BAD]),
                       output=OutputPolicy(format="json", schema=SCHEMA,
                                           max_attempts=1, require_valid=True))
    with pytest.raises(OutputRejected):
        agent.ask("список")
    assert len(agent.conversation) == 0


# --------------------------------------------------------------------------
# Судья
# --------------------------------------------------------------------------

def test_judge_scores_the_answer_and_costs_tokens():
    client = ScriptedClient(["длинный ответ", "полнота: 5\nясность: 4\nобоснованность: 3"])
    agent = agent_with(client, judge=JudgeConfig())
    result = agent.ask("вопрос")
    assert result.scores == {"полнота": 5, "ясность": 4, "обоснованность": 3}
    assert result.mean_score == 4
    assert agent.usage.by_kind["оценка"].requests == 1


def test_judge_sees_the_question_not_the_repair_note():
    client = ScriptedClient([BAD, GOOD, "полнота: 5\nясность: 5\nобоснованность: 5"])
    agent = agent_with(client, output=OutputPolicy(format="json", schema=SCHEMA,
                                                   max_attempts=2),
                       judge=JudgeConfig())
    agent.ask("исходный вопрос")
    assert "исходный вопрос" in client.calls[-1][-1]["content"]
    assert "не подошёл" not in client.calls[-1][-1]["content"]


def test_broken_judge_does_not_break_the_answer():
    client = ScriptedClient(["ответ", "я не понял задание"])
    agent = agent_with(client, judge=JudgeConfig())
    result = agent.ask("вопрос")
    assert result.ok and result.text == "ответ"
    assert result.judge_note == "оценщик ответил не по форме"


def test_low_score_is_reported_against_the_threshold():
    client = ScriptedClient(["ответ", "полнота: 2\nясность: 2\nобоснованность: 2"])
    agent = agent_with(client, judge=JudgeConfig(min_mean=3.5))
    assert "ниже порога" in agent.ask("вопрос").judge_note


# --------------------------------------------------------------------------
# Бюджет
# --------------------------------------------------------------------------

def test_request_limit_stops_before_spending():
    client = ScriptedClient("ответ")
    agent = agent_with(client, budget=Budget(max_requests=2))
    agent.ask("раз")
    agent.ask("два")
    with pytest.raises(BudgetExceeded):
        agent.ask("три")
    assert len(client.calls) == 2


def test_token_limit_is_checked_before_the_request_not_after():
    agent = agent_with(ScriptedClient("ответ"), budget=Budget(max_tokens=10))
    with pytest.raises(BudgetExceeded):
        agent.ask("вопрос")
    assert agent.usage.requests == 0


# --------------------------------------------------------------------------
# Смена конфига на лету
# --------------------------------------------------------------------------

def test_reconfigure_changes_model_and_keeps_history_and_spending():
    agent = agent_with(ScriptedClient("ответ"))
    agent.ask("вопрос")
    spent, messages = agent.usage.total_tokens, len(agent.conversation)

    agent.reconfigure(model="gpt-5.4")
    assert agent.config.model == "gpt-5.4"
    assert agent.context_limit == 1_050_000
    assert agent.usage.total_tokens == spent
    assert len(agent.conversation) == messages


def test_reconfigure_applies_generation_params_to_next_request():
    client = ScriptedClient(["раз", "два"])
    agent = agent_with(client)
    agent.ask("первый")
    agent.reconfigure(generation=GenerationParams(temperature=0.1, max_tokens=99))
    agent.ask("второй")
    assert client.kwargs[1]["temperature"] == 0.1
    assert client.kwargs[1]["max_tokens"] == 99


def test_reconfigure_rejects_unknown_model_and_keeps_the_old_one():
    agent = agent_with(ScriptedClient("ответ"))
    with pytest.raises(Exception):
        agent.reconfigure(model="нетакой")
    assert agent.config.model == "gpt-5.4-mini"


def test_retry_asks_the_same_question_without_previous_answers():
    client = ScriptedClient(["первый ответ", "второй ответ"])
    agent = agent_with(client)
    agent.ask("вопрос")
    agent.retry_last()
    # Во втором запросе прежнего ответа быть не должно: задача та же самая.
    assert [m["role"] for m in client.calls[1]] == ["system", "user"]
    assert len(agent.conversation) == 3   # в переписке видно оба ответа


def test_reset_clears_dialogue_but_not_spending():
    agent = agent_with(ScriptedClient("ответ"))
    agent.ask("вопрос")
    agent.reset()
    assert len(agent.conversation) == 0
    assert agent.usage.total_tokens > 0   # деньги уже потрачены


# --------------------------------------------------------------------------
# Окно контекста
# --------------------------------------------------------------------------

def test_context_budget_shrinks_as_dialogue_grows():
    agent = agent_with(ScriptedClient("ответ"))
    free_before = agent.free_tokens()
    agent.ask("вопрос")
    assert agent.free_tokens() < free_before
    assert 0 < agent.fill_ratio() < 1
    assert not agent.is_full()


def test_smaller_max_tokens_frees_room_for_history():
    agent = agent_with(ScriptedClient("ответ"))
    wide = agent.input_budget
    agent.reconfigure(generation=GenerationParams(max_tokens=100))
    assert agent.input_budget > wide
