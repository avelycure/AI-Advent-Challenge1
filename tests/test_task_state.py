"""Конечный автомат задачи: переходы, пауза и продолжение из сессии."""
from __future__ import annotations

import pytest

from llmagent import (
    DONE,
    EXECUTION,
    PLANNING,
    VALIDATION,
    Agent,
    AgentConfig,
    AgentError,
    HistoryConfig,
    SessionStore,
    Transport,
    restore_agent,
)

from conftest import ScriptedClient
from llmchat.app import handle_task, pause_task, resume_task
from llmchat.session import Session

FAST = Transport(demo=True, demo_delay=0.0)


def agent_with(*answers, history=True):
    config = AgentConfig(transport=FAST, history=HistoryConfig(enabled=history))
    return Agent(config, client=ScriptedClient(list(answers) or ["готово"]))


def test_state_machine_moves_through_every_stage_in_order():
    agent = agent_with()
    agent.start_task("Собрать требования", "Согласовать план")
    assert agent.task_state.stage == PLANNING

    agent.advance_task("Написать код", "Запустить тесты")
    assert agent.task_state.stage == EXECUTION
    agent.advance_task("Проверить сценарии", "Подтвердить результат")
    assert agent.task_state.stage == VALIDATION
    agent.advance_task("Работа завершена", "Нет")
    assert agent.task_state.stage == DONE


def test_state_machine_rejects_skips_backwards_moves_and_steps_after_done():
    agent = agent_with()
    agent.start_task()
    with pytest.raises(AgentError, match="пропускать этапы"):
        agent.transition_task(VALIDATION)
    agent.transition_task(EXECUTION)
    with pytest.raises(AgentError, match="идти назад"):
        agent.transition_task(PLANNING)
    agent.transition_task(VALIDATION)
    agent.transition_task(DONE)
    with pytest.raises(AgentError, match="завершающий"):
        agent.advance_task()


@pytest.mark.parametrize("stage", [PLANNING, EXECUTION, VALIDATION, DONE])
def test_pause_and_resume_preserve_the_exact_place_at_every_stage(stage):
    agent = agent_with()
    agent.start_task("Не повторять этот шаг", "Продолжить отсюда")
    while agent.task_state.stage != stage:
        agent.advance_task("Не повторять этот шаг", "Продолжить отсюда")

    before = agent.task_state.to_dict()
    agent.pause_task()
    assert agent.task_state.paused
    assert agent.task_state.stage == before["stage"]
    assert agent.task_state.current_step == before["current_step"]
    assert agent.task_state.expected_action == before["expected_action"]

    agent.resume_task()
    assert not agent.task_state.paused
    assert agent.task_state.stage == before["stage"]
    assert agent.task_state.current_step == before["current_step"]
    assert agent.task_state.expected_action == before["expected_action"]


def test_state_is_sent_again_after_resume_without_reexplaining_the_task():
    client = ScriptedClient(["первый ответ", "продолжение"])
    agent = Agent(AgentConfig(transport=FAST), client=client)
    agent.start_task("Реализовать обработчик", "Запустить тесты")
    agent.transition_task(EXECUTION, "Реализовать обработчик", "Запустить тесты")
    agent.ask("Начинай")
    agent.pause_task()
    agent.resume_task()

    agent.ask("Продолжай")
    sent = "\n".join(message["content"] for message in client.calls[-1])
    assert "Этап: execution" in sent
    assert "Текущий шаг: Реализовать обработчик" in sent
    assert "Ожидаемое действие: Запустить тесты" in sent
    assert sent.count("Формализованное состояние текущей задачи") == 1


def test_state_survives_a_saved_pause_before_the_first_message(tmp_path):
    agent = agent_with()
    agent.start_task("Подготовить миграцию", "Проверить схему")
    agent.transition_task(EXECUTION, "Подготовить миграцию", "Проверить схему")
    agent.pause_task()

    store = SessionStore(str(tmp_path))
    saved = store.save(agent, title="миграция")
    assert saved.path.is_file()

    returned = restore_agent(store.load(saved.session_id), agent.config)
    assert returned.task_state.to_dict() == agent.task_state.to_dict()
    returned.resume_task()
    assert returned.task_state.stage == EXECUTION
    assert returned.task_state.current_step == "Подготовить миграцию"


def test_each_branch_keeps_its_own_copy_of_task_state():
    agent = agent_with("ответ")
    agent.start_task("Общий план", "Выбрать вариант")
    agent.ask("Начало")
    agent.mark_checkpoint("выбор")
    agent.fork("вариант-а")
    agent.transition_task(EXECUTION, "Сделать А", "Проверить А")

    agent.fork("вариант-б", "выбор")
    assert agent.task_state.stage == PLANNING
    assert agent.task_state.current_step == "Общий план"
    agent.switch_branch("вариант-а")
    assert agent.task_state.stage == EXECUTION
    assert agent.task_state.current_step == "Сделать А"


def test_task_state_is_kept_when_conversation_history_is_disabled():
    client = ScriptedClient("ответ")
    agent = Agent(
        AgentConfig(transport=FAST, history=HistoryConfig(enabled=False)),
        client=client)
    agent.start_task("Собрать отчёт", "Отдать итог")
    agent.ask("Продолжай")
    sent = "\n".join(message["content"] for message in client.calls[0])
    assert "Текущий шаг: Собрать отчёт" in sent


def test_chat_commands_control_the_same_state_machine():
    session = Session(agent_with())
    handle_task(session, "start Составить план | Согласовать план")
    handle_task(session, "next Написать код | Запустить тесты")
    pause_task(session)
    assert session.task_state.stage == EXECUTION
    assert session.task_state.paused

    resume_task(session)
    assert not session.task_state.paused
    assert session.task_state.current_step == "Написать код"
    assert session.task_state.expected_action == "Запустить тесты"
