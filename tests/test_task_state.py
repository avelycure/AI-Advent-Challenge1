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
from llmagent.task_state import split_task_directive

from conftest import ScriptedClient
from llmchat.app import handle_task, pause_task, resume_task
from llmchat.session import Session

FAST = Transport(demo=True, demo_delay=0.0)


def answer_with_state(text, stage, step, expected, paused=False):
    return (
        text + "\n\n[[TASK_STATE "
        + '{{"stage":"{}","current_step":"{}","expected_action":"{}",'
          '"paused":{}}}'.format(
              stage, step, expected, str(paused).lower())
        + "]]"
    )


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


def test_a_plain_user_task_starts_and_advances_the_machine_automatically():
    client = ScriptedClient([
        answer_with_state("План готов.", EXECUTION, "Реализовать калькулятор",
                          "Запустить тесты"),
        answer_with_state("Код готов.", VALIDATION, "Проверить операции",
                          "Подтвердить результат"),
    ])
    session = Session(Agent(AgentConfig(transport=FAST), client=client))

    first = session.agent.ask(
        "Создай калькулятор с четырьмя операциями и обработкой деления на ноль")
    assert first.text == "План готов."
    assert session.task_state.stage == EXECUTION
    assert session.task_state.current_step == "Реализовать калькулятор"
    assert "TASK_STATE" not in first.text

    second = session.agent.ask("Продолжай")
    assert second.text == "Код готов."
    assert session.task_state.stage == VALIDATION
    assert session.task_state.expected_action == "Подтвердить результат"


def test_a_model_result_can_finish_several_stages_but_the_machine_walks_forward():
    client = ScriptedClient(answer_with_state(
        "Решение создано и проверяется.", VALIDATION, "Проверить", "Завершить"))
    session = Session(Agent(AgentConfig(transport=FAST), client=client))
    session.agent.ask("Создай калькулятор")
    assert session.task_state.stage == VALIDATION


def test_the_model_can_pause_on_any_current_stage():
    client = ScriptedClient(answer_with_state(
        "Нужно выбрать формат ввода.", PLANNING,
        "Уточнить формат ввода", "Получить решение пользователя", paused=True))
    session = Session(Agent(AgentConfig(transport=FAST), client=client))
    session.agent.ask("Создай калькулятор")
    assert session.task_state.stage == PLANNING
    assert session.task_state.paused
    assert session.task_state.expected_action == "Получить решение пользователя"


def test_task_directive_parser_leaves_an_invalid_tail_visible():
    visible, state = split_task_directive("Ответ\n[[TASK_STATE не-json]]")
    assert visible.endswith("[[TASK_STATE не-json]]")
    assert state is None


def test_automatic_task_resumes_from_disk_without_repeating_requirements(tmp_path):
    first_client = ScriptedClient(answer_with_state(
        "План готов.", EXECUTION, "Реализовать калькулятор", "Запустить тесты"))
    session = Session(Agent(AgentConfig(transport=FAST), client=first_client))
    requirement = "Создай калькулятор с четырьмя операциями и делением на ноль"
    session.agent.ask(requirement)
    session.agent.pause_task()

    store = SessionStore(str(tmp_path))
    saved = store.save(session.agent, title="калькулятор")
    continued_client = ScriptedClient(answer_with_state(
        "Реализация готова.", VALIDATION, "Проверить операции", "Завершить"))
    returned = restore_agent(
        store.load(saved.session_id), session.agent.config, client=continued_client)
    continued = Session(returned)
    continued.agent.resume_task()
    continued.agent.ask("Продолжай")

    sent = "\n".join(message["content"] for message in continued_client.calls[0])
    assert requirement in sent
    assert "Этап: execution" in sent
    assert "Текущий шаг: Реализовать калькулятор" in sent
    assert continued.task_state.stage == VALIDATION


def test_state_reviewer_moves_a_finished_solution_out_of_execution_and_to_done():
    client = ScriptedClient([
        answer_with_state("План готов.", EXECUTION,
                          "Реализовать калькулятор", "Создать решение"),
        answer_with_state("Готово. Калькулятор полностью реализован.", EXECUTION,
                          "Ждать улучшений", "Пользователь должен попросить ещё"),
        ('{"stage":"validation","current_step":"Проверить калькулятор",'
         '"expected_action":"Запустить проверки","paused":false}'),
        answer_with_state("Все операции и деление на ноль проверены.", VALIDATION,
                          "Проверка завершена", "Подтвердить результат"),
        ('{"stage":"done","current_step":"Калькулятор готов и проверен",'
         '"expected_action":"Действий не требуется","paused":false}'),
    ])
    session = Session(Agent(AgentConfig(transport=FAST), client=client))

    session.agent.ask("Создай калькулятор")
    session.agent.ask("Заканчивай решение")
    assert session.task_state.stage == VALIDATION
    assert session.task_state.current_step == "Проверить калькулятор"

    session.agent.ask("Продолжай")
    assert session.task_state.stage == DONE
    assert session.task_state.current_step == "Калькулятор готов и проверен"
    assert len(client.calls) == 5
