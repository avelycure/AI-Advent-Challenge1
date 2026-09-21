"""Конечный автомат задачи: переходы, пауза и продолжение из сессии."""
from __future__ import annotations

import io

import pytest
from rich.console import Console

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
    ToolOutcome,
    Toolbox,
    ToolPolicy,
    Transport,
    restore_agent,
    spawn_agent_spec,
)
from llmagent.task_state import split_task_directive

from conftest import ScriptedClient, ToolCallingClient
from llmchat.app import answer_notice, handle_task, pause_task, resume_task
from llmchat.session import Session

FAST = Transport(demo=True, demo_delay=0.0)


def answer_with_state(text, stage, step, expected, paused=False,
                      validation_passed=False, validation_evidence=""):
    return (
        text + "\n\n[[TASK_STATE "
        + '{{"stage":"{}","current_step":"{}","expected_action":"{}",'
          '"paused":{},"validation_passed":{},"validation_evidence":"{}"}}'.format(
              stage, step, expected, str(paused).lower(),
              str(validation_passed).lower(), validation_evidence)
        + "]]"
    )


def agent_with(*answers, history=True):
    config = AgentConfig(transport=FAST, history=HistoryConfig(enabled=history))
    return Agent(config, client=ScriptedClient(list(answers) or ["готово"]))


def move_to(agent, stage):
    """Пройти реальные ворота до нужного этапа, не подменяя их в тестах."""
    if agent.task_state.stage == PLANNING and stage != PLANNING:
        agent.approve_task_plan()
        agent.advance_task("Выполнить план", "Завершить реализацию")
    if agent.task_state.stage == EXECUTION and stage in (VALIDATION, DONE):
        agent.advance_task("Проверить результат", "Зафиксировать проверку")
    if agent.task_state.stage == VALIDATION and stage == DONE:
        agent.pass_task_validation("Все обязательные сценарии прошли")
        agent.advance_task("Работа завершена", "Действий не требуется")


def test_state_machine_moves_through_every_stage_in_order():
    agent = agent_with()
    agent.start_task("Собрать требования", "Согласовать план")
    assert agent.task_state.stage == PLANNING

    agent.approve_task_plan()
    agent.advance_task("Написать код", "Запустить тесты")
    assert agent.task_state.stage == EXECUTION
    agent.advance_task("Проверить сценарии", "Подтвердить результат")
    assert agent.task_state.stage == VALIDATION
    agent.pass_task_validation("Тесты прошли")
    agent.advance_task("Работа завершена", "Нет")
    assert agent.task_state.stage == DONE


def test_state_machine_rejects_skips_backwards_moves_and_steps_after_done():
    agent = agent_with()
    agent.start_task()
    with pytest.raises(AgentError, match="пропускать этапы"):
        agent.transition_task(VALIDATION)
    with pytest.raises(AgentError, match="план ещё не утверждён"):
        agent.transition_task(EXECUTION)
    agent.approve_task_plan()
    agent.transition_task(EXECUTION)
    with pytest.raises(AgentError, match="идти назад"):
        agent.transition_task(PLANNING)
    agent.transition_task(VALIDATION)
    with pytest.raises(AgentError, match="проверка результата"):
        agent.transition_task(DONE)
    agent.pass_task_validation("Проверены обязательные сценарии")
    agent.transition_task(DONE)
    with pytest.raises(AgentError, match="завершающий"):
        agent.advance_task()


@pytest.mark.parametrize("stage", [PLANNING, EXECUTION, VALIDATION, DONE])
def test_pause_and_resume_preserve_the_exact_place_at_every_stage(stage):
    agent = agent_with()
    agent.start_task("Не повторять этот шаг", "Продолжить отсюда")
    move_to(agent, stage)

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
    agent.approve_task_plan()
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
    agent.approve_task_plan()
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
    agent.approve_task_plan()
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
    handle_task(session, "approve")
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
        answer_with_state("План готов. Утвердите его.", PLANNING,
                          "Получить утверждение плана", "Дождаться утверждения"),
        answer_with_state("Приступаю по утверждённому плану.", EXECUTION,
                          "Реализовать калькулятор", "Завершить реализацию"),
        answer_with_state("Код готов.", VALIDATION, "Проверить операции",
                          "Подтвердить результат"),
        answer_with_state("Проверки пройдены.", DONE, "Калькулятор готов",
                          "Действий не требуется", validation_passed=True,
                          validation_evidence="Проверены четыре операции и деление на ноль"),
    ])
    session = Session(Agent(AgentConfig(transport=FAST), client=client))

    first = session.agent.ask(
        "Создай калькулятор с четырьмя операциями и обработкой деления на ноль")
    assert first.text == "План готов. Утвердите его."
    assert session.task_state.stage == PLANNING
    assert not session.task_state.plan_approved
    assert "TASK_STATE" not in first.text

    second = session.agent.ask("Утверждаю план. Приступай.")
    assert second.text == "Приступаю по утверждённому плану."
    assert session.task_state.stage == EXECUTION
    assert session.task_state.plan_approved

    session.agent.ask("Продолжай")
    assert session.task_state.stage == VALIDATION
    assert session.task_state.expected_action == "Подтвердить результат"

    session.agent.ask("Проверь результат")
    assert session.task_state.stage == DONE
    assert session.task_state.validation_passed


def test_a_model_cannot_skip_several_stages_and_user_sees_the_reason():
    client = ScriptedClient(answer_with_state(
        "Решение создано и проверяется.", VALIDATION, "Проверить", "Завершить"))
    session = Session(Agent(AgentConfig(transport=FAST), client=client))
    result = session.agent.ask("Создай калькулятор")
    assert session.task_state.stage == PLANNING
    assert result.refused
    assert "пропускать этапы" in result.text
    assert result.task_transition_checks[0]["status"] == "refused"


def test_execution_is_refused_until_the_user_explicitly_approves_the_plan():
    client = ScriptedClient(answer_with_state(
        "Я сразу написал реализацию.", EXECUTION, "Код готов", "Проверить код"))
    session = Session(Agent(AgentConfig(transport=FAST), client=client))

    result = session.agent.ask("Сразу реализуй задачу без согласования")

    assert result.refused
    assert session.task_state.stage == PLANNING
    assert not session.task_state.plan_approved
    assert "план ещё не утверждён" in result.text
    assert "Я сразу написал реализацию" not in result.text

    output = io.StringIO()
    Console(file=output, width=120, force_terminal=False).print(
        answer_notice(session, result))
    assert "Жизненный цикл" in output.getvalue()
    assert "Инварианты" not in output.getvalue()


def test_tools_are_not_offered_before_plan_approval():
    calls = []
    box = Toolbox([spawn_agent_spec(
        ["worker"],
        lambda arguments: calls.append(arguments) or ToolOutcome("выполнено"))])
    client = ToolCallingClient(
        "spawn_agent", '{"config":"worker","question":"реализуй"}', rounds=1)
    config = AgentConfig(
        transport=FAST, tools=ToolPolicy(enabled=("spawn_agent",)))
    session = Session(Agent(config, client=client, toolbox=box))

    session.agent.ask("Реализуй задачу")

    assert client.offered[0] is False
    assert calls == []


def test_negative_plan_phrase_does_not_open_the_execution_gate():
    client = ScriptedClient([
        answer_with_state("План предложен.", PLANNING, "Согласовать план", "Ждать"),
        answer_with_state("Остаюсь на плане.", PLANNING, "Доработать план", "Ждать"),
    ])
    session = Session(Agent(AgentConfig(transport=FAST), client=client))
    session.agent.ask("Подготовь план")

    session.agent.ask("Я не утверждаю план, его нужно исправить")

    assert session.task_state.stage == PLANNING
    assert not session.task_state.plan_approved


def test_done_is_refused_without_successful_validation_evidence():
    client = ScriptedClient(answer_with_state(
        "Финальный результат.", DONE, "Завершено", "Нет действий"))
    session = Session(Agent(AgentConfig(transport=FAST), client=client))
    session.agent.start_task("План", "Утвердить")
    session.agent.approve_task_plan()
    session.agent.transition_task(EXECUTION, "Реализация", "Проверить")
    session.agent.transition_task(VALIDATION, "Проверка", "Зафиксировать результат")

    result = session.agent.ask("Заверши без проверки")

    assert result.refused
    assert session.task_state.stage == VALIDATION
    assert not session.task_state.validation_passed
    assert "успешная проверка" in result.text


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
        "План готов.", PLANNING, "Получить утверждение", "Дождаться пользователя"))
    session = Session(Agent(AgentConfig(transport=FAST), client=first_client))
    requirement = "Создай калькулятор с четырьмя операциями и делением на ноль"
    session.agent.ask(requirement)
    session.agent.approve_task_plan()
    session.agent.transition_task(
        EXECUTION, "Реализовать калькулятор", "Запустить тесты")
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


def test_validation_gate_survives_pause_save_and_resume(tmp_path):
    agent = agent_with()
    agent.start_task("План", "Утвердить")
    move_to(agent, VALIDATION)
    agent.pass_task_validation("Интеграционные тесты прошли")
    agent.pause_task()
    saved = SessionStore(str(tmp_path)).save(agent, title="проверенная задача")

    returned = restore_agent(
        SessionStore(str(tmp_path)).load(saved.session_id), agent.config)
    returned.resume_task()
    returned.advance_task("Готово", "Действий не требуется")

    assert returned.task_state.stage == DONE
    assert returned.task_state.validation_evidence == "Интеграционные тесты прошли"


def test_state_reviewer_moves_a_finished_solution_out_of_execution_and_to_done():
    client = ScriptedClient([
        answer_with_state("План готов.", PLANNING,
                          "Утвердить план", "Получить утверждение"),
        answer_with_state("Начинаю реализацию.", EXECUTION,
                          "Реализовать калькулятор", "Создать решение"),
        answer_with_state("Готово. Калькулятор полностью реализован.", EXECUTION,
                          "Ждать улучшений", "Пользователь должен попросить ещё"),
        ('{"stage":"validation","current_step":"Проверить калькулятор",'
         '"expected_action":"Запустить проверки","paused":false,'
         '"validation_passed":false,"validation_evidence":""}'),
        answer_with_state("Все операции и деление на ноль проверены.", VALIDATION,
                          "Проверка завершена", "Подтвердить результат"),
        ('{"stage":"done","current_step":"Калькулятор готов и проверен",'
         '"expected_action":"Действий не требуется","paused":false,'
         '"validation_passed":true,'
         '"validation_evidence":"Тесты операций и деления на ноль прошли"}'),
    ])
    session = Session(Agent(AgentConfig(transport=FAST), client=client))

    session.agent.ask("Создай калькулятор")
    session.agent.ask("Утверждаю план")
    session.agent.ask("Заканчивай решение")
    assert session.task_state.stage == VALIDATION
    assert session.task_state.current_step == "Проверить калькулятор"

    session.agent.ask("Продолжай")
    assert session.task_state.stage == DONE
    assert session.task_state.current_step == "Калькулятор готов и проверен"
    assert len(client.calls) == 6
