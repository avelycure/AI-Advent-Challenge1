"""Ветки разговора: два продолжения от одного места.

Ни один тест не ходит в сеть: отвечает штатная заглушка с нулевой паузой.
"""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    AgentError,
    HistoryConfig,
    SessionStore,
    Transport,
    restore_agent,
)
from llmagent.branches import MAIN


def agent_with(strategy: str = "window", keep_last: int = 6, **history) -> Agent:
    config = AgentConfig(
        name="t", transport=Transport(demo=True, demo_delay=0.0),
        history=HistoryConfig(strategy=strategy, keep_last=keep_last, **history))
    return Agent(config)


def texts(agent: Agent):
    return [message.content for message in agent.conversation.messages]


# --------------------------------------------------------------------------
# Пока веток не заводили
# --------------------------------------------------------------------------

def test_a_plain_dialog_has_one_branch_and_does_not_notice_it():
    agent = agent_with()
    agent.ask("Собираем ТЗ")
    assert agent.branches.active == MAIN
    assert not agent.branches.branched
    assert len(agent.branches.all()) == 1


# --------------------------------------------------------------------------
# Точки возврата
# --------------------------------------------------------------------------

def test_a_checkpoint_remembers_the_conversation_not_a_number():
    """Номер сообщения не переживает обрезки, а снимок переживает."""
    agent = agent_with()
    agent.ask("Платформа Android")
    point = agent.mark_checkpoint("выбор")
    agent.ask("Ещё вопрос")
    assert point.messages == 2
    assert len(point.conversation.messages) == 2
    assert point.conversation.messages[0].content == "Платформа Android"


def test_a_checkpoint_without_a_name_gets_one():
    agent = agent_with()
    agent.ask("Вопрос")
    assert agent.mark_checkpoint().label == "точка-1"


def test_a_repeated_checkpoint_name_is_refused():
    agent = agent_with()
    agent.ask("Вопрос")
    agent.mark_checkpoint("выбор")
    with pytest.raises(AgentError):
        agent.mark_checkpoint("выбор")


# --------------------------------------------------------------------------
# Ветки
# --------------------------------------------------------------------------

def test_two_branches_from_one_place_do_not_see_each_other():
    """Ради этого ветвление и делается."""
    agent = agent_with()
    agent.ask("Платформа Android, выбираем стек")
    agent.mark_checkpoint("выбор")

    agent.fork("вариант-а")
    agent.ask("Берём Kotlin и Ktor")
    in_a = texts(agent)

    agent.fork("вариант-б", "выбор")
    agent.ask("Берём Flutter и Dart")
    in_b = texts(agent)

    assert "Берём Kotlin и Ktor" in in_a and "Берём Flutter и Dart" not in in_a
    assert "Берём Flutter и Dart" in in_b and "Берём Kotlin и Ktor" not in in_b
    assert in_a[0] == in_b[0] == "Платформа Android, выбираем стек"


def test_a_branch_without_a_checkpoint_starts_from_here():
    agent = agent_with()
    agent.ask("Первый вопрос")
    branch = agent.fork("проба")
    assert branch.inherited == 2
    # Точка под неё поставлена сама: иначе от того же места не отвести вторую.
    assert agent.branches.checkpoints()


def test_switching_returns_the_other_conversation():
    agent = agent_with()
    agent.ask("Общее начало")
    agent.mark_checkpoint("развилка")
    agent.fork("а")
    agent.ask("Только в ветке а")
    agent.fork("б", "развилка")
    agent.ask("Только в ветке б")

    agent.switch_branch("а")
    assert "Только в ветке а" in texts(agent)
    assert "Только в ветке б" not in texts(agent)


def test_a_branch_is_reachable_by_number():
    agent = agent_with()
    agent.ask("Начало")
    agent.fork("проба")
    assert agent.switch_branch("1").name == MAIN


def test_a_repeated_branch_name_is_refused():
    agent = agent_with()
    agent.ask("Начало")
    agent.fork("проба")
    with pytest.raises(AgentError):
        agent.fork("проба")


def test_an_unknown_branch_is_refused_with_the_list():
    agent = agent_with()
    agent.ask("Начало")
    with pytest.raises(AgentError) as exc:
        agent.switch_branch("которой-нет")
    assert MAIN in str(exc.value)


def test_an_unknown_checkpoint_is_refused():
    agent = agent_with()
    agent.ask("Начало")
    with pytest.raises(AgentError):
        agent.fork("ветка", "точки-такой-нет")


# --------------------------------------------------------------------------
# Ветки и остальное хозяйство агента
# --------------------------------------------------------------------------

def test_every_branch_gets_the_current_strategy():
    agent = agent_with(strategy="window", keep_last=4)
    agent.ask("Начало")
    agent.fork("проба")
    agent.reconfigure(history=HistoryConfig(strategy="summary", keep_last=8))
    for branch in agent.branches.all():
        assert branch.conversation.strategy == "summary"
        assert branch.conversation.keep_last == 8


def test_spending_is_counted_once_for_all_branches():
    """Деньги общие: два продолжения оплачивал один кошелёк."""
    agent = agent_with()
    agent.ask("Начало")
    agent.mark_checkpoint("развилка")
    agent.fork("а")
    agent.ask("Вопрос в ветке а")
    agent.fork("б", "развилка")
    agent.ask("Вопрос в ветке б")
    assert agent.usage.requests == 3


def test_a_note_about_a_failure_does_not_follow_into_another_branch():
    """Заметка описывает ту переписку, из которой ушли, а не эту."""
    agent = agent_with()
    agent.ask("Начало")
    agent.facts_note = "картотека не разобрана"
    agent.fork("проба")
    assert agent.facts_note == ""


def test_a_new_dialog_leaves_one_branch():
    agent = agent_with()
    agent.ask("Начало")
    agent.fork("проба")
    agent.new_session()
    assert [branch.name for branch in agent.branches.all()] == [MAIN]
    assert not agent.branches.checkpoints()
    assert not agent.conversation.messages


# --------------------------------------------------------------------------
# Сессия
# --------------------------------------------------------------------------

def test_branches_survive_the_session_file(tmp_path):
    agent = agent_with()
    agent.ask("Общее начало")
    agent.mark_checkpoint("развилка")
    agent.fork("а")
    agent.ask("Только в ветке а")
    agent.fork("б", "развилка")
    agent.ask("Только в ветке б")

    store = SessionStore(str(tmp_path))
    saved = store.save(agent, title="ветвление")
    returned = restore_agent(store.load(saved.session_id), agent.config)

    assert [branch.name for branch in returned.branches.all()] == [MAIN, "а", "б"]
    assert returned.branches.active == "б"
    assert "Только в ветке б" in texts(returned)
    returned.switch_branch("а")
    assert "Только в ветке а" in texts(returned)
    assert [point.label for point in returned.branches.checkpoints()] == ["развилка"]


def test_a_session_from_a_previous_version_has_a_single_branch(tmp_path):
    agent = agent_with()
    agent.ask("Разговор без ветвей")
    store = SessionStore(str(tmp_path))
    saved = store.save(agent, title="старая")
    record = store.load(saved.session_id)
    record.branches = {}
    returned = restore_agent(record, agent.config)
    assert [branch.name for branch in returned.branches.all()] == [MAIN]
    assert texts(returned) == texts(agent)
