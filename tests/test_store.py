"""Сессии на диске: сохранение, возврат, безопасность файлов, удаление.

Каталог сессий в каждом тесте свой временный — настоящий ``~/.llm-agent``
трогать нельзя ни при каких условиях.
"""
from __future__ import annotations

import json
import os
import pathlib
import stat

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    AgentError,
    SessionNotFound,
    SessionStore,
    Transport,
    restore_agent,
)

from conftest import ScriptedClient

FAST_DEMO = Transport(demo=True, demo_delay=0.0)


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(str(tmp_path / "sessions"))


def talked(config: AgentConfig = None, replies=("ответ",)) -> Agent:
    agent = Agent(config or AgentConfig(name="проба", transport=FAST_DEMO),
                  client=ScriptedClient(list(replies)))
    agent.ask("Привет, меня зовут Влад")
    return agent


# --------------------------------------------------------------------------
# Круг «сохранить — вернуть»
# --------------------------------------------------------------------------

def test_memory_and_spending_come_back(store):
    agent = talked(replies=("Приятно познакомиться!", "Тебя зовут Влад."))
    agent.ask("Как меня зовут?")
    store.save(agent, topic="Знакомство")

    record = store.load(agent.session_id)
    returned = restore_agent(record, agent.config)

    assert returned.session_id == agent.session_id
    assert [m.content for m in returned.conversation.messages] == \
           [m.content for m in agent.conversation.messages]
    assert any("Влад" in m.content for m in returned.conversation.messages)
    assert returned.usage.total_tokens == agent.usage.total_tokens
    assert returned.usage.costs == agent.usage.costs
    assert returned.usage.requests == agent.usage.requests


def test_returned_session_keeps_answering_with_context(store):
    agent = talked(replies=("Приятно познакомиться!",))
    store.save(agent)

    record = store.load(agent.session_id)
    returned = restore_agent(record, agent.config)
    client = ScriptedClient(["Тебя зовут Влад."])
    returned._client = client
    returned.ask("Как меня зовут?")

    # В запросе должна быть вся прежняя переписка, иначе возврат бессмысленен.
    sent = " ".join(m["content"] for m in client.calls[0])
    assert "Влад" in sent


def test_record_carries_what_the_list_shows(store):
    agent = talked()
    store.save(agent, topic="Знакомство")
    record = store.load(agent.session_id)
    assert record.topic == "Знакомство"
    assert record.agent == "проба"
    assert record.model == "gpt-5.4-mini"
    assert record.exchanges == 1
    assert record.total_tokens == agent.usage.total_tokens
    assert record.created_at > 0 and record.updated_at >= record.created_at


def test_saving_twice_keeps_the_creation_time(store):
    agent = talked()
    store.save(agent)
    created = store.load(agent.session_id).created_at
    agent.ask("ещё вопрос")
    store.save(agent)
    again = store.load(agent.session_id)
    assert again.created_at == created
    assert again.updated_at >= created


def test_empty_session_is_not_saved(store):
    agent = Agent(AgentConfig(transport=FAST_DEMO), client=ScriptedClient("ответ"))
    store.save(agent)
    assert store.recent() == []


# --------------------------------------------------------------------------
# Безопасность файла
# --------------------------------------------------------------------------

def test_the_key_never_reaches_the_disk(store):
    agent = talked(AgentConfig(name="проба", api_key="sk-очень-секретный",
                               transport=FAST_DEMO))
    raw = store.save(agent).path.read_text(encoding="utf-8")
    assert "очень-секретный" not in raw
    # Именно null, а не звёздочки: строка «***» при возврате ушла бы
    # провайдеру как настоящий ключ.
    assert json.loads(raw)["config"]["api_key"] is None
    assert "***" not in raw


def test_judge_key_never_reaches_the_disk(store):
    from dataclasses import replace

    config = AgentConfig.from_file("configs/reviewer.yaml").with_changes(
        transport=FAST_DEMO)
    config = config.with_changes(judge=replace(
        config.judge, agent=config.judge.agent.with_changes(api_key="sk-судейский")))
    agent = talked(config, replies=("ответ", "полнота: 5\nясность: 5\nобоснованность: 5"))
    raw = store.save(agent).path.read_text(encoding="utf-8")
    assert "судейский" not in raw


def test_files_are_readable_only_by_their_owner(store):
    path = store.save(talked()).path
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, oct(mode)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_no_temporary_leftovers(store):
    store.save(talked())
    assert list(store.root.glob("*.tmp")) == []


def test_restored_config_has_no_key_and_looks_for_it_again(store):
    agent = talked(AgentConfig(name="проба", api_key="sk-secret", transport=FAST_DEMO))
    store.save(agent)
    assert store.load(agent.session_id).to_config().api_key is None


# --------------------------------------------------------------------------
# Список, поиск, удаление
# --------------------------------------------------------------------------

def test_recent_goes_from_fresh_to_old(store):
    ids = []
    for number in range(3):
        agent = talked()
        store.save(agent, topic="тема {}".format(number))
        ids.append(agent.session_id)
        os.utime(store.path_for(agent.session_id))
    recent = [record.session_id for record in store.recent()]
    assert set(recent) == set(ids)
    assert store.latest().session_id == recent[0]


def test_latest_on_empty_store_is_nothing(store):
    assert store.latest() is None
    assert store.recent() == []


def test_missing_session_names_the_recent_ones(store):
    agent = talked()
    store.save(agent)
    with pytest.raises(SessionNotFound) as info:
        store.load("нетакой")
    assert agent.session_id in str(info.value)


def test_broken_file_does_not_break_the_listing(store):
    agent = talked()
    store.save(agent)
    (store.root / "мусор.json").write_text("это не json", encoding="utf-8")
    assert [record.session_id for record in store.recent()] == [agent.session_id]


def test_removing(store):
    agent = talked()
    store.save(agent)
    assert store.remove(agent.session_id) is True
    assert store.remove(agent.session_id) is False
    assert store.recent() == []


def test_clearing_removes_everything(store):
    for _ in range(3):
        store.save(talked())
    assert store.clear() == 3
    assert store.recent() == []


def test_identifier_cannot_escape_the_directory(store):
    """Идентификатор приходит из командной строки, и путь наружу недопустим."""
    for bad in ("../../etc/passwd", "/etc/passwd", "..", "a/b"):
        try:
            path = store.path_for(bad)
        except AgentError:
            continue
        assert path.parent == store.root, bad


# --------------------------------------------------------------------------
# Новый диалог
# --------------------------------------------------------------------------

def test_new_dialogue_does_not_overwrite_the_saved_one(store):
    agent = talked()
    first = agent.session_id
    store.save(agent, topic="Знакомство")

    agent.new_session()
    assert agent.session_id != first
    assert len(agent.conversation) == 0
    agent._client = ScriptedClient("другой ответ")
    agent.ask("совсем другой вопрос")
    store.save(agent, topic="Другое")

    kept = store.load(first)
    assert kept.topic == "Знакомство"
    assert any("Влад" in m["content"] for m in kept.messages)
    assert len(store.recent()) == 2


def test_returning_with_another_model_forgets_the_exact_size(store):
    """Точный размер контекста измерен прежним токенизатором и после смены неверен."""
    agent = talked()
    store.save(agent)
    record = store.load(agent.session_id)
    assert record.conversation["exact_upto"] > 0

    same = restore_agent(record, record.to_config())
    other = restore_agent(record, record.to_config().with_changes(model="gpt-5.4"))
    assert same.conversation.exact_upto > 0
    assert other.conversation.exact_upto == 0
