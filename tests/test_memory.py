"""Явная модель памяти: три слоя, разные сроки жизни и разные хранилища.

Тесты намеренно не ходят в сеть. Состав запроса проверяется через
``ScriptedClient``, который сохраняет каждое отправленное модели сообщение.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from llmagent import (
    LONG_TERM,
    PROFILE,
    RULES,
    SHORT_TERM,
    WORKING,
    Agent,
    AgentConfig,
    AgentError,
    HistoryConfig,
    LongTermMemory,
    MemoryStore,
    SessionStore,
    Transport,
    WorkingMemory,
    restore_agent,
)
from llmagent.memory import PROFILE_PREFIX, PROJECT_PREFIX, WORKING_PREFIX
from llmagent.branches import MAIN
from llmagent.transport import Completion, count_message_tokens, count_text_tokens
from llmchat.app import handle_memory
from llmchat.session import Session
from llmchat import subagent

from conftest import ScriptedClient


FAST_DEMO = Transport(demo=True, demo_delay=0.0)


class MemoryAwareClient:
    """Детерминированная модель: ответ меняется только от блоков памяти."""

    def complete(self, model_ref, messages, max_tokens, **kwargs):
        prompt = "\n".join(message["content"] for message in messages)
        known = []
        if "обращение: Иван" in prompt:
            known.append("Иван")
        if "цель: подготовить релиз" in prompt:
            known.append("релиз")
        text = ", ".join(known) if known else "память пуста"
        return Completion(text, count_message_tokens(messages), count_text_tokens(text),
                          elapsed=0.001)


def make_agent(client=None, long_term=None) -> Agent:
    return Agent(
        AgentConfig(name="memory-test", system_prompt="Системная инструкция.",
                    transport=FAST_DEMO),
        client=client or ScriptedClient("ответ"),
        long_term_memory=long_term,
    )


def test_information_reaches_only_the_explicitly_selected_layer():
    """Обычная реплика — short; working и long заполняются только явно."""
    agent = make_agent()

    agent.ask("Запомни: кодовое слово апельсин")
    assert [message.content for message in agent.conversation.messages
            if message.role == "user"] == ["Запомни: кодовое слово апельсин"]
    assert agent.working_memory.items == {}
    assert agent.long_term_memory.profile == {}

    assert agent.remember(WORKING, "цель", "реализовать импорт")
    assert agent.remember(LONG_TERM, "обращение", "Иван", section=PROFILE)
    assert agent.working_memory.items == {"цель": "реализовать импорт"}
    assert agent.long_term_memory.profile == {"обращение": "Иван"}
    # Явные слои не превращаются в реплики текущего диалога.
    assert all("реализовать импорт" not in message.content
               for message in agent.conversation.messages)

    with pytest.raises(AgentError, match="краткосрочная"):
        agent.remember(SHORT_TERM, "код", "47")
    with pytest.raises(AgentError, match="неизвестный слой"):
        agent.remember("случайный", "код", "47")


def test_long_and_working_memory_reach_the_prompt_before_short_term():
    """Модель получает long -> working -> текущий диалог -> новый вопрос."""
    client = ScriptedClient(["первый ответ", "второй ответ"])
    memory = LongTermMemory()
    memory.remember(PROFILE, "обращение", "Иван")
    memory.remember(RULES, "push", "только по явной команде")
    agent = make_agent(client, memory)
    agent.remember(WORKING, "задача", "реализовать импорт")

    agent.ask("Сначала обсудим интерфейс")
    agent.ask("Что учитывать сейчас?")
    sent = client.calls[-1]
    contents = [message["content"] for message in sent]

    assert [message["role"] for message in sent[:4]] == [
        "system", "user", "user", "user"]
    assert contents[0] == "Системная инструкция."
    assert contents[1].startswith(PROFILE_PREFIX)
    assert "обращение: Иван" in contents[1]
    assert contents[2].startswith(PROJECT_PREFIX)
    assert "push: только по явной команде" in contents[2]
    assert contents[3].startswith(WORKING_PREFIX)
    assert "задача: реализовать импорт" in contents[3]
    assert contents.index("Сначала обсудим интерфейс") > 3
    assert contents[-1] == "Что учитывать сейчас?"


def test_memory_layers_change_the_assistant_answer():
    """Проверяется не только payload: те же слои меняют наблюдаемый ответ."""
    agent = make_agent(MemoryAwareClient())
    assert agent.ask("Что ты помнишь?").text == "память пуста"

    agent.remember(LONG_TERM, "обращение", "Иван", section=PROFILE)
    agent.remember(WORKING, "цель", "подготовить релиз")

    assert agent.ask("А теперь?").text == "Иван, релиз"


def test_new_session_forgets_short_and_working_but_keeps_long_term():
    client = ScriptedClient(["до", "после"])
    memory = LongTermMemory(profile={"обращение": "Иван"})
    agent = make_agent(client, memory)
    agent.remember(WORKING, "задача", "реализовать импорт")
    agent.ask("Кодовое слово апельсин")
    previous_id = agent.session_id

    agent.new_session()

    assert agent.session_id != previous_id
    assert agent.conversation.messages == []
    assert agent.working_memory.items == {}
    assert agent.long_term_memory.profile == {"обращение": "Иван"}

    agent.ask("Что осталось в памяти?")
    sent = "\n".join(message["content"] for message in client.calls[-1])
    assert "обращение: Иван" in sent
    assert "реализовать импорт" not in sent
    assert "апельсин" not in sent


def test_working_memory_is_copied_at_a_checkpoint_and_then_branch_local():
    agent = make_agent()
    agent.remember(WORKING, "платформа", "Android")
    agent.mark_checkpoint("выбор-стека")

    agent.fork("нативная", "выбор-стека")
    agent.remember(WORKING, "стек", "Kotlin")

    agent.fork("кросс", "выбор-стека")
    assert agent.working_memory.items == {"платформа": "Android"}
    agent.remember(WORKING, "стек", "Flutter")

    agent.switch_branch("нативная")
    assert agent.working_memory.items == {
        "платформа": "Android",
        "стек": "Kotlin",
    }
    agent.switch_branch("кросс")
    assert agent.working_memory.items == {
        "платформа": "Android",
        "стек": "Flutter",
    }
    agent.switch_branch(MAIN)
    assert agent.working_memory.items == {"платформа": "Android"}


def test_session_persists_working_memory_but_not_long_term(tmp_path):
    memory = LongTermMemory(profile={"обращение": "Иван"})
    agent = make_agent(long_term=memory)
    agent.remember(WORKING, "цель", "подготовить релиз")
    store = SessionStore(str(tmp_path / "sessions"))

    saved = store.save(agent, title="текущая задача")
    payload = json.loads(saved.path.read_text(encoding="utf-8"))
    returned = restore_agent(store.load(saved.session_id), agent.config)

    assert payload["working_memory"] == {"цель": "подготовить релиз"}
    assert "long_term_memory" not in payload
    assert returned.working_memory.items == {"цель": "подготовить релиз"}
    # Long-term загружается своим MemoryStore, а не возвращается из session-файла.
    assert returned.long_term_memory.profile == {}


def test_session_restores_different_working_memory_in_each_branch(tmp_path):
    agent = make_agent()
    agent.remember(WORKING, "вариант", "общий")
    agent.mark_checkpoint("развилка")
    agent.fork("а", "развилка")
    agent.remember(WORKING, "стек", "Kotlin")
    agent.fork("б", "развилка")
    agent.remember(WORKING, "стек", "Flutter")

    store = SessionStore(str(tmp_path / "sessions"))
    saved = store.save(agent, title="ветвящаяся задача")
    returned = restore_agent(store.load(saved.session_id), agent.config)

    assert returned.branches.active == "б"
    assert returned.working_memory.items["стек"] == "Flutter"
    returned.switch_branch("а")
    assert returned.working_memory.items["стек"] == "Kotlin"
    returned.switch_branch(MAIN)
    assert returned.working_memory.items == {"вариант": "общий"}


def test_memory_store_separates_global_profile_from_project_memory(tmp_path):
    root = tmp_path / "memory"
    first_project = tmp_path / "first-project"
    second_project = tmp_path / "second-project"
    first_project.mkdir()
    second_project.mkdir()

    first = MemoryStore(str(root), str(first_project))
    memory = first.load()
    memory.remember(PROFILE, "язык", "русский")
    memory.remember(RULES, "push", "запрещён без команды")
    first.save(memory)

    same_project = MemoryStore(str(root), str(first_project)).load()
    other_project = MemoryStore(str(root), str(second_project)).load()

    assert same_project.profile == {"язык": "русский"}
    assert same_project.rules == {"push": "запрещён без команды"}
    # Профиль общий для пользователя, правила принадлежат конкретному проекту.
    assert other_project.profile == {"язык": "русский"}
    assert other_project.rules == {}
    assert first.profile_path != first.project_path
    assert first.project_path != MemoryStore(
        str(root), str(second_project)).project_path


def test_memory_store_uses_private_file_and_directory_modes(tmp_path):
    store = MemoryStore(str(tmp_path / "memory"), str(tmp_path / "project"))
    memory = LongTermMemory(profile={"обращение": "Иван"},
                            rules={"push": "только по команде"})
    store.save(memory)

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.project_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.profile_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.project_path.stat().st_mode) == 0o600


def test_secrets_are_refused_before_they_reach_a_memory_file(tmp_path):
    agent = make_agent()
    for key in ("api_key", "openai_api_key", "auth-token", "github_token", "private_key",
                "authorization", "пароль"):
        with pytest.raises(AgentError, match="секреты нельзя"):
            agent.remember(WORKING, key, "sk-secret")
    assert agent.working_memory.items == {}
    assert agent.long_term_memory.profile == {}
    for ordinary in ("token_budget", "authorization_flow", "secret_santa",
                     "password_policy"):
        assert agent.remember(WORKING, ordinary, "обычное значение")


def test_memory_command_names_the_destination_and_saves_long_term(tmp_path):
    agent = make_agent()
    session = Session(agent)
    store = MemoryStore(str(tmp_path / "memory"), str(tmp_path / "project"))

    handle_memory(session, "working срок=1 марта", store)
    handle_memory(session, "long profile язык=русский", store)
    handle_memory(session, "long rules push=только по команде", store)

    assert agent.working_memory.items == {"срок": "1 марта"}
    restored = store.load()
    assert restored.profile == {"язык": "русский"}
    assert restored.rules == {"push": "только по команде"}


def test_stale_full_snapshot_is_refused_instead_of_erasing_newer_memory(tmp_path):
    store = MemoryStore(str(tmp_path / "memory"), str(tmp_path / "project"))
    first = store.load()
    stale = store.load()
    first.remember(PROFILE, "язык", "русский")
    store.save(first)

    stale.remember(RULES, "push", "только по команде")
    with pytest.raises(AgentError, match="другом процессе"):
        store.save(stale)

    assert store.load().profile == {"язык": "русский"}


def test_failed_long_term_write_does_not_change_live_answers():
    class BrokenStore:
        def remember(self, section, key, value):
            raise OSError("диск недоступен")

    agent = make_agent()
    session = Session(agent)
    handle_memory(session, "long profile обращение=Иван", BrokenStore())

    assert agent.long_term_memory.profile == {}


def test_loading_empty_long_term_memory_is_read_only(tmp_path):
    root = tmp_path / "memory"
    memory = MemoryStore(str(root), str(tmp_path / "project")).load()
    assert memory.profile == {} and memory.rules == {}
    assert not root.exists()


def test_restored_or_directly_mutated_secrets_never_reach_a_prompt(tmp_path):
    working = WorkingMemory.restore({"github_token": "SECRET", "цель": "релиз"})
    assert working.items == {"цель": "релиз"}
    assert "SECRET" not in working.prompt_message()["content"]

    store = MemoryStore(str(tmp_path / "memory"), str(tmp_path / "project"))
    store.profile_path.parent.mkdir(parents=True)
    store.profile_path.write_text(json.dumps({
        "items": {"openai_api_key": "SECRET", "язык": "русский"},
    }), encoding="utf-8")
    memory = store.load()
    memory.profile["github_token"] = "ANOTHER_SECRET"  # обход публичного API
    prompt = "\n".join(item["content"] for item in memory.prompt_messages())
    assert "SECRET" not in prompt
    assert "язык: русский" in prompt


def test_no_history_interactive_path_sends_no_memory_or_previous_questions():
    client = ScriptedClient(["первый", "второй"])
    config = AgentConfig(name="memory-test", system_prompt="Система.",
                         history=HistoryConfig(enabled=False), transport=FAST_DEMO)
    agent = Agent(config, client=client,
                  long_term_memory=LongTermMemory(profile={"обращение": "Иван"}))
    agent.remember(WORKING, "цель", "релиз")

    agent.conversation.add_user("первый вопрос")
    agent.answer_pending()
    agent.conversation.add_user("второй вопрос")
    agent.answer_pending()

    assert [message["content"] for message in client.calls[0]] == [
        "Система.", "первый вопрос"]
    assert [message["content"] for message in client.calls[1]] == [
        "Система.", "второй вопрос"]
    breakdown = agent.breakdown()
    assert breakdown.long_term == 0 and breakdown.working == 0


def test_subagent_inherits_the_calling_project_directory(tmp_path, monkeypatch):
    project = tmp_path / "calling-project"
    project.mkdir()
    (project / ".git").mkdir()
    monkeypatch.chdir(project)
    seen = {}

    class Finished:
        stdout = '{"ok": true, "text": "готово"}'
        stderr = ""

    def fake_run(command, **kwargs):
        seen["cwd"] = os.getcwd()
        seen["overridden"] = "cwd" in kwargs
        return Finished()

    monkeypatch.setattr(subagent.subprocess, "run", fake_run)
    parent = AgentConfig(transport=FAST_DEMO)
    result = subagent.run("frugal", "вопрос", parent)

    assert result.ok
    assert seen == {"cwd": str(project), "overridden": False}


def test_breakdown_reports_explicit_memory_layers_separately():
    memory = LongTermMemory(profile={"обращение": "Иван"},
                            rules={"push": "только по команде"})
    agent = make_agent(long_term=memory)
    agent.remember(WORKING, "цель", "реализовать импорт")
    agent.conversation.add_user("Как продолжить?")

    breakdown = agent.breakdown()
    parts = {part.title: part.tokens for part in breakdown.parts()}

    assert breakdown.long_term > 0
    assert breakdown.working > 0
    assert breakdown.pending > 0
    assert parts["Долговременная память"] == breakdown.long_term
    assert parts["Рабочая память"] == breakdown.working
    assert breakdown.estimated_input == sum((
        breakdown.system,
        breakdown.long_term,
        breakdown.working,
        breakdown.summary,
        breakdown.facts,
        breakdown.history,
        breakdown.pending,
        breakdown.overhead,
    ))
