"""Персонализация поверх памяти: профиль и правила для типов задач."""
from __future__ import annotations

from llmagent import (
    Agent, AgentConfig, HistoryConfig, LongTermMemory, MemoryStore, Transport,
)
from llmagent.transport import Completion, count_message_tokens, count_text_tokens
from llmchat.app import handle_profile
from llmchat.session import Session

from conftest import ScriptedClient


FAST_DEMO = Transport(demo=True, demo_delay=0.0)


class ProfileAwareClient:
    """Детерминированно отражает профиль, показывая разницу в ответах."""

    def complete(self, model_ref, messages, max_tokens, **kwargs):
        prompt = "\n".join(message["content"] for message in messages)
        if "address: Анна" in prompt and "tone: тепло" in prompt:
            text = "Анна, с радостью помогу!"
        elif "address: Борис" in prompt and "tone: строго" in prompt:
            text = "Борис. Ответ по существу."
        else:
            text = "Профиль не применён"
        return Completion(text, count_message_tokens(messages), count_text_tokens(text),
                          elapsed=0.001)


def _agent(profile, client=None, history=True):
    config = AgentConfig(
        name="profile-test",
        system_prompt="Помогай пользователю.",
        history=HistoryConfig(enabled=history),
        transport=FAST_DEMO,
    )
    return Agent(config, client=client or ScriptedClient("готово"),
                 long_term_memory=LongTermMemory(profile=profile))


def test_different_profiles_produce_different_answers():
    anna = _agent({"identity.address": "Анна", "style.tone": "тепло"},
                  ProfileAwareClient())
    boris = _agent({"identity.address": "Борис", "style.tone": "строго"},
                   ProfileAwareClient())

    assert anna.ask("Помоги составить план").text == "Анна, с радостью помогу!"
    assert boris.ask("Помоги составить план").text == "Борис. Ответ по существу."


def test_global_preferences_are_attached_to_every_request():
    client = ScriptedClient(["раз", "два"])
    agent = _agent({
        "style.tone": "коротко и по делу",
        "format.answer": "Markdown, сначала вывод",
        "constraints.no_push": "не выполнять push без явной команды",
    }, client, history=False)

    agent.ask("Первый вопрос")
    agent.ask("Второй вопрос")

    for call in client.calls:
        prompt = "\n".join(message["content"] for message in call)
        assert "tone: коротко и по делу" in prompt
        assert "answer: Markdown, сначала вывод" in prompt
        assert "no_push: не выполнять push без явной команды" in prompt


def test_task_preferences_are_selected_from_the_current_request_automatically():
    client = ScriptedClient(["фича", "покупки"])
    agent = _agent({
        "style.tone": "лаконично",
        "task.feature.match": "напиши фичу, реализуй фичу",
        "task.feature.instructions": "сначала исследователь, затем два исполнителя",
        "domain.shopping.match": "собери корзину, подбери товары",
        "domain.shopping.instructions": "сравни цену, рейтинг и доставку",
    }, client)

    agent.ask("Напиши фичу импорта")
    agent.ask("Собери корзину для пикника")

    feature_prompt = "\n".join(message["content"] for message in client.calls[0])
    shopping_prompt = "\n".join(message["content"] for message in client.calls[1])
    assert "текущей задачи «feature»" in feature_prompt
    assert "сначала исследователь" in feature_prompt
    assert "сравни цену" not in feature_prompt
    assert "текущей задачи «shopping»" in shopping_prompt
    assert "сравни цену, рейтинг и доставку" in shopping_prompt
    assert "сначала исследователь" not in shopping_prompt
    assert "tone: лаконично" in feature_prompt and "tone: лаконично" in shopping_prompt


def test_profile_command_updates_the_persistent_profile(tmp_path):
    agent = _agent({})
    session = Session(agent)
    store = MemoryStore(str(tmp_path / "memory"), str(tmp_path / "project"))

    handle_profile(session, "identity.address=Иван", store)
    handle_profile(session, "style.tone=коротко", store)
    handle_profile(session, "task.feature.match=напиши фичу, реализуй", store)
    handle_profile(session, "task.feature.instructions=сначала составь план", store)

    restored = store.load().profile
    assert restored["identity.address"] == "Иван"
    assert restored["style.tone"] == "коротко"
    assert restored.active_tasks("Пожалуйста, напиши фичу") == ["feature"]
