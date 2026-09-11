"""Память фактов: ключ-значение вместо прозы пересказа.

Ни один тест не ходит в сеть: картотеку пишет заглушка, которая узнаёт запрос
за фактами по системному промпту и отвечает на него отдельным текстом.
"""
from __future__ import annotations

import json

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    ConfigError,
    HistoryConfig,
    LLMError,
    SessionStore,
    Transport,
    overrides,
    restore_agent,
)
from llmagent import facts as recording
from llmagent.config import FactsConfig
from llmagent.history import FACTS_PREFIX, Message
from llmagent.transport import Completion, count_message_tokens, count_text_tokens
from llmagent.usage import FACTS, MAIN

ANSWER = "Ответ модели на заданный вопрос, довольно подробный и не короткий."
CARD = {"платформа": "Android", "бюджет": "300 тысяч рублей"}


class RecordingClient:
    """Заглушка, отвечающая на просьбу о картотеке иначе, чем на вопрос."""

    def __init__(self, card=None, answer: str = ANSWER) -> None:
        self.card = CARD if card is None else card
        self.answer = answer
        self.calls = []
        self.kwargs = []

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref: str, messages, max_tokens: int, temperature: float = 0.7,
                 top_p=None, stop=None, response_format=None, tools=None) -> Completion:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append({"max_tokens": max_tokens, "temperature": temperature,
                            "stop": stop, "response_format": response_format})
        text = self.reply(messages)
        return Completion(text, count_message_tokens(messages),
                          count_text_tokens(text), elapsed=0.001)

    def reply(self, messages) -> str:
        if not self.asked_for_facts(messages):
            return self.answer
        # Последний заготовленный ответ повторяется: заготовок ровно столько,
        # сколько нужно проверке, а запросов бывает и больше.
        card = (self.card.pop(0) if len(self.card) > 1 else self.card[0]) \
            if isinstance(self.card, list) else self.card
        return card if isinstance(card, str) else json.dumps(card, ensure_ascii=False)

    @staticmethod
    def asked_for_facts(messages) -> bool:
        return bool(messages) and messages[0].get("content") == recording.FACTS_SYSTEM

    @property
    def updates(self) -> int:
        return sum(1 for sent in self.calls if self.asked_for_facts(sent))


class BrokenRecorder(RecordingClient):
    """Картотека не даётся: на просьбу о ней приходит ошибка."""

    def reply(self, messages) -> str:
        if self.asked_for_facts(messages):
            raise LLMError("провайдер недоступен")
        return self.answer


def agent_with(client, keep_last: int = 4, strategy: str = "facts",
               max_facts: int = 20, max_tokens: int = 300, update_every: int = 1,
               **history) -> Agent:
    config = AgentConfig(
        name="t", transport=Transport(demo=True, demo_delay=0.0),
        history=HistoryConfig(
            strategy=strategy, keep_last=keep_last,
            facts=FactsConfig(max_facts=max_facts, max_tokens=max_tokens,
                              update_every=update_every), **history))
    return Agent(config, client=client)


def sent_texts(agent: Agent):
    return [message["content"] for message
            in agent.conversation.api_messages("Системный промпт")]


# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

def test_nonsense_numbers_are_refused():
    with pytest.raises(ConfigError):
        FactsConfig(max_facts=0)
    with pytest.raises(ConfigError):
        FactsConfig(max_tokens=0)
    with pytest.raises(ConfigError):
        FactsConfig(update_every=0)


def test_settings_are_reachable_from_the_command_line():
    config = overrides.apply(AgentConfig(), ["history.strategy=facts",
                                             "history.facts.max_facts=8"])
    assert config.history.collects_facts
    assert config.history.facts.max_facts == 8


# --------------------------------------------------------------------------
# Когда картотека обновляется
# --------------------------------------------------------------------------

def test_facts_are_updated_after_every_user_message():
    client = RecordingClient()
    agent = agent_with(client)
    agent.ask("Собираем ТЗ: платформа Android, бюджет 300 тысяч")
    assert agent.conversation.facts == CARD
    assert client.updates == 1


def test_no_facts_are_collected_under_another_strategy():
    client = RecordingClient()
    agent = agent_with(client, strategy="window")
    agent.ask("Собираем ТЗ: платформа Android")
    assert client.updates == 0
    assert agent.conversation.facts == {}


def test_only_new_messages_go_into_the_request():
    """Картотека затем и нужна, чтобы за память не платить дважды."""
    client = RecordingClient()
    agent = agent_with(client)
    agent.ask("Первый вопрос про сроки")
    agent.ask("Второй вопрос про бюджет")
    second = [sent for sent in client.calls if client.asked_for_facts(sent)][-1]
    assert "Первый вопрос про сроки" not in second[1]["content"]
    assert "Второй вопрос про бюджет" in second[1]["content"]


def test_updates_can_be_made_rarer():
    client = RecordingClient()
    agent = agent_with(client, update_every=2)
    agent.ask("Первый вопрос")
    assert client.updates == 0
    agent.ask("Второй вопрос")
    assert client.updates == 1


def test_the_facts_request_is_counted_separately():
    client = RecordingClient()
    agent = agent_with(client)
    agent.ask("Вопрос про сроки")
    assert agent.usage.by_kind[FACTS].requests == 1
    assert agent.usage.by_kind[MAIN].requests == 1


def test_the_facts_request_asks_for_json_not_for_the_output_format():
    """Выходная политика к картотеке отношения не имеет: от неё нужен JSON."""
    client = RecordingClient()
    agent = agent_with(client)
    agent.ask("Вопрос про сроки")
    asked = [kwargs for sent, kwargs in zip(client.calls, client.kwargs)
             if client.asked_for_facts(sent)][0]
    assert asked["response_format"] == {}
    assert asked["stop"] == []
    assert asked["temperature"] == 0.0


# --------------------------------------------------------------------------
# Что уходит в модель
# --------------------------------------------------------------------------

def test_facts_go_into_the_request_together_with_the_last_messages():
    client = RecordingClient()
    agent = agent_with(client, keep_last=4)
    for number in range(4):
        agent.ask("Вопрос номер {} про требования".format(number))
    sent = sent_texts(agent)
    assert any(text.startswith(FACTS_PREFIX) for text in sent)
    assert "Вопрос номер 0 про требования" not in sent
    assert "Вопрос номер 3 про требования" in sent


def test_the_facts_block_survives_what_the_window_forgets():
    """В этом весь смысл стратегии: окно забыло, а сведения остались."""
    client = RecordingClient()
    agent = agent_with(client, keep_last=2)
    agent.ask("Платформа Android, бюджет 300 тысяч")
    for number in range(3):
        agent.ask("Вопрос номер {}".format(number))
    block = [text for text in sent_texts(agent) if text.startswith(FACTS_PREFIX)][0]
    assert "Android" in block and "300 тысяч" in block


def test_an_empty_card_does_not_send_an_empty_block():
    client = RecordingClient(card={})
    agent = agent_with(client)
    agent.ask("Вопрос ни о чём")
    assert not any(text.startswith(FACTS_PREFIX) for text in sent_texts(agent))


def test_the_breakdown_shows_the_facts_block():
    client = RecordingClient()
    agent = agent_with(client, keep_last=2)
    for number in range(4):
        agent.ask("Вопрос номер {} про требования и сроки".format(number))
    breakdown = agent.breakdown()
    assert breakdown.facts > 0
    assert breakdown.summary == 0
    assert breakdown.saved > 0


# --------------------------------------------------------------------------
# Разбор ответа модели
# --------------------------------------------------------------------------

def test_json_wrapped_in_prose_is_still_understood():
    """Модель то обрамит ответ кавычками, то припишет вступление."""
    card = recording.parse('Вот картотека:\n```json\n{"срок": "1 марта"}\n```\nГотово.')
    assert card == {"срок": "1 марта"}


def test_a_nested_value_is_flattened_not_dropped():
    card = recording.parse('{"стек": ["Kotlin", "Ktor"], "команда": {"бэкенд": 2}}')
    assert card == {"стек": "Kotlin, Ktor", "команда": "бэкенд: 2"}


def test_an_answer_without_json_is_refused():
    with pytest.raises(ValueError):
        recording.parse("Никакой картотеки я составлять не буду.")


def test_a_broken_answer_keeps_the_previous_card_and_leaves_a_note():
    client = RecordingClient(card=["{\"срок\": \"1 марта\"}", "это не JSON"])
    agent = agent_with(client)
    agent.ask("Срок — первое марта")
    agent.ask("А ещё нужен тёмный режим")
    assert agent.conversation.facts == {"срок": "1 марта"}
    assert agent.facts_note


def test_failures_in_a_row_are_not_paid_for_on_every_question():
    """Модель, которая вместо JSON пишет прозу, пишет её и на десятом вопросе."""
    client = RecordingClient(card="это не JSON")
    agent = agent_with(client)
    for number in range(6):
        agent.ask("Вопрос номер {}".format(number))
    assert client.updates < 6
    assert agent.facts_note


def test_a_success_returns_the_usual_pace():
    client = RecordingClient(card=["это не JSON", "и это не JSON",
                                   {"итог": "разобрано"}, {"ещё": "факт"}])
    agent = agent_with(client)
    for number in range(8):
        agent.ask("Вопрос номер {}".format(number))
    assert agent.conversation.facts == {"итог": "разобрано", "ещё": "факт"}
    assert not agent.facts_note


def test_a_failed_request_does_not_break_the_answer():
    agent = agent_with(BrokenRecorder())
    result = agent.ask("Вопрос про сроки")
    assert result.text == ANSWER
    assert agent.facts_note


def test_only_the_freshest_unseen_messages_go_into_a_tight_request():
    """Непросмотренного бывает больше, чем влезает в запрос.

    Копится оно после неудачного разбора: отметка просмотренного не двигается.
    Тогда разбираются самые свежие сообщения — остальные и без того остались бы
    неразобранными.
    """
    block = [Message("user", "Сообщение номер {}".format(number))
             for number in range(20)]
    kept = recording.fitting(block, 40, count_text_tokens)
    assert kept and len(kept) < len(block)
    assert kept[-1] is block[-1]


def test_a_message_wider_than_the_whole_request_is_clipped_not_skipped():
    block = [Message("user", "очень длинное сообщение " * 200)]
    kept = recording.fitting(block, 50, count_text_tokens)
    assert len(kept) == 1
    assert count_text_tokens(kept[0].content) < count_text_tokens(block[0].content)


def test_the_facts_request_stays_inside_the_window():
    """Иначе картотека не обновилась бы уже никогда: запрос сам не влезает."""
    client = RecordingClient(card="это не JSON")
    agent = agent_with(client, keep_last=2, window=2000, on_overflow="trim")
    for number in range(6):
        agent.ask("Вопрос номер {}. {}".format(number, "Подробности разговора. " * 20))
    asked = [sent for sent in client.calls if client.asked_for_facts(sent)][-1]
    assert count_message_tokens(asked) < 2000
    assert agent.facts_note


# --------------------------------------------------------------------------
# Слияние и вытеснение
# --------------------------------------------------------------------------

def test_facts_are_merged_not_replaced():
    """Модель, забывшая факт, не должна его стирать."""
    client = RecordingClient(card=[{"срок": "1 марта", "бюджет": "300 тысяч"},
                                   {"платформа": "Android"}])
    agent = agent_with(client)
    agent.ask("Срок и бюджет")
    agent.ask("Платформа")
    assert agent.conversation.facts == {"срок": "1 марта", "бюджет": "300 тысяч",
                                        "платформа": "Android"}


def test_an_empty_value_forgets_the_fact():
    client = RecordingClient(card=[{"срок": "1 марта"}, {"срок": ""}])
    agent = agent_with(client)
    agent.ask("Срок первое марта")
    agent.ask("Срок отменяется")
    assert agent.conversation.facts == {}


def test_extra_facts_are_squeezed_out_by_age():
    client = RecordingClient(card={"а": "1", "б": "2", "в": "3", "г": "4"})
    agent = agent_with(client, max_facts=2)
    agent.ask("Вопрос")
    assert list(agent.conversation.facts) == ["в", "г"]


def test_only_what_fits_the_limit_is_sent():
    client = RecordingClient(card={"старый": "значение " * 40,
                                   "свежий": "короткое значение"})
    agent = agent_with(client, max_tokens=60)
    agent.ask("Вопрос")
    sent = agent.conversation.sent_facts()
    assert list(sent) == ["свежий"]
    assert len(agent.conversation.facts) == 2


def test_a_fact_can_be_written_by_hand():
    client = RecordingClient(card={})
    agent = agent_with(client)
    agent.remember_fact("срок", "к 1 марта")
    assert agent.conversation.facts == {"срок": "к 1 марта"}
    agent.remember_fact("срок", "")
    assert agent.conversation.facts == {}


def test_a_handwritten_fact_survives_the_next_update():
    client = RecordingClient(card={"платформа": "Android"})
    agent = agent_with(client)
    agent.remember_fact("срок", "к 1 марта")
    agent.ask("Вопрос про платформу")
    assert agent.conversation.facts == {"срок": "к 1 марта", "платформа": "Android"}


# --------------------------------------------------------------------------
# Сессия
# --------------------------------------------------------------------------

def test_facts_survive_the_session_file(tmp_path):
    client = RecordingClient()
    agent = agent_with(client)
    agent.ask("Платформа Android, бюджет 300 тысяч")
    store = SessionStore(str(tmp_path))
    saved = store.save(agent, title="сбор ТЗ")
    returned = restore_agent(store.load(saved.session_id), agent.config)
    assert returned.conversation.facts == CARD
    assert returned.conversation.facts_seen == agent.conversation.facts_seen


def test_a_session_from_a_previous_version_simply_has_no_facts(tmp_path):
    client = RecordingClient()
    agent = agent_with(client, strategy="window")
    agent.ask("Вопрос без картотеки")
    store = SessionStore(str(tmp_path))
    saved = store.save(agent, title="старая")
    payload = store.load(saved.session_id).conversation
    payload.pop("facts", None)
    payload.pop("facts_seen", None)
    record = store.load(saved.session_id)
    record.conversation = payload
    returned = restore_agent(record, agent.config)
    assert returned.conversation.facts == {}
