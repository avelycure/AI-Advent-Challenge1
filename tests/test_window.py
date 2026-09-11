"""Скользящее окно: в модель уходят только последние N сообщений.

Ни один тест не ходит в сеть: отвечает заглушка или заранее написанный клиент.
"""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    ConfigError,
    Conversation,
    ContextOverflow,
    HistoryConfig,
    Transport,
    overrides,
)

ANSWER = "Ответ модели на заданный вопрос, довольно подробный и не короткий."


def agent_with(keep_last: int = 4, strategy: str = "window", client=None,
               **history) -> Agent:
    config = AgentConfig(
        name="t", transport=Transport(demo=True, demo_delay=0.0),
        history=HistoryConfig(strategy=strategy, keep_last=keep_last, **history))
    return Agent(config, client=client)


def converse(agent: Agent, rounds: int, first: str = "Меня зовут Иван") -> None:
    agent.ask(first)
    for number in range(rounds - 1):
        agent.ask("Вопрос {}: расскажи про структуры данных".format(number))


def sent_texts(agent: Agent):
    return [message["content"] for message
            in agent.conversation.api_messages("Системный промпт")]


# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

def test_the_default_strategy_is_the_one_from_the_previous_day():
    """Умолчание не меняется: вчерашний запуск должен работать как вчера."""
    assert AgentConfig().history.strategy == "summary"


def test_an_unknown_strategy_is_refused_at_startup():
    with pytest.raises(ConfigError):
        HistoryConfig(strategy="magic")


def test_the_strategy_is_reachable_from_the_command_line():
    config = overrides.apply(AgentConfig(), ["history.strategy=window",
                                             "history.keep_last=4"])
    assert config.history.strategy == "window"
    assert config.history.keep_last == 4


def test_day9_settings_are_still_understood():
    """Конфиг и сессия дня 9 должны открываться, а не падать с неизвестным полем."""
    config = AgentConfig.from_dict(
        {"history": {"compression": {"enabled": False, "keep_last": 4, "every": 6}}})
    assert config.history.strategy == "full"
    assert config.history.keep_last == 4
    assert config.history.compression.every == 6

    still_folding = AgentConfig.from_dict(
        {"history": {"compression": {"enabled": True, "keep_last": 3}}})
    assert still_folding.history.strategy == "summary"
    assert still_folding.history.keep_last == 3


# --------------------------------------------------------------------------
# Что уходит в модель
# --------------------------------------------------------------------------

def test_only_the_last_messages_are_sent():
    agent = agent_with(keep_last=4)
    converse(agent, 5)
    sent = sent_texts(agent)
    assert "Меня зовут Иван" not in sent
    assert len(sent) == 5                      # системный промпт и две пары


def test_the_dropped_messages_stay_on_the_screen_and_in_the_session():
    """Окно решает, что отправить, а не что помнить: переписка остаётся целой."""
    agent = agent_with(keep_last=4)
    converse(agent, 6)
    assert len(agent.conversation.messages) == 12
    assert agent.conversation.messages[0].content == "Меня зовут Иван"


def test_the_window_starts_with_a_question_not_with_an_answer():
    agent = agent_with(keep_last=3)
    converse(agent, 5)
    start = agent.conversation.sent_from()
    assert agent.conversation.messages[start].role == "user"


def test_a_short_dialog_sends_everything():
    agent = agent_with(keep_last=6)
    converse(agent, 2)
    assert agent.conversation.sent_from() == 0


def test_the_window_costs_no_extra_requests():
    """Ради экономии окно не платит ничего: лишних обращений к модели нет."""
    agent = agent_with(keep_last=4)
    converse(agent, 6)
    assert agent.usage.requests == 6


def test_the_breakdown_counts_only_what_is_sent():
    agent = agent_with(keep_last=4)
    converse(agent, 6)
    breakdown = agent.breakdown()
    assert breakdown.held_back == agent.conversation.sent_from()
    assert breakdown.summary == 0
    assert breakdown.saved > 0
    assert breakdown.input_tokens < agent.conversation.weigh(
        [{"role": m.role, "content": m.content} for m in agent.conversation.messages])


# --------------------------------------------------------------------------
# Переполнение
# --------------------------------------------------------------------------

def test_overflow_narrows_the_window_instead_of_deleting_messages():
    """Выбрасывать сообщения при скользящем окне бессмысленно.

    Окно отсчитывается от конца: на месте выброшенной пары в него втянулась бы
    пара постарше, и запрос не уменьшился бы. Поэтому сужается само окно, а
    переписка остаётся целой.
    """
    agent = agent_with(keep_last=8, window=700, on_overflow="trim")
    converse(agent, 6)
    before = len(agent.conversation.messages)
    agent.conversation.add_user("Ещё один вопрос, на который нужен ответ")
    agent.fit_context()
    assert len(agent.conversation.messages) == before + 1
    assert agent.conversation.keep_last < 8
    assert agent.breakdown().fits


def test_a_hopeless_overflow_still_refuses():
    agent = agent_with(keep_last=4, window=300, on_overflow="trim")
    with pytest.raises(ContextOverflow):
        agent.ask("Очень длинный вопрос. " * 200)


# --------------------------------------------------------------------------
# Переключение стратегии
# --------------------------------------------------------------------------

def test_switching_to_the_window_keeps_the_summary_for_later():
    """Пересказ — это текст, а не измерение: смена стратегии его не отменяет."""
    agent = agent_with(strategy="summary", keep_last=4)
    converse(agent, 4)
    agent.conversation.absorb_summary("Пересказ начала разговора", 4)
    agent.reconfigure(history=agent.config.history.with_strategy("window"))
    assert agent.conversation.summary == "Пересказ начала разговора"
    assert all("Пересказ начала" not in text for text in sent_texts(agent))

    agent.reconfigure(history=agent.config.history.with_strategy("summary"))
    assert any("Пересказ начала разговора" in text for text in sent_texts(agent))


def test_the_window_size_reaches_the_conversation_through_the_config():
    agent = agent_with(keep_last=4)
    agent.reconfigure(history=HistoryConfig(strategy="window", keep_last=8))
    assert agent.conversation.keep_last == 8


def test_a_restored_conversation_gets_the_current_strategy():
    """Переписка из файла не приносит с собой чужую стратегию: её задаёт конфиг."""
    agent = agent_with(keep_last=4)
    converse(agent, 3)
    payload = agent.conversation.to_dict()
    fresh = agent_with(keep_last=6, strategy="summary")
    fresh.conversation = Conversation.restore(payload)
    assert fresh.conversation.strategy == "summary"
    assert fresh.conversation.keep_last == 6
