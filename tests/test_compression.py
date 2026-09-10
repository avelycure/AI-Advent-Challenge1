"""Сжатие истории: пересказ вместо начала разговора.

Ни один тест не ходит в сеть: пересказ пишет заглушка, которая узнаёт запрос
за пересказом по системному промпту и отвечает на него отдельным текстом.
"""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    CompressionConfig,
    ConfigError,
    Conversation,
    HistoryConfig,
    LLMError,
    SessionStore,
    Transport,
    overrides,
    restore_agent,
)
from llmagent import compression
from llmagent.history import SUMMARY_PREFIX
from llmagent.transport import Completion, count_message_tokens, count_text_tokens
from llmagent.usage import MAIN, SUMMARY

ANSWER = "Ответ модели на заданный вопрос, довольно подробный и не короткий."
RETELLING = "Пересказ начала: пользователя зовут Иван, обсуждали структуры данных."


class SummarizingClient:
    """Заглушка, отвечающая на просьбу пересказать иначе, чем на вопрос."""

    def __init__(self, answer: str = ANSWER, retelling: str = RETELLING) -> None:
        self.answer = answer
        self.retelling = retelling
        self.calls = []
        self.kwargs = []

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref: str, messages, max_tokens: int, temperature: float = 0.7,
                 top_p=None, stop=None, response_format=None, tools=None) -> Completion:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append({"max_tokens": max_tokens, "temperature": temperature,
                            "stop": stop, "response_format": response_format})
        text = self.retelling if self.asked_to_summarize(messages) else self.answer
        return Completion(text, count_message_tokens(messages),
                          count_text_tokens(text), elapsed=0.001)

    @staticmethod
    def asked_to_summarize(messages) -> bool:
        return bool(messages) and messages[0].get("content") == compression.SUMMARY_SYSTEM

    @property
    def summaries(self) -> int:
        return sum(1 for sent in self.calls if self.asked_to_summarize(sent))


class BrokenSummarizer(SummarizingClient):
    """Пересказ не даётся: на просьбу о нём приходит ошибка."""

    def complete(self, model_ref: str, messages, max_tokens: int, **kwargs) -> Completion:
        if self.asked_to_summarize(messages):
            # Запрос записываем и в этом случае: без записи не проверить, что
            # за пересказом вообще обращались.
            self.calls.append([dict(m) for m in messages])
            raise LLMError("провайдер недоступен")
        return super().complete(model_ref, messages, max_tokens, **kwargs)


def agent_with(client, keep_last: int = 6, every: int = 10, enabled: bool = True,
               max_tokens: int = 400, **history) -> Agent:
    config = AgentConfig(
        name="t", transport=Transport(demo=True, demo_delay=0.0),
        history=HistoryConfig(compression=CompressionConfig(
            enabled=enabled, keep_last=keep_last, every=every,
            max_tokens=max_tokens), **history))
    return Agent(config, client=client)


def converse(agent: Agent, rounds: int, first: str = "Меня зовут Иван") -> None:
    agent.ask(first)
    for number in range(rounds - 1):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))


# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

def test_compression_is_on_by_default_with_sane_numbers():
    policy = AgentConfig().history.compression
    assert policy.enabled
    assert policy.keep_last >= 2
    assert policy.every >= 1


def test_a_tail_shorter_than_a_pair_is_refused():
    """Ответ без своего вопроса модель прочтёт как ответ неизвестно на что."""
    with pytest.raises(ConfigError):
        CompressionConfig(keep_last=1)


def test_nonsense_numbers_are_refused():
    with pytest.raises(ConfigError):
        CompressionConfig(every=0)
    with pytest.raises(ConfigError):
        CompressionConfig(max_tokens=0)


def test_settings_are_reachable_from_the_command_line():
    config = overrides.apply(AgentConfig(), [
        "history.compression.enabled=false",
        "history.compression.keep_last=4",
        "history.compression.every=2",
    ])
    assert config.history.compression.enabled is False
    assert config.history.compression.keep_last == 4
    assert config.history.compression.every == 2


def test_a_typo_in_the_settings_is_refused_at_startup():
    with pytest.raises(ConfigError):
        overrides.apply(AgentConfig(), ["history.compression.keep_lst=4"])


def test_settings_survive_the_round_trip_through_a_dictionary():
    config = AgentConfig(history=HistoryConfig(
        compression=CompressionConfig(keep_last=4, every=3)))
    restored = AgentConfig.from_dict(config.to_dict())
    assert restored.history.compression == config.history.compression


# --------------------------------------------------------------------------
# Когда сжатие срабатывает
# --------------------------------------------------------------------------

def test_a_short_dialog_costs_no_extra_requests():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 4)
    assert client.summaries == 0
    assert agent.conversation.summary == ""
    assert agent.usage.by_kind[SUMMARY].requests == 0


def test_the_beginning_is_folded_once_enough_has_piled_up():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    assert client.summaries == 1
    assert agent.conversation.summary == RETELLING
    assert agent.conversation.summarized > 0


def test_nothing_new_is_summarized_twice():
    """Порог считается от прежней границы, а не от начала разговора."""
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    agent.ask("Ещё один вопрос")
    assert client.summaries == 1


def test_the_second_fold_takes_the_first_summary_as_material():
    client = SummarizingClient()
    agent = agent_with(client, every=4)
    converse(agent, 14)
    assert client.summaries > 1
    asked = [sent for sent in client.calls if client.asked_to_summarize(sent)]
    assert RETELLING in asked[-1][-1]["content"]


def test_switching_it_off_keeps_the_whole_history():
    client = SummarizingClient()
    agent = agent_with(client, enabled=False)
    converse(agent, 12)
    assert client.summaries == 0
    assert agent.conversation.summary == ""
    assert "Иван" in client.calls[-1][1]["content"]


# --------------------------------------------------------------------------
# Что уходит в модель
# --------------------------------------------------------------------------

def test_the_folded_beginning_goes_as_a_summary_instead_of_messages():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    sent = agent.conversation.api_messages("Системный промпт")
    assert sent[0]["role"] == "system"
    assert sent[1]["content"].endswith(RETELLING)
    assert not any("Иван" in message["content"] for message in sent[2:])


def test_the_summary_goes_as_a_message_the_provider_will_accept():
    """Двух системных сообщений подряд принимают не все, двух вопросов — все."""
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    sent = agent.conversation.api_messages("Системный промпт")
    assert sent[1]["role"] == "user"


def test_the_folded_messages_stay_in_the_conversation():
    """На экране разговор виден целиком: сжат он только по дороге в модель."""
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    assert any("Иван" in message.content for message in agent.conversation.messages)


def test_the_verbatim_tail_starts_with_a_question():
    client = SummarizingClient()
    agent = agent_with(client, keep_last=5)
    converse(agent, 10)
    tail = agent.conversation.messages[agent.conversation.summarized]
    assert tail.role == "user"


def test_the_tail_keeps_at_least_the_asked_for_number_of_messages():
    client = SummarizingClient()
    agent = agent_with(client, keep_last=6)
    converse(agent, 12)
    kept = len(agent.conversation.messages) - agent.conversation.summarized
    assert kept >= 6


def test_the_request_gets_smaller_after_folding():
    """Ради этого всё и делается: тот же вопрос уходит в модель дешевле.

    Сравниваются два одинаковых разговора: в одном сжатие включено, в другом
    нет. Меряется последний запрос — тот, в который у несжатой истории успело
    набежать больше всего.
    """
    plain, folding = SummarizingClient(), SummarizingClient()
    converse(agent_with(plain, enabled=False), 12)
    converse(agent_with(folding), 12)
    answered = [sent for sent in folding.calls if not folding.asked_to_summarize(sent)]
    assert count_message_tokens(answered[-1]) < count_message_tokens(plain.calls[-1])


def test_the_breakdown_shows_the_summary_and_what_it_saves():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    breakdown = agent.breakdown()
    assert breakdown.summary > 0
    assert breakdown.folded_messages == agent.conversation.summarized
    assert breakdown.saved > 0
    assert sum(part.tokens for part in breakdown.parts()) == breakdown.estimated_input
    assert any(part.title == "Пересказ начала" for part in breakdown.parts())


def test_the_measured_size_is_forgotten_when_the_history_changes():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 9)
    agent.conversation.absorb_summary("Пересказ", 4)
    assert agent.conversation.exact_context == 0


# --------------------------------------------------------------------------
# Сам запрос за пересказом
# --------------------------------------------------------------------------

def test_the_summary_request_is_counted_separately():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    assert agent.usage.by_kind[SUMMARY].requests == 1
    assert agent.usage.by_kind[SUMMARY].total_tokens > 0
    assert agent.usage.by_kind[MAIN].requests == 10


def test_the_summary_request_does_not_inherit_the_answer_format():
    """Просить у пересказа JSON и договорный маркер конца бессмысленно."""
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    asked = [kwargs for sent, kwargs in zip(client.calls, client.kwargs)
             if client.asked_to_summarize(sent)]
    assert not asked[0]["stop"]
    assert not asked[0]["response_format"]


def test_the_summary_request_fits_the_window():
    client = SummarizingClient()
    agent = agent_with(client, window=2000, on_overflow="trim")
    converse(agent, 20)
    for sent in client.calls:
        assert count_message_tokens(sent) + agent.output_reserve <= agent.context_limit


def test_a_message_wider_than_the_window_is_clipped_rather_than_left_alone():
    """Иначе такое сообщение заперло бы разговор навсегда: сжать его нельзя."""
    client = SummarizingClient()
    agent = agent_with(client, window=2000, every=2, keep_last=2,
                       on_overflow="trim")
    agent.conversation.add_user("очень длинное сообщение " * 500)
    agent.conversation.add_assistant("Ответ.")
    agent.conversation.add_user("Второй вопрос")
    agent.conversation.add_assistant("Второй ответ.")
    agent.ask("А теперь короткий вопрос")
    assert agent.conversation.summary
    asked = [sent for sent in client.calls if client.asked_to_summarize(sent)]
    assert compression.CLIPPED in asked[0][-1]["content"]


def test_a_long_beginning_is_retold_in_several_goes():
    client = SummarizingClient()
    agent = agent_with(client, window=2000, every=20, keep_last=2,
                       on_overflow="trim")
    for _ in range(15):
        agent.conversation.add_user("вопрос про структуры данных " * 20)
        agent.conversation.add_assistant("ответ про структуры данных " * 20)
    agent.ask("Короткий вопрос в конце")
    assert client.summaries > 1


def test_a_window_too_small_for_a_summary_says_so():
    """Отказ с числами понятнее, чем молчаливое переполнение изнутри сжатия."""
    agent = agent_with(SummarizingClient(), window=600, keep_last=2, every=2,
                       max_tokens=400, on_overflow="trim")
    converse(agent, 6)
    assert "не хватает на пересказ" in agent.compression_note
    assert agent.conversation.exchanges == 6


# --------------------------------------------------------------------------
# Когда пересказ не получился
# --------------------------------------------------------------------------

def test_a_failed_summary_does_not_break_the_answer():
    client = BrokenSummarizer()
    agent = agent_with(client)
    converse(agent, 10)
    assert agent.conversation.exchanges == 10
    assert agent.conversation.summary == ""


def test_a_failed_summary_says_why():
    agent = agent_with(BrokenSummarizer())
    converse(agent, 10)
    assert "недоступен" in agent.compression_note


def test_a_failed_summary_is_not_paid_for_again_at_every_question():
    """Неудачный пересказ — оплаченный запрос: повторять его каждый раз нельзя."""
    client = BrokenSummarizer()
    agent = agent_with(client, every=6)
    converse(agent, 10)
    failed = client.summaries
    agent.ask("Ещё вопрос")
    agent.ask("И ещё вопрос")
    assert client.summaries == failed


def test_a_failed_summary_is_retried_once_the_next_block_piles_up():
    client = BrokenSummarizer()
    agent = agent_with(client, every=4)
    converse(agent, 10)
    failed = client.summaries
    converse(agent, 5, first="Продолжаем разговор")
    assert client.summaries > failed


def test_an_empty_summary_leaves_the_history_alone():
    agent = agent_with(SummarizingClient(retelling="   "))
    converse(agent, 10)
    assert agent.conversation.summary == ""
    assert agent.conversation.summarized == 0
    assert "пуст" in agent.compression_note


def test_asking_for_it_by_hand_reports_the_failure():
    """``/compress`` просят намеренно — промолчать о неудаче было бы обманом."""
    agent = agent_with(BrokenSummarizer())
    converse(agent, 6)
    with pytest.raises(LLMError):
        agent.compress_history(force=True)


# --------------------------------------------------------------------------
# Сжатие по требованию
# --------------------------------------------------------------------------

def test_compressing_by_hand_does_not_wait_for_the_threshold():
    client = SummarizingClient()
    agent = agent_with(client, every=100)
    converse(agent, 6)
    assert agent.compress_history(force=True) > 0
    assert agent.conversation.summary == RETELLING


def test_compressing_an_empty_dialog_asks_nothing():
    client = SummarizingClient()
    agent = agent_with(client)
    assert agent.compress_history(force=True) == 0
    assert client.calls == []


def test_compressing_twice_in_a_row_asks_nothing_the_second_time():
    client = SummarizingClient()
    agent = agent_with(client, every=100)
    converse(agent, 6)
    agent.compress_history(force=True)
    asked = client.summaries
    assert agent.compress_history(force=True) == 0
    assert client.summaries == asked


# --------------------------------------------------------------------------
# Обрезка и сжатие вместе
# --------------------------------------------------------------------------

def test_trimming_throws_out_what_is_actually_sent():
    """Выбрасывать сжатое начало бессмысленно: в запросе его и так нет."""
    client = SummarizingClient()
    agent = agent_with(client, window=2000, keep_last=2, every=2,
                       on_overflow="trim")
    converse(agent, 25)
    assert agent.conversation.summary
    assert agent.conversation.messages[agent.conversation.summarized].role == "user"
    for sent in client.calls:
        assert count_message_tokens(sent) + agent.output_reserve <= agent.context_limit


def test_a_full_window_is_compressed_before_anything_is_thrown_out():
    """Обрезка теряет начало насовсем, пересказ его сохраняет — он и первый."""
    client = SummarizingClient()
    agent = agent_with(client, window=1500, keep_last=2, every=100,
                       max_tokens=200, on_overflow="trim")
    converse(agent, 30)
    assert client.summaries > 0
    assert agent.conversation.summary == RETELLING


def test_compression_is_not_paid_for_when_it_would_not_free_space():
    """Пересказ длиннее того, что он заменит, места не освободит."""
    client = SummarizingClient()
    agent = agent_with(client, keep_last=2, every=100, max_tokens=1000)
    converse(agent, 4)
    assert agent.fold_to_fit() == 0
    assert client.summaries == 0


def test_a_dialog_that_would_have_stopped_keeps_going_with_compression():
    """Тот же разговор в том же окне: без сжатия он упирается, со сжатием идёт."""
    without = agent_with(SummarizingClient(), enabled=False, window=2000)
    with pytest.raises(Exception):
        converse(without, 40)
    withal = agent_with(SummarizingClient(), window=2000, keep_last=4, every=4)
    converse(withal, 40)
    assert withal.conversation.exchanges == 40


# --------------------------------------------------------------------------
# Пересказ и сессия
# --------------------------------------------------------------------------

def test_the_summary_is_saved_and_comes_back(tmp_path):
    store = SessionStore(str(tmp_path))
    agent = agent_with(SummarizingClient())
    converse(agent, 10)
    store.save(agent, title="Со сжатием")

    returned = restore_agent(store.load(agent.session_id), agent.config,
                             client=SummarizingClient())
    assert returned.conversation.summary == RETELLING
    assert returned.conversation.summarized == agent.conversation.summarized


def test_a_file_from_an_older_version_has_no_summary_and_that_is_fine():
    conversation = Conversation.restore({"messages": [
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": "Ответ"}]})
    assert conversation.summary == ""
    assert conversation.summarized == 0
    assert len(conversation.api_messages("")) == 2


def test_a_boundary_beyond_the_messages_is_pulled_back():
    """Иначе в модель ушёл бы один пересказ без единого сообщения."""
    conversation = Conversation.restore({
        "summary": "Пересказ", "summarized": 99,
        "messages": [{"role": "user", "content": "Вопрос"}]})
    assert conversation.summarized == 1


def test_starting_over_forgets_the_summary():
    agent = agent_with(SummarizingClient())
    converse(agent, 10)
    agent.new_session()
    assert agent.conversation.summary == ""
    assert agent.conversation.summarized == 0
    assert agent.compression_note == ""


def test_changing_the_model_keeps_the_summary():
    """Пересказ — это текст, а не измерение чужим токенизатором."""
    agent = agent_with(SummarizingClient())
    converse(agent, 10)
    agent.reconfigure(provider="deepseek", model="deepseek-chat")
    assert agent.conversation.summary == RETELLING


def test_repeating_a_question_from_the_tail_still_sees_the_summary():
    client = SummarizingClient()
    agent = agent_with(client)
    converse(agent, 10)
    agent.retry_last()
    assert client.calls[-1][1]["content"].endswith(RETELLING)


def test_repeating_a_question_from_the_folded_beginning_sees_the_messages():
    """Пересказ ответа на этот вопрос не заменит: нужен сам вопрос."""
    agent = agent_with(SummarizingClient())
    converse(agent, 10)
    messages = agent.conversation.api_messages_upto(1, "Системный промпт")
    assert "Иван" in messages[1]["content"]
    assert not messages[1]["content"].startswith(SUMMARY_PREFIX)
