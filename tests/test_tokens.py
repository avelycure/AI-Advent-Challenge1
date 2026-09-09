"""Подсчёт токенов: раскладка запроса, сверка с фактом, рост по обменам."""
from __future__ import annotations

import pytest

from llmagent import (
    Agent,
    AgentConfig,
    ConfigError,
    ContextOverflow,
    GenerationParams,
    HistoryConfig,
    InputPolicy,
    Transport,
)

from llmagent.transport import count_message_tokens

from conftest import ScriptedClient


def agent_with(client, **changes) -> Agent:
    config = AgentConfig(name="t", transport=Transport(demo=True, demo_delay=0.0),
                         **changes)
    return Agent(config, client=client)


# --------------------------------------------------------------------------
# Раскладка запроса
# --------------------------------------------------------------------------

def test_empty_dialog_costs_only_the_system_prompt():
    breakdown = agent_with(ScriptedClient(["Ответ."])).breakdown()
    assert breakdown.system > 0
    assert breakdown.history == 0
    assert breakdown.pending == 0
    assert breakdown.input_tokens == breakdown.estimated_input


def test_unanswered_question_counts_as_pending_not_history():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.conversation.add_user("Расскажи про деревья поиска")
    breakdown = agent.breakdown()
    assert breakdown.pending > 0
    assert breakdown.history == 0


def test_answered_exchange_moves_into_history():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Первый вопрос")
    agent.conversation.add_user("Второй вопрос")
    breakdown = agent.breakdown()
    assert breakdown.history > 0
    assert breakdown.pending > 0


def test_parts_add_up_to_the_estimate_and_skip_the_empty():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    breakdown = agent.breakdown()
    parts = breakdown.parts()
    assert sum(part.tokens for part in parts) == breakdown.estimated_input
    # Неотвеченного вопроса нет — строки про него быть не должно.
    assert all(part.title != "Новый вопрос" for part in parts)
    assert pytest.approx(sum(part.share for part in parts), abs=1e-6) == 1.0


def test_reserve_is_counted_against_the_window():
    agent = agent_with(ScriptedClient(["Ответ."]))
    breakdown = agent.breakdown()
    assert breakdown.planned == breakdown.input_tokens + breakdown.reserve
    assert breakdown.free == breakdown.window - breakdown.planned
    assert breakdown.fits
    assert breakdown.excess == 0


# --------------------------------------------------------------------------
# Сверка оценки с фактом
# --------------------------------------------------------------------------

def test_before_the_first_answer_there_is_nothing_to_compare():
    assert agent_with(ScriptedClient(["Ответ."])).breakdown().drift is None


def test_after_the_answer_the_estimate_is_compared_with_the_provider():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    breakdown = agent.breakdown()
    assert breakdown.measured > 0
    assert breakdown.measured_estimate > 0
    assert breakdown.drift is not None


def test_the_estimate_is_corrected_by_what_the_provider_counted():
    """Провайдер насчитал вдвое меньше — следующая оценка это учитывает."""
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    agent.conversation.note_exchange(50, 0, estimate=100)
    breakdown = agent.breakdown()
    assert breakdown.scale == pytest.approx(0.5)
    assert breakdown.input_tokens == round(breakdown.estimated_input * 0.5)


def test_one_odd_measurement_cannot_skew_the_count():
    """Границы поправки: короткий первый обмен не должен перекосить весь счёт."""
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    agent.conversation.note_exchange(10_000, 0, estimate=10)
    assert agent.breakdown().scale == 2.0
    agent.conversation.note_exchange(1, 0, estimate=10_000)
    assert agent.breakdown().scale == 0.5


def test_the_correction_survives_trimming():
    """Поправка описывает токенизатор модели, а не выброшенные сообщения."""
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    agent.conversation.note_exchange(50, 0, estimate=100)
    agent.conversation.drop_oldest_exchange()
    assert agent.conversation.exact_context == 0
    assert agent.conversation.scale == pytest.approx(0.5)


def test_changing_the_model_drops_the_comparison():
    """Точная цифра измерена чужим токенизатором и после смены модели ничего не значит."""
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    agent.conversation.note_exchange(50, 0, estimate=100)
    agent.reconfigure(provider="groq", model="qwen/qwen3.8-27b")
    breakdown = agent.breakdown()
    assert breakdown.measured == 0
    assert breakdown.drift is None
    assert breakdown.scale == 1.0
    assert breakdown.input_tokens == breakdown.estimated_input


def test_a_trimming_agent_never_refuses_on_its_own_arithmetic():
    """Обрезка и застава перед отправкой обязаны считать одним числом.

    Однажды считали разными: обрезка мерила поправленной оценкой, застава —
    сырой, и агент с history.on_overflow=trim всё равно вставал с отказом.
    """
    agent = agent_with(ScriptedClient(["Короткий ответ."]),
                       history=HistoryConfig(window=1500, on_overflow="trim"))
    # Провайдер, который считает вдвое экономнее нашего токенизатора.
    agent.ask("Первый вопрос")
    agent.conversation.note_exchange(agent.conversation.exact_estimate // 2, 0,
                                     estimate=agent.conversation.exact_estimate)
    for number in range(30):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))


# --------------------------------------------------------------------------
# Рост по обменам
# --------------------------------------------------------------------------

def test_growth_is_empty_until_the_first_answer():
    assert agent_with(ScriptedClient(["Ответ."])).growth() == []


def test_each_exchange_sends_the_whole_dialog_again():
    """Главное про токены: вход растёт, хотя вопросы одинаковой длины."""
    agent = agent_with(ScriptedClient(["Ответ."]))
    for _ in range(4):
        agent.ask("Один и тот же по длине вопрос")
    steps = agent.growth()
    assert [step.number for step in steps] == [1, 2, 3, 4]
    prompts = [step.prompt_tokens for step in steps]
    assert prompts == sorted(prompts)
    assert prompts[-1] > prompts[0]


def test_growth_keeps_a_running_total():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Первый")
    agent.ask("Второй")
    steps = agent.growth()
    assert steps[1].total_tokens == steps[0].tokens + steps[1].tokens


def test_growth_remembers_which_question_it_answered():
    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Что такое куча")
    assert agent.growth()[0].question == "Что такое куча"


# --------------------------------------------------------------------------
# Сохранение
# --------------------------------------------------------------------------

def test_saved_dialog_keeps_the_comparison():
    from llmagent.history import Conversation

    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    restored = Conversation.restore(agent.conversation.to_dict())
    assert restored.exact_estimate == agent.conversation.exact_estimate


def test_dialog_saved_by_an_older_version_still_reads():
    from llmagent.history import Conversation

    payload = {"messages": [], "exact_context": 10}
    assert Conversation.restore(payload).exact_estimate == 0


# --------------------------------------------------------------------------
# Суженное окно
# --------------------------------------------------------------------------

def test_window_can_be_narrowed_below_the_model():
    agent = agent_with(ScriptedClient(["Ответ."]), history=HistoryConfig(window=4000))
    assert agent.context_limit == 4000


def test_narrowing_never_widens_the_model_window():
    """Окно шире модельного — не окно, а обещание, которого провайдер не даст."""
    agent = agent_with(ScriptedClient(["Ответ."]),
                       history=HistoryConfig(window=10_000_000))
    assert agent.context_limit == AgentConfig().model_info.context_window


def test_default_reserve_shrinks_with_the_window():
    agent = agent_with(ScriptedClient(["Ответ."]), history=HistoryConfig(window=2000))
    assert agent.output_reserve == 500
    assert agent.input_budget == 1500


def test_asked_max_tokens_is_taken_as_it_is():
    agent = agent_with(ScriptedClient(["Ответ."]), history=HistoryConfig(window=2000),
                       generation=GenerationParams(max_tokens=1200))
    assert agent.output_reserve == 1200


def test_answer_longer_than_the_whole_window_is_refused_at_startup():
    with pytest.raises(ConfigError):
        AgentConfig(history=HistoryConfig(window=1000),
                    generation=GenerationParams(max_tokens=2000))


def test_useless_window_is_refused():
    with pytest.raises(ConfigError):
        HistoryConfig(window=10)


def test_unknown_overflow_policy_is_refused():
    with pytest.raises(ConfigError):
        HistoryConfig(on_overflow="забыть всё")


# --------------------------------------------------------------------------
# Переполнение: отказ
# --------------------------------------------------------------------------

def crowded(**history) -> Agent:
    """Агент в тесном окне: переполнение достижимо за десяток вопросов."""
    return agent_with(ScriptedClient(["Короткий ответ."]),
                      history=HistoryConfig(window=1500, **history))


# Вопрос, который сам по себе шире окна, но короче предела входной политики.
HUGE = "очень длинный вопрос " * 200


def fill(agent: Agent, attempts: int = 60) -> int:
    """Спрашивать, пока помещается. Возвращает число состоявшихся обменов."""
    for number in range(attempts):
        try:
            agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
        except ContextOverflow:
            return number
    raise AssertionError("окно так и не переполнилось")


def test_overflow_stops_the_dialog_by_default():
    agent = crowded()
    assert fill(agent) > 0
    with pytest.raises(ContextOverflow):
        agent.ask("Ещё один вопрос")


def test_refusal_says_how_much_is_too_much():
    agent = crowded()
    fill(agent)
    with pytest.raises(ContextOverflow) as failure:
        agent.ask("Ещё один вопрос")
    for expected in ("не помещается", "окне 1500", "лишних"):
        assert expected in str(failure.value)


def test_refused_question_does_not_stay_in_the_dialog():
    """История остаётся целой: неотвеченный вопрос из неё убирается."""
    agent = crowded()
    fill(agent)
    before = len(agent.conversation.messages)
    with pytest.raises(ContextOverflow):
        agent.ask("Ещё один вопрос")
    assert len(agent.conversation.messages) == before


def test_refusal_costs_nothing():
    agent = crowded()
    fill(agent)
    spent = agent.usage.total_tokens
    with pytest.raises(ContextOverflow):
        agent.ask("Ещё один вопрос")
    assert agent.usage.total_tokens == spent


def test_one_question_longer_than_the_window_is_refused_outright():
    agent = crowded()
    with pytest.raises(ContextOverflow):
        agent.ask(HUGE)


def test_question_longer_than_the_window_is_refused_even_with_trimming():
    """Обрезать нечего: вопрос и есть то, ради чего идёт запрос."""
    agent = crowded(on_overflow="trim")
    with pytest.raises(ContextOverflow):
        agent.ask(HUGE)


# --------------------------------------------------------------------------
# Переполнение: обрезка
# --------------------------------------------------------------------------

def test_trimming_keeps_the_dialog_going():
    agent = crowded(on_overflow="trim")
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert agent.usage.by_kind["основной"].requests == 40


def test_trimming_forgets_the_beginning():
    agent = crowded(on_overflow="trim")
    agent.ask("Меня зовут Иван")
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert all("Иван" not in message.content for message in agent.conversation.messages)


def test_trimming_drops_a_question_together_with_its_answer():
    agent = crowded(on_overflow="trim")
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert agent.conversation.messages[0].role == "user"


def test_every_request_actually_sent_fits_the_window():
    """Обрезка не для красоты: провайдер не должен увидеть запрос сверх окна."""
    client = ScriptedClient(["Короткий ответ."])
    agent = agent_with(client, history=HistoryConfig(window=1500, on_overflow="trim"))
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert client.calls
    for sent in client.calls:
        assert count_message_tokens(sent) + agent.output_reserve <= agent.context_limit


def test_trimming_never_touches_the_system_prompt():
    agent = crowded(on_overflow="trim")
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    sent = agent.conversation.api_messages(agent.config.system_prompt)
    assert sent[0]["role"] == "system"


def test_trimming_does_not_make_the_spending_disappear():
    """Забытое всё равно оплачено: счётчик помнит то, чего не помнит модель."""
    agent = crowded(on_overflow="trim")
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert agent.usage.total_tokens > agent.context_used()


def test_trimming_forgets_the_exact_size_it_measured():
    """Точная цифра относилась к прежней, длинной переписке."""
    agent = crowded(on_overflow="trim")
    agent.ask("Первый вопрос")
    assert agent.conversation.exact_context > 0
    agent.conversation.drop_oldest_exchange()
    assert agent.conversation.exact_context == 0


def test_nothing_to_trim_when_no_question_was_answered():
    agent = crowded(on_overflow="trim")
    agent.conversation.add_user("Один неотвеченный вопрос")
    assert agent.conversation.drop_oldest_exchange() == 0


# --------------------------------------------------------------------------
# Краевые случаи
# --------------------------------------------------------------------------

class SilentClient(ScriptedClient):
    """Провайдер, не вернувший usage: такие есть, и падать из-за них нельзя."""

    def complete(self, *args, **kwargs):
        completion = super().complete(*args, **kwargs)
        completion.prompt_tokens = 0
        completion.completion_tokens = 0
        return completion


def test_a_provider_that_reports_nothing_leaves_the_estimate_alone():
    agent = agent_with(SilentClient(["Ответ."]))
    agent.ask("Вопрос")
    breakdown = agent.breakdown()
    assert breakdown.drift is None
    assert breakdown.scale == 1.0
    assert breakdown.input_tokens == breakdown.estimated_input


def test_a_provider_that_reports_nothing_does_not_break_the_dialog():
    agent = agent_with(SilentClient(["Ответ."]), history=HistoryConfig(window=1500))
    for number in range(5):
        agent.ask("Вопрос {}".format(number))
    assert agent.conversation.exchanges == 5


def test_the_correction_is_saved_with_the_dialog():
    from llmagent.history import Conversation

    agent = agent_with(ScriptedClient(["Ответ."]))
    agent.ask("Вопрос")
    agent.conversation.note_exchange(50, 0, estimate=100)
    restored = Conversation.restore(agent.conversation.to_dict())
    assert restored.scale == pytest.approx(0.5)


def test_a_dialog_saved_before_the_correction_existed_reads_as_uncorrected():
    from llmagent.history import Conversation

    assert Conversation.restore({"messages": []}).scale == 1.0


def test_a_truncated_answer_is_not_an_overflow():
    """Упереться в max_tokens и не влезть в окно — разные беды с разным лечением."""
    agent = agent_with(ScriptedClient(["Ответ."], finish_reason="length"))
    result = agent.ask("Вопрос")
    assert result.finish_reason == "length"
    assert agent.breakdown().fits


# --------------------------------------------------------------------------
# Переполнение: отправить как есть
# --------------------------------------------------------------------------

def test_send_lets_the_provider_have_the_last_word():
    """Своя проверка молчит: показать надо именно отказ провайдера."""
    client = ScriptedClient(["Короткий ответ."])
    agent = agent_with(client, history=HistoryConfig(window=1500, on_overflow="send"),
                       input=InputPolicy(max_chars=100_000))
    for number in range(40):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert not agent.breakdown().fits
    assert len(client.calls) == 40


def test_send_does_not_forget_anything():
    agent = agent_with(ScriptedClient(["Короткий ответ."]),
                       history=HistoryConfig(window=1500, on_overflow="send"))
    agent.ask("Меня зовут Иван")
    for number in range(30):
        agent.ask("Вопрос {}: расскажи подробно про структуры данных".format(number))
    assert any("Иван" in message.content for message in agent.conversation.messages)


def test_send_passes_a_single_question_wider_than_the_window():
    agent = agent_with(ScriptedClient(["Короткий ответ."]),
                       history=HistoryConfig(window=1500, on_overflow="send"))
    assert agent.ask(HUGE).text == "Короткий ответ."


def test_only_send_skips_the_guard():
    for policy in ("stop", "trim"):
        agent = agent_with(ScriptedClient(["Ответ."]),
                           history=HistoryConfig(window=1500, on_overflow=policy))
        assert not agent.sends_anyway
    agent = agent_with(ScriptedClient(["Ответ."]),
                       history=HistoryConfig(window=1500, on_overflow="send"))
    assert agent.sends_anyway


# --------------------------------------------------------------------------
# Отказ провайдера
# --------------------------------------------------------------------------

# Подлинный ответ OpenRouter на запрос шире окна — снят живым прогоном.
REAL_OVERFLOW = (
    "Error code: 400 - {'error': {'message': \"This endpoint's maximum context "
    "length is 65536 tokens. However, you requested about 141100 tokens (141000 "
    "of text input, 100 in the output). Please reduce the length of either one.\", "
    "'code': 400}}")


class BadRequestError(Exception):
    """Имя как у исключения SDK: разбор смотрит именно на него."""


def test_real_overflow_answer_is_explained_not_dumped():
    """Отказ приходит с кодом 400, и общая ветка про 400 его перехватывала."""
    from llmagent.transport.client import describe_error

    explained = describe_error(BadRequestError(REAL_OVERFLOW))
    assert "Контекст переполнен" in explained
    assert "/new" in explained
    assert "Error code" not in explained


def test_other_bad_requests_are_still_shown_as_they_are():
    from llmagent.transport.client import describe_error

    explained = describe_error(BadRequestError("Error code: 400 - unknown parameter foo"))
    assert "отклонил запрос (400)" in explained
    assert "unknown parameter foo" in explained


def test_overflow_is_recognised_however_the_provider_words_it():
    from llmagent.transport.client import describe_error

    for wording in ("context_length_exceeded",
                    "This model's maximum context length is 4095 tokens",
                    "Input is too many tokens for this context window",
                    "Please reduce the length of the messages"):
        assert "Контекст переполнен" in describe_error(BadRequestError(wording)), wording


def test_the_provider_own_words_are_shown_after_the_advice():
    """Точный размер по счёту провайдера — то, ради чего эту ошибку и читают."""
    from llmagent.transport.client import describe_error

    explained = describe_error(BadRequestError(REAL_OVERFLOW))
    assert "Контекст переполнен" in explained
    assert "Провайдер:" in explained
    for number in ("65536", "141100", "141000"):
        assert number in explained
    # Обёртка SDK человеку не нужна: только строка, которую написал провайдер.
    assert "Error code" not in explained
    assert "'code': 400" not in explained


def test_a_provider_without_a_message_field_gets_only_the_advice():
    from llmagent.transport.client import describe_error

    explained = describe_error(BadRequestError("context_length_exceeded"))
    assert "Контекст переполнен" in explained
    assert "Провайдер:" not in explained


def test_provider_words_are_shown_literally_not_as_markup():
    """В тексте провайдера бывают квадратные скобки, и rich принимает их за разметку.

    Без экранирования «[bold]» и «[see docs]» пропадали из сообщения молча,
    а «[/]» роняло весь чат с MarkupError — ровно там, где человеку нужны
    точные цифры.
    """
    import io

    from rich.console import Console

    from llmchat.ui import error_panel

    for payload in ("длина [bold] превышена", "смотри [see docs]", "лишний [/] в тексте"):
        console = Console(file=io.StringIO(), width=120)
        console.print(error_panel(payload))
        shown = console.file.getvalue()
        assert payload in " ".join(shown.split()), payload
