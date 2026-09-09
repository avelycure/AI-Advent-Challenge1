"""Учёт токенов и денег — то, что ментор просил положить внутрь агента."""
from __future__ import annotations

import pytest

from llmagent import Budget, BudgetExceeded, UsageMeter
from llmagent.transport import PROVIDERS, Completion
from llmagent.usage import JUDGE, MAIN, REPAIR

OPENAI = PROVIDERS["openai"]
MINI = OPENAI.models[0]
FREE = PROVIDERS["groq"]


def completion(prompt=1000, out=500, reasoning=0, cached=0, elapsed=1.5) -> Completion:
    return Completion("текст", prompt, out, reasoning_tokens=reasoning,
                      cached_tokens=cached, elapsed=elapsed)


def test_price_follows_the_model_price_list():
    meter = UsageMeter()
    cost = meter.record(completion(1_000_000, 1_000_000), OPENAI, MINI)
    assert cost == pytest.approx(MINI.input_price + MINI.output_price)
    assert meter.cost("USD") == pytest.approx(cost)


def test_cached_input_is_cheaper():
    meter = UsageMeter()
    full = meter.record(completion(1_000_000, 0), OPENAI, MINI)
    cached = UsageMeter().record(completion(1_000_000, 0, cached=1_000_000), OPENAI, MINI)
    assert cached < full


def test_free_tariff_costs_nothing_rather_than_being_unknown():
    meter = UsageMeter()
    assert meter.record(completion(), FREE, FREE.models[0]) == 0.0
    assert meter.unpriced_requests == 0


def test_model_without_a_price_is_counted_separately():
    # Модель ищется по признаку, а не по месту в каталоге: каталог правят,
    # и тест не должен ломаться оттого, что у провайдера прибавилось моделей.
    unpriced = [(provider, model) for provider in PROVIDERS.values()
                for model in provider.models
                if not model.priced and not provider.free]
    assert unpriced, "в каталоге не осталось модели без цены — проверять нечего"
    provider, model = unpriced[0]
    meter = UsageMeter()
    assert meter.record(completion(), provider, model) is None
    assert meter.unpriced_requests == 1
    assert meter.costs == {}


def test_spending_is_split_by_purpose():
    meter = UsageMeter()
    meter.record(completion(100, 50), OPENAI, MINI, MAIN)
    meter.record(completion(120, 40), OPENAI, MINI, REPAIR)
    meter.record(completion(80, 20), OPENAI, MINI, JUDGE)
    assert meter.requests == 3
    assert meter.total_tokens == 410
    assert meter.by_kind[REPAIR].total_tokens == 160
    assert meter.snapshot()["by_kind"] == {MAIN: 150, REPAIR: 160, JUDGE: 100}


def test_absorb_merges_a_sub_agent_spending():
    main, judge = UsageMeter(), UsageMeter()
    main.record(completion(100, 50), OPENAI, MINI, MAIN)
    judge.record(completion(10, 5), OPENAI, MINI, JUDGE)
    main.absorb(judge)
    assert main.requests == 2
    assert main.by_kind[JUDGE].total_tokens == 15


def test_currencies_are_kept_apart():
    yandex = PROVIDERS["yandex"]
    meter = UsageMeter()
    meter.record(completion(), OPENAI, MINI)
    meter.record(completion(), yandex, yandex.models[0])
    assert set(meter.costs) <= {"USD", "RUB"}
    if "RUB" in meter.costs:
        assert meter.cost("USD") != meter.cost("RUB")


def test_limits_trigger_before_the_money_is_spent():
    meter = UsageMeter()
    meter.record(completion(1000, 1000), OPENAI, MINI)

    meter.check(Budget())                                   # без лимитов — молча
    with pytest.raises(BudgetExceeded):
        meter.check(Budget(max_tokens=2500), upcoming_tokens=1000)
    with pytest.raises(BudgetExceeded):
        meter.check(Budget(max_requests=1))
    with pytest.raises(BudgetExceeded):
        meter.check(Budget(max_cost=0.0001))


def test_cost_ceiling_stops_the_first_request_too():
    """Потолок ниже цены одного запроса обязан защищать, а не срабатывать после.

    Раньше проверка смотрела только на уже потраченное, поэтому первый запрос
    уходил при любом потолке — и лимит узнавал о трате, когда деньги ушли.
    """
    meter = UsageMeter()
    with pytest.raises(BudgetExceeded) as info:
        meter.check(Budget(max_cost=0.0000001), upcoming_tokens=1000,
                    upcoming_cost=0.0007)
    assert "будет превышен" in str(info.value)
    assert meter.requests == 0


def test_generous_ceiling_lets_the_request_through():
    UsageMeter().check(Budget(max_cost=1.0), upcoming_tokens=1000, upcoming_cost=0.0007)


def test_ceiling_still_triggers_on_what_is_already_spent():
    meter = UsageMeter()
    meter.record(completion(1_000_000, 1_000_000), OPENAI, MINI)
    with pytest.raises(BudgetExceeded):
        meter.check(Budget(max_cost=0.01))
