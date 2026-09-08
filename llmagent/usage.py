"""Учёт расхода: токены, время, деньги и лимиты.

Счётчики живут в агенте, а не в интерфейсе, потому что расход — свойство
работы агента, а не картинки на экране. Точные цифры приходят от API в поле
``usage`` каждого ответа; локальная оценка нужна только для ещё не
отправленного текста.

Запросы делятся по назначению: основной ответ, переспрос после нарушения
политики, оценка судьёй и служебные вроде темы диалога. Без этого разбиения
непонятно, за что заплачено: короткий ответ с судьёй и двумя переспросами
стоит дороже длинного ответа с первого раза.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from .errors import BudgetExceeded
from .transport import ModelInfo, ProviderInfo, request_cost

MAIN = "основной"
REPAIR = "переспрос"
JUDGE = "оценка"
SIDE = "служебный"
SUB = "под-агент"

KINDS = (MAIN, REPAIR, JUDGE, SIDE, SUB)


@dataclass
class Spent:
    """Расход по одному виду запросов."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageMeter:
    """Расход агента за всё время его жизни."""

    by_kind: Dict[str, Spent] = field(default_factory=lambda: {k: Spent() for k in KINDS})
    # По каждой валюте отдельно: рубли с долларами не складываются.
    costs: Dict[str, float] = field(default_factory=dict)
    # Запросы к моделям, цена которых не задана: без этого счётчика итоговая
    # сумма выглядела бы полной, хотя часть расхода в неё не вошла.
    unpriced_requests: int = 0

    def record(self, completion, provider: ProviderInfo, model: ModelInfo,
               kind: str = MAIN) -> Optional[float]:
        """Записать один ответ провайдера. Возвращает стоимость запроса."""
        spent = self.by_kind.setdefault(kind, Spent())
        spent.requests += 1
        spent.prompt_tokens += completion.prompt_tokens
        spent.completion_tokens += completion.completion_tokens
        spent.reasoning_tokens += completion.reasoning_tokens
        spent.seconds += completion.elapsed

        cost = request_cost(provider, model, completion.prompt_tokens,
                            completion.completion_tokens, completion.cached_tokens)
        if cost is None:
            self.unpriced_requests += 1
        else:
            self.costs[model.currency] = self.costs.get(model.currency, 0.0) + cost
        return cost

    def record_external(self, prompt_tokens: int, completion_tokens: int,
                        seconds: float = 0.0, cost: Optional[float] = None,
                        currency: str = "USD", kind: str = SUB,
                        requests: int = 1) -> None:
        """Учесть расход, случившийся вне этого процесса.

        Под-агент живёт отдельным процессом, и объекта ``Completion`` у нас
        на руках нет — приходят только числа из его ответа. Не учитывать их
        значило бы соврать в итоговой цене: платил тот же кошелёк.
        """
        spent = self.by_kind.setdefault(kind, Spent())
        spent.requests += requests
        spent.prompt_tokens += prompt_tokens
        spent.completion_tokens += completion_tokens
        spent.seconds += seconds
        if cost is None:
            self.unpriced_requests += requests
        else:
            self.costs[currency] = self.costs.get(currency, 0.0) + cost

    def absorb(self, other: "UsageMeter") -> None:
        """Вобрать расход другого счётчика — например, судьи со своей моделью."""
        for kind, spent in other.by_kind.items():
            mine = self.by_kind.setdefault(kind, Spent())
            mine.requests += spent.requests
            mine.prompt_tokens += spent.prompt_tokens
            mine.completion_tokens += spent.completion_tokens
            mine.reasoning_tokens += spent.reasoning_tokens
            mine.seconds += spent.seconds
        for currency, amount in other.costs.items():
            self.costs[currency] = self.costs.get(currency, 0.0) + amount
        self.unpriced_requests += other.unpriced_requests

    # --- итоги ---------------------------------------------------------
    def _sum(self, attribute: str):
        return sum(getattr(spent, attribute) for spent in self.by_kind.values())

    @property
    def requests(self) -> int:
        return self._sum("requests")

    @property
    def prompt_tokens(self) -> int:
        return self._sum("prompt_tokens")

    @property
    def completion_tokens(self) -> int:
        return self._sum("completion_tokens")

    @property
    def reasoning_tokens(self) -> int:
        return self._sum("reasoning_tokens")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def seconds(self) -> float:
        return self._sum("seconds")

    @property
    def avg_seconds(self) -> float:
        return self.seconds / self.requests if self.requests else 0.0

    def cost(self, currency: str = "USD") -> float:
        return self.costs.get(currency, 0.0)

    # --- лимиты --------------------------------------------------------
    def check(self, budget, upcoming_tokens: int = 0,
              upcoming_cost: Optional[float] = None) -> None:
        """Не пора ли остановиться. Проверка до запроса, а не после траты."""
        if budget.max_requests is not None and self.requests >= budget.max_requests:
            raise BudgetExceeded("исчерпан лимит запросов: {} из {}".format(
                self.requests, budget.max_requests))
        if budget.max_tokens is not None:
            planned = self.total_tokens + upcoming_tokens
            if planned > budget.max_tokens:
                raise BudgetExceeded(
                    "лимит токенов {} будет превышен: потрачено {}, в запросе ещё ~{}".format(
                        budget.max_tokens, self.total_tokens, upcoming_tokens))
        if budget.max_cost is not None:
            spent = max(self.costs.values()) if self.costs else 0.0
            # Прикидку стоимости учитываем намеренно. Без неё потолок ниже цены
            # одного запроса ни от чего не защищал: первый запрос уходил всегда,
            # и лимит срабатывал уже после того, как деньги были потрачены.
            planned = spent + (upcoming_cost or 0.0)
            if planned > budget.max_cost:
                raise BudgetExceeded(
                    "денежный лимит {:.6f} будет превышен: потрачено {:.6f}, "
                    "этот запрос обойдётся примерно в {:.6f}".format(
                        budget.max_cost, spent, upcoming_cost or 0.0))
            if spent >= budget.max_cost:
                raise BudgetExceeded("исчерпан денежный лимит: {:.6f} из {:.6f}".format(
                    spent, budget.max_cost))

    # --- сохранение ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Полный вид счётчиков — в отличие от ``snapshot``, восстановимый."""
        return {
            "by_kind": {kind: asdict(spent) for kind, spent in self.by_kind.items()
                        if spent.requests},
            "costs": dict(self.costs),
            "unpriced_requests": self.unpriced_requests,
        }

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "UsageMeter":
        meter = cls()
        known = {f.name for f in fields(Spent)}
        for kind, spent in (payload.get("by_kind") or {}).items():
            meter.by_kind[kind] = Spent(**{k: v for k, v in spent.items() if k in known})
        meter.costs = dict(payload.get("costs") or {})
        meter.unpriced_requests = int(payload.get("unpriced_requests", 0) or 0)
        return meter

    def snapshot(self) -> Dict[str, object]:
        """Плоская сводка — для отчётов, тестов и тела HTTP-ответа."""
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
            "costs": {currency: round(amount, 6) for currency, amount in self.costs.items()},
            "unpriced_requests": self.unpriced_requests,
            "by_kind": {kind: spent.total_tokens for kind, spent in self.by_kind.items()
                        if spent.requests},
        }
