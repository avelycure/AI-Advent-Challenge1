"""Массовый запуск агентов с разными конфигами.

Проверка, ради которой всё и переставлялось: сто агентов, у каждого свой
конфиг, все создаются и отвечают без единого вопроса человеку.

Клиент провайдера синхронный, поэтому каждый агент работает в своём потоке,
а ``asyncio`` только собирает результаты. Это дешевле, чем переписывать
транспорт на асинхронный: потоки почти всё время ждут сеть. Одновременность
ограничена семафором — сто одновременных запросов провайдер встретит отказом
по частоте, и это была бы не проверка агента, а проверка лимитов.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .agent import Agent
from .config import AgentConfig
from .errors import AgentError, LLMError
from .registry import ClientRegistry
from .result import AgentResult

DEFAULT_CONCURRENCY = 20


def build_agents(configs: Sequence[AgentConfig], *,
                 registry: Optional[ClientRegistry] = None) -> List[Agent]:
    """Создать агентов по конфигам. Негодный конфиг падает сразу и по имени."""
    return [Agent(config, registry=registry) for config in configs]


def matrix(base: AgentConfig, changes: Iterable[Dict[str, Any]],
           name: Callable[[int, Dict[str, Any]], str] = None) -> List[AgentConfig]:
    """Набор конфигов из одного основания и списка отличий.

    Так набор из ста агентов описывается списком того, чем они различаются,
    а не сотней копий целого конфига.
    """
    configs: List[AgentConfig] = []
    for index, change in enumerate(changes):
        label = name(index, change) if name else "{}-{}".format(base.name, index + 1)
        configs.append(replace(base, name=label, **change))
    return configs


async def spawn(configs: Sequence[AgentConfig], question: str, *,
                concurrency: int = DEFAULT_CONCURRENCY,
                registry: Optional[ClientRegistry] = None) -> List[AgentResult]:
    """Задать один вопрос всем конфигам. Порядок результатов — как у конфигов.

    Падение одного агента не роняет пачку: его беда попадает в ``error``
    своего результата, остальные доводятся до конца.
    """
    limit = asyncio.Semaphore(max(1, concurrency))

    async def run(config: AgentConfig) -> AgentResult:
        async with limit:
            return await asyncio.to_thread(ask_safely, config, question, registry)

    return list(await asyncio.gather(*(run(config) for config in configs)))


def spawn_sync(configs: Sequence[AgentConfig], question: str, *,
               concurrency: int = DEFAULT_CONCURRENCY,
               registry: Optional[ClientRegistry] = None) -> List[AgentResult]:
    """Тот же спавн для вызывающего, который не живёт в событийном цикле."""
    return asyncio.run(spawn(configs, question, concurrency=concurrency, registry=registry))


def ask_safely(config: AgentConfig, question: str,
               registry: Optional[ClientRegistry] = None) -> AgentResult:
    """Один агент от создания до ответа, с ошибкой вместо исключения."""
    try:
        agent = Agent(config, registry=registry)
    except (AgentError, ValueError) as exc:
        return AgentResult(agent=config.name, provider=config.provider,
                           model=config.model, error=str(exc))
    try:
        return agent.ask(question)
    except (AgentError, LLMError) as exc:
        return AgentResult(agent=agent.name, provider=config.provider,
                           model=config.model, error=str(exc))


def summarize(results: Sequence[AgentResult]) -> Dict[str, Any]:
    """Сводка по пачке: сколько ответило, во что обошлось, что сломалось."""
    done = [r for r in results if r.error is None]
    return {
        "agents": len(results),
        "answered": len(done),
        "failed": len(results) - len(done),
        "valid": sum(1 for r in done if r.ok),
        "total_tokens": sum(r.total_tokens for r in done),
        "seconds": round(sum(r.seconds for r in done), 3),
        "cost": round(sum(r.cost or 0.0 for r in done), 6),
        "errors": sorted({r.error for r in results if r.error}),
    }
