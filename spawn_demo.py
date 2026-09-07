#!/usr/bin/env python3
"""Сто агентов с разными конфигами — проверка, которую предложил ментор.

Каждый агент создаётся из своего конфига и отвечает на общий вопрос. Ни одного
вопроса человеку не задаётся: реквизиты берутся из конфига или из окружения,
а всё остальное поведение описано данными.

По умолчанию работает заглушка: сеть не трогается, деньги не тратятся. Живой
прогон включается явным ``--live`` и стоит настоящих денег — столько запросов,
сколько агентов.

Запуск:

    ./spawn.sh                    сто агентов на заглушке
    ./spawn.sh --agents 300       больше агентов
    ./spawn.sh --config configs/frugal.yaml   своё основание для набора
    ./spawn.sh --live             живые запросы (платно!)
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmagent import (
    AgentConfig,
    AgentResult,
    Budget,
    ClientRegistry,
    GenerationParams,
    HistoryConfig,
    InputPolicy,
    OutputPolicy,
    Transport,
)
from llmagent.spawn import matrix, spawn_sync, summarize
from llmagent.transport import PROVIDER_ORDER, PROVIDERS
from llmchat.ui import fmt, make_console, plural

QUESTION = "Назови одно свойство хорошего кода и объясни его одним предложением."

# Отличия, которые перебираются по кругу. Каждое — поле конфига, а не ветка
# в коде: в этом вся суть требования «всё за неделю в виде конфига агента».
ROLES: List[Dict[str, object]] = [
    {"system_prompt": "Ты архитектор. Отвечай о структуре и границах."},
    {"system_prompt": "Ты ревьюер. Отвечай о читаемости и сопровождении."},
    {"system_prompt": "Ты тестировщик. Отвечай о проверяемости."},
    {"system_prompt": "Ты новичок. Отвечай простыми словами."},
]
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
SHAPES: List[Dict[str, object]] = [
    {"output": OutputPolicy()},
    {"output": OutputPolicy(format="json", max_attempts=2)},
    {"input": InputPolicy(max_chars=300, on_too_long="trim")},
    {"budget": Budget(max_tokens=20_000, max_requests=4)},
    {"history": HistoryConfig(enabled=False)},
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spawn.sh", description="Массовый запуск агентов с разными конфигами")
    parser.add_argument("--agents", type=int, default=100, help="сколько агентов (по умолчанию 100)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="сколько запросов одновременно")
    parser.add_argument("--question", default=QUESTION, help="общий вопрос всем агентам")
    parser.add_argument("--config", metavar="ФАЙЛ",
                        help="основание набора: конфиг, поверх которого идут отличия")
    parser.add_argument("--providers", default="",
                        help="через запятую: каких провайдеров перебирать "
                             "(по умолчанию — только модель по умолчанию)")
    parser.add_argument("--live", action="store_true",
                        help="настоящие запросы к провайдерам — ПЛАТНО")
    parser.add_argument("--rows", type=int, default=12,
                        help="сколько строк показать в таблице")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> List[AgentConfig]:
    """Набор конфигов: одно основание плюс перебор отличий по кругу."""
    base = AgentConfig.from_file(args.config) if args.config else AgentConfig()
    if not args.live:
        # Нулевая пауза: проверка про «моментально», а не про то, как долго
        # заглушка изображает раздумье.
        base = base.with_changes(transport=Transport(demo=True, demo_delay=0.0))

    if args.providers:
        wanted = [key.strip() for key in args.providers.split(",") if key.strip()]
        pairs = [(key, model.id) for key in wanted for model in PROVIDERS[key].models]
    elif args.live:
        # Живой прогон по всем провайдерам потребовал бы восьми ключей, поэтому
        # без явного указания берётся одна модель основания.
        pairs = [(base.provider, base.model)]
    else:
        pairs = [(key, model.id) for key in PROVIDER_ORDER
                 for model in PROVIDERS[key].models]

    changes = []
    for number in range(args.agents):
        provider, model = pairs[number % len(pairs)]
        change: Dict[str, object] = {"provider": provider, "model": model}
        change.update(ROLES[number % len(ROLES)])
        change.update(SHAPES[number % len(SHAPES)])
        change["generation"] = GenerationParams(
            temperature=TEMPERATURES[number % len(TEMPERATURES)], max_tokens=200)
        changes.append(change)

    return matrix(base, changes, name=lambda i, c: "agent-{:03d}".format(i + 1))


def plan_panel(configs: List[AgentConfig], args: argparse.Namespace) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Агентов", str(len(configs)))
    table.add_row("Различаются", "провайдер и модель · роль в системном промпте · "
                                 "температура · политики и бюджет")
    table.add_row("Моделей в наборе", str(len({(c.provider, c.model) for c in configs})))
    table.add_row("Одновременно", str(args.concurrency))
    table.add_row("Режим", "[red]живые запросы — платно[/]" if args.live
                  else "[green]заглушка: сеть не трогается, денег не тратится[/]")
    table.add_row("Вопрос", args.question)
    return Panel(table, title="🧪 Проверка: заспавнить агентов с разными конфигами",
                 title_align="left", border_style="cyan", box=box.ROUNDED, padding=(0, 1))


def results_table(results: List[AgentResult], rows: int) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("Агент", style="bold", no_wrap=True)
    table.add_column("Модель", no_wrap=True)
    table.add_column("t°", justify="right", style="dim")
    table.add_column("Ток.", justify="right", style="dim")
    table.add_column("Итог", justify="center")
    table.add_column("Ответ")

    shown = results[:rows]
    for result, config in shown:
        if result.error:
            verdict = Text("✗", style="red")
            answer = Text(result.error[:70], style="red")
        else:
            verdict = Text("✓", style="green") if result.ok else Text("~", style="yellow")
            answer = Text(" ".join(result.text.split())[:70])
        table.add_row(result.agent, result.model,
                      "{:.1f}".format(config.generation.temperature),
                      fmt(result.total_tokens), verdict, answer)

    tail = Text("\n… и ещё {} {}".format(
        len(results) - len(shown),
        plural(len(results) - len(shown), ("агент", "агента", "агентов"))), style="dim")
    return Group(table, tail) if len(results) > len(shown) else table


def summary_panel(results: List[AgentResult], elapsed: float,
                  registry: ClientRegistry) -> RenderableType:
    summary = summarize(results)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Создано и опрошено", "{} {}".format(
        summary["agents"], plural(summary["agents"], ("агент", "агента", "агентов"))))
    table.add_row("Ответили", "[green]{}[/]".format(summary["answered"]))
    if summary["failed"]:
        table.add_row("Не смогли", "[red]{}[/] — {}".format(
            summary["failed"], "; ".join(summary["errors"])[:120]))
    strict = summary["answered"] - summary["valid"]
    table.add_row("Прошли выходную политику", str(summary["valid"]))
    if strict:
        table.add_row("Форму не соблюли", "{} — это агенты с политикой JSON: "
                                          "заглушка отвечает прозой, и переспрос "
                                          "ей не помогает".format(strict))
    table.add_row("Токенов всего", fmt(summary["total_tokens"]))
    table.add_row("Стоимость", "${:.4f}".format(summary["cost"]))
    table.add_row("Соединений открыто", "{} на {} агентов".format(
        len(registry), summary["agents"]))
    table.add_row("Времени с начала до конца", "{:.2f} с".format(elapsed))
    table.add_row("В среднем на агента", "{:.3f} с".format(elapsed / max(1, len(results))))

    hint = Text.from_markup(
        "\n[dim]Ни одного вопроса человеку не задано: каждый агент собран из своего "
        "конфига.\nЭто и есть ответ на «можно ли моментально заспавнить сто агентов "
        "с разными конфигами».[/]")
    return Panel(Group(table, hint), title="📊 Итог", title_align="left",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def confirm_live(console: Console, count: int) -> bool:
    console.print(Panel(
        Text.from_markup(
            "Будет отправлено [bold]{}[/] настоящих запросов к провайдерам. "
            "Это стоит денег.\nБез флага живого режима тот же прогон идёт на "
            "заглушке и бесплатен.".format(count)),
        title="⚠ Платный прогон", border_style="red", box=box.ROUNDED))
    try:
        return input("Продолжить? [y/N]: ").strip().lower() in ("y", "yes", "д", "да")
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    console = make_console()

    try:
        configs = build_configs(args)
    except Exception as exc:  # noqa: BLE001 — сообщение уже человеческое
        console.print(Panel(Text(str(exc), style="red"), title="⚠ Конфиг",
                            border_style="red", box=box.ROUNDED))
        return 2

    console.print()
    console.print(plan_panel(configs, args))
    if args.live and not confirm_live(console, len(configs)):
        console.print("[dim]Отменено — денег не потрачено.[/]")
        return 0

    registry = ClientRegistry()
    started = time.perf_counter()
    with console.status("[bold]Спавню {} агентов…[/]".format(len(configs)), spinner="dots"):
        results = spawn_sync(configs, args.question, concurrency=args.concurrency,
                             registry=registry)
    elapsed = time.perf_counter() - started

    console.print(results_table(list(zip(results, configs)), args.rows))
    console.print(summary_panel(results, elapsed, registry))
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
