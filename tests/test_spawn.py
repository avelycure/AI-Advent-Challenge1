"""Проверка ментора: моментально заспавнить 100 агентов с разными конфигами.

Всё идёт на заглушке с нулевой паузой — ни одного обращения в сеть и ни одного
потраченного цента. Проверяется именно то, что раньше было невозможно: агенты
создаются и отвечают, не задав человеку ни одного вопроса.
"""
from __future__ import annotations

import time
from dataclasses import replace

import pytest

from llmagent import (
    AgentConfig,
    ClientRegistry,
    GenerationParams,
    HistoryConfig,
    InputPolicy,
    Transport,
)
from llmagent.spawn import ask_safely, build_agents, matrix, spawn_sync, summarize
from llmagent.transport import PROVIDER_ORDER, PROVIDERS

HUNDRED = 100
DEMO = Transport(demo=True, demo_delay=0.0)
BASE = AgentConfig(name="mass", transport=DEMO)


def hundred_configs():
    """Сто конфигов, различающихся моделью, температурой и промптом.

    Модели перебираются по всему каталогу восьми провайдеров: если бы агент
    втайне зависел от одного из них, пачка бы это показала.
    """
    catalogue = [(key, model.id) for key in PROVIDER_ORDER
                 for model in PROVIDERS[key].models]
    changes = []
    for number in range(HUNDRED):
        provider, model = catalogue[number % len(catalogue)]
        changes.append({
            "provider": provider,
            "model": model,
            "system_prompt": "Ты агент номер {}.".format(number),
            "generation": GenerationParams(temperature=round(number % 20 / 10, 2)),
            "history": HistoryConfig(enabled=number % 2 == 0),
        })
    return matrix(BASE, changes, name=lambda i, c: "agent-{:03d}".format(i + 1))


def test_a_hundred_agents_are_built_without_asking_anyone():
    configs = hundred_configs()
    agents = build_agents(configs, registry=ClientRegistry())
    assert len(agents) == HUNDRED
    assert len({a.name for a in agents}) == HUNDRED
    assert len({(a.config.provider, a.config.model) for a in agents}) > 1


def test_a_hundred_agents_answer_and_each_keeps_its_own_state():
    registry = ClientRegistry()
    started = time.perf_counter()
    results = spawn_sync(hundred_configs(), "Привет", concurrency=25, registry=registry)
    elapsed = time.perf_counter() - started

    assert len(results) == HUNDRED
    assert [r.agent for r in results] == ["agent-{:03d}".format(i + 1)
                                          for i in range(HUNDRED)]
    assert all(r.error is None for r in results), [r.error for r in results if r.error]
    assert all(r.text for r in results)
    # «Моментально» — это про порядок величины, а не про точное число.
    assert elapsed < 10.0, "сто агентов заняли {:.1f} с".format(elapsed)


def test_clients_are_shared_not_multiplied():
    """Сто агентов не должны открыть сто соединений: ключ у многих один."""
    registry = ClientRegistry()
    spawn_sync(hundred_configs(), "Привет", concurrency=25, registry=registry)
    assert len(registry) <= len(PROVIDER_ORDER)


def test_one_broken_config_does_not_take_down_the_batch():
    configs = [replace(BASE, name="ok-1"),
               replace(BASE, name="broken", model="нет-такой-модели"),
               replace(BASE, name="ok-2")]
    results = spawn_sync(configs, "Привет", registry=ClientRegistry())
    assert [r.error is None for r in results] == [True, False, True]
    assert "нет-такой-модели" in results[1].error


def test_one_rejected_input_does_not_take_down_the_batch():
    configs = [replace(BASE, name="ok"),
               replace(BASE, name="strict", input=InputPolicy(max_chars=2))]
    results = spawn_sync(configs, "довольно длинный запрос", registry=ClientRegistry())
    assert results[0].error is None
    assert results[1].error is not None and "длиннее" in results[1].error


def test_summary_counts_what_happened():
    results = spawn_sync(hundred_configs()[:10], "Привет", registry=ClientRegistry())
    summary = summarize(results)
    assert summary["agents"] == 10 and summary["answered"] == 10
    assert summary["failed"] == 0 and summary["valid"] == 10
    assert summary["total_tokens"] > 0


def test_configs_can_come_from_files(tmp_path):
    """Сто агентов описываются данными, а данные лежат в файлах."""
    paths = []
    for number in range(5):
        path = tmp_path / "agent-{}.yaml".format(number)
        replace(BASE, name="from-file-{}".format(number),
                generation=GenerationParams(temperature=number / 10)).to_file(str(path))
        paths.append(path)

    configs = [AgentConfig.from_file(str(p)) for p in paths]
    results = spawn_sync(configs, "Привет", registry=ClientRegistry())
    assert [r.agent for r in results] == ["from-file-{}".format(i) for i in range(5)]
    assert all(r.error is None for r in results)


def test_the_mentor_check_script_runs_and_reports():
    """Сам ./spawn.sh обязан работать: на него ссылается описание задания."""
    import subprocess
    import sys
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    finished = subprocess.run(
        [sys.executable, "spawn_demo.py", "--agents", "100", "--rows", "3"],
        cwd=str(root), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "COLUMNS": "120", "TERM": "dumb",
             "PYTHONPATH": str(root)})
    assert finished.returncode in (0, 1), finished.stderr[-2000:]
    assert "100 агентов" in finished.stdout
    assert "Ответили" in finished.stdout
    assert "Traceback" not in finished.stderr
