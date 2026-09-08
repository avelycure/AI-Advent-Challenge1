#!/usr/bin/env python3
"""Живая проверка: настоящие запуски агента из терминала, а не модульные тесты.

Модульные тесты проверяют части. Здесь проверяется то, что человек делает
руками: запускает команду с разными параметрами, разговаривает, зовёт
под-агента, возвращается в сессию. Каждый сценарий привязан к требованию,
чтобы было видно, что именно он подтверждает.

По умолчанию всё идёт на заглушке: сеть не трогается, денег не тратится.
С ``--live`` те же сценарии идут в настоящий API — это платно, и программа
сначала спросит подтверждение.

    ./livecheck.py                 бесплатно, на заглушке
    ./livecheck.py --live          настоящие запросы (платно)
    ./livecheck.py --only сессии   только сценарии, чьё имя содержит слово
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = pathlib.Path(__file__).resolve().parent
ENTRY = ROOT / "chat.py"
TIMEOUT = 300.0

# Что засчитывается только в живом режиме: на заглушке эти проверки
# бессмысленны, потому что она отвечает заранее заготовленной прозой.
LIVE_ONLY = "требует живой модели"

# Бесплатные тарифы отвечают отказом 429, если стрелять быстро. Это про тариф,
# а не про агента, поэтому проверка ждёт и пробует снова.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_PAUSE = 8.0


def flat(text: str) -> str:
    """Текст без переносов и лишних пробелов.

    Панели ``rich`` переносят строки по ширине окна, поэтому искать в выводе
    целую фразу можно только так: иначе проверка ломается от смены ширины.
    """
    return " ".join(text.split())


@dataclass
class Outcome:
    ok: Optional[bool]          # None — проверка неприменима в этом режиме
    detail: str = ""


@dataclass
class Scenario:
    number: int
    requirement: str
    title: str
    run: Callable[["Harness"], Outcome]
    live_only: bool = False


class Harness:
    """Запуск настоящей программы в отдельном домашнем каталоге."""

    def __init__(self, live: bool, console: Console, provider: str = "",
                 model: str = "") -> None:
        self.live = live
        self.console = console
        self.provider = provider
        self.model = model
        self.throttled = 0
        self.homes: List[pathlib.Path] = []
        self.home = self._new_home()
        # Реквизиты в живом режиме нужны настоящие, поэтому окружение
        # наследуется целиком, а подменяется только домашний каталог —
        # чтобы проверка не трогала настоящие сессии пользователя.
        self.environment = dict(os.environ)
        self.environment["HOME"] = str(self.home)
        self.environment["COLUMNS"] = "150"
        self.environment["TERM"] = "dumb"
        if live:
            # Ключ ищется в настоящем домашнем каталоге, а не в подменённом.
            for name, path in real_key_files().items():
                self.environment.setdefault(name, path)

    @property
    def free_tariff(self) -> bool:
        """Бесплатный тариф: цена запроса нулевая, и потолку трат нечего ловить."""
        from llmagent.transport import PROVIDERS

        if not self.live:
            return False
        return PROVIDERS[self.provider].free

    def other_model(self) -> str:
        """Ещё одна модель того же провайдера — для проверки правок конфига."""
        from llmagent.transport import PROVIDERS

        if not self.live:
            return "gpt-5.4"
        models = [m.id for m in PROVIDERS[self.provider].models if m.id != self.model]
        return models[0] if models else self.model

    def _new_home(self) -> pathlib.Path:
        home = pathlib.Path(tempfile.mkdtemp(prefix="livecheck-home-"))
        self.homes.append(home)
        return home

    def fresh_home(self) -> None:
        """Свой домашний каталог каждому сценарию.

        Общий каталог связывал сценарии между собой: сохранённая раньше сессия
        превращала проверку «возвращаться некуда» в успешный запуск диалога.
        """
        self.home = self._new_home()
        self.environment["HOME"] = str(self.home)

    def cleanup(self) -> None:
        for home in self.homes:
            shutil.rmtree(home, ignore_errors=True)

    @property
    def base(self) -> List[str]:
        """Общее начало команды.

        Конфиг задаётся всегда, иначе программа начнёт расспрос о провайдере и
        модели и съест сценарий, поданный в ввод. Сценарий может назвать свой
        конфиг — он идёт позже и потому побеждает, а ``--demo`` держит
        заглушку даже для конфигов, в которых её нет.
        """
        if self.live:
            # Конфиг задаём всегда, иначе начнётся расспрос. Провайдера и
            # модель — флагами: они перебивают конфиг, поэтому сценарий может
            # взять свой конфиг из configs/, а модель останется заданной здесь.
            return ["--config", "default", "--provider", self.provider,
                    "--model", self.model]
        return ["--demo", "--config", "demo", "--set", "transport.demo_delay=0"]

    def run(self, *arguments: str, script: str = "", check: bool = False):
        """Один запуск программы. При отказе по частоте — пауза и повтор.

        Бесплатные тарифы ограничивают частоту, и сценарии стреляют быстрее,
        чем те разрешают. Отказ 429 — это про тариф, а не про агента, поэтому
        обвязка ждёт и пробует снова, а не записывает провал.
        """
        command = [sys.executable, str(ENTRY), *self.base, *arguments]
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            finished = subprocess.run(command, cwd=str(ROOT), input=script,
                                      capture_output=True, text=True,
                                      env=self.environment, timeout=TIMEOUT)
            whole = finished.stdout + finished.stderr
            if not (self.live and "429" in whole and attempt < RATE_LIMIT_RETRIES):
                break
            self.throttled += 1
            time.sleep(RATE_LIMIT_PAUSE)
        if check and finished.returncode != 0:
            raise AssertionError("код {}: {}".format(
                finished.returncode, (finished.stdout + finished.stderr)[-400:]))
        return finished

    def ask_json(self, question: str, *options: str, allow_failure: bool = False) -> dict:
        """Один вопрос с разбором ответа. Вопрос отдельно от флагов: иначе он
        рискует занять место значения соседнего флага.

        Неудачный ответ поднимается как понятная ошибка, а не превращается в
        ``KeyError`` на отсутствующем поле: причина должна быть видна сразу.
        """
        finished = self.run(*options, "--ask", question, "--json")
        try:
            payload = json.loads(finished.stdout)
        except ValueError:
            raise AssertionError("не JSON: {}".format(
                (finished.stdout + finished.stderr)[-300:]))
        if not allow_failure and not payload.get("ok"):
            raise AssertionError("запрос не удался: {}".format(
                payload.get("error", "без объяснения"))[:200])
        return payload

    def sessions(self) -> List[pathlib.Path]:
        return sorted((self.home / ".llm-agent" / "sessions").glob("*.json"))


def real_key_files() -> dict:
    """Переменные окружения с ключами, собранные из файлов настоящего дома."""
    from llmagent.transport import PROVIDER_ORDER, PROVIDERS, find_sources

    found = {}
    for key in PROVIDER_ORDER:
        provider = PROVIDERS[key]
        sources = find_sources(provider.api_key_env, provider.key_files)
        if sources:
            found[provider.api_key_env] = sources[0].value
    return found


# --------------------------------------------------------------------------
# Сценарии
# --------------------------------------------------------------------------

SCENARIOS: List[Scenario] = []


def scenario(requirement: str, title: str, live_only: bool = False):
    def register(function):
        SCENARIOS.append(Scenario(len(SCENARIOS) + 1, requirement, title,
                                  function, live_only))
        return function
    return register


@scenario("1-4", "разговор в терминале: вопрос, ответ, счётчики на экране")
def talk(harness: Harness) -> Outcome:
    shown = flat(harness.run(script="Назови три языка программирования\n/exit\n",
                             check=True).stdout)
    marks = ["🧑 Вы" in shown or "Вы ›" in shown, "🤖" in shown,
             "Контекст" in shown, "потрачено" in shown]
    return Outcome(all(marks), "панели вопроса, ответа, окна и расхода: {}/4".format(
        sum(marks)))


@scenario("5", "запрос ушёл по HTTP в настоящий API", live_only=True)
def http(harness: Harness) -> Outcome:
    payload = harness.ask_json("Ответь одним словом: столица Франции")
    return Outcome(bool(payload.get("ok")) and payload["total_tokens"] > 0,
                   "модель {} · {} токенов · ${:.6f}".format(
                       payload.get("model"), payload.get("total_tokens"),
                       payload.get("cost") or 0.0))


@scenario("6-8", "агент как сущность: один вопрос — разобранный результат")
def one_shot(harness: Harness) -> Outcome:
    payload = harness.ask_json("Привет")
    needed = ("ok", "text", "session_id", "model", "prompt_tokens", "usage")
    missing = [name for name in needed if name not in payload]
    return Outcome(not missing and bool(payload["text"]),
                   "в ответе {} полей, сессия {}".format(len(payload),
                                                         payload.get("session_id")))


@scenario("9", "входная политика отклоняет запрос до траты денег")
def input_policy(harness: Harness) -> Outcome:
    finished = harness.run("--set", "input.max_chars=20", "--ask", "х" * 50, "--json")
    payload = json.loads(finished.stdout)
    stopped = finished.returncode == 1 and not payload["ok"]
    return Outcome(stopped and "длиннее" in payload.get("error", ""),
                   "код {} · {}".format(finished.returncode,
                                        payload.get("error", "")[:60]))


@scenario("10", "выходная политика: JSON по схеме, переспрос при нарушении")
def output_policy(harness: Harness) -> Outcome:
    payload = harness.ask_json("Книги Нассима Талеба", "--config", "books-json",
                               allow_failure=True)
    checks = payload.get("checks") or []
    if not harness.live:
        # Заглушка отвечает прозой, поэтому проверяем сам механизм: проверки
        # выполнены и переспрос состоялся.
        return Outcome(bool(checks) and payload["attempts"] > 1,
                       "проверок {} · попыток {} (заглушка форму не соблюдает)".format(
                           len(checks), payload["attempts"]))
    passed = sum(1 for item in checks if item["passed"])
    return Outcome(payload["ok"] and isinstance(payload.get("data"), dict),
                   "проверок пройдено {}/{} · попыток {}".format(
                       passed, len(checks), payload["attempts"]))


@scenario("11", "судья оценивает ответ отдельной моделью")
def judge(harness: Harness) -> Outcome:
    options = ["--config", "reviewer", "--set", "judge.max_tokens=150"]
    if harness.live:
        # Судья в конфиге назван отдельным агентом со своей моделью, и флаги
        # его не касаются — иначе он ушёл бы к другому провайдеру.
        options += ["--set", "judge.agent.provider=" + harness.provider,
                    "--set", "judge.agent.model=" + harness.model]
    payload = harness.ask_json("Чем полезна неизменяемость данных?", *options)
    scores = payload.get("scores") or {}
    if not harness.live:
        return Outcome(None, "заглушка не отвечает по форме оценщика — {}".format(
            LIVE_ONLY))
    return Outcome(len(scores) == 3, "оценки: {}".format(
        ", ".join("{}={}".format(k, v) for k, v in scores.items()) or "нет"))


@scenario("12", "в stdout режима --json нет ничего, кроме JSON")
def clean_stdout(harness: Harness) -> Outcome:
    finished = harness.run("--ask", "Привет", "--json")
    first = finished.stdout.lstrip()[:1]
    single = finished.stdout.strip().count("\n") == 0
    return Outcome(first == "{" and single,
                   "вывод начинается с «{}», строк {}".format(
                       first, finished.stdout.strip().count("\n") + 1))


@scenario("13", "подсчёт токенов и денег внутри агента")
def accounting(harness: Harness) -> Outcome:
    payload = harness.ask_json("Привет")
    usage = payload.get("usage") or {}
    marks = [usage.get("prompt_tokens", 0) > 0, usage.get("completion_tokens", 0) > 0,
             usage.get("total_tokens") == payload["total_tokens"],
             payload.get("cost") is not None]
    return Outcome(all(marks), "вход {} · выход {} · ${:.6f}".format(
        usage.get("prompt_tokens"), usage.get("completion_tokens"),
        payload.get("cost") or 0.0))


@scenario("14", "конфиг задаётся при запуске: слои и их порядок")
def layers(harness: Harness) -> Outcome:
    shown = harness.run("--config", "frugal", "--temperature", "0.2",
                        "--set", "generation.temperature=0.9",
                        "--set", "attempts=3", "--show-config", check=True).stdout
    rows = dict(line.split()[1:3] for line in shown.splitlines()
                if line.startswith("│ ") and len(line.split()) > 2
                and "." in line.split()[1])
    marks = [rows.get("generation.temperature") == "0.9",     # --set победил флаг
             rows.get("output.max_attempts") == "3",
             rows.get("budget.max_cost") == "0.05",           # из файла
             rows.get("input.max_chars") == "500"]
    return Outcome(all(marks), "температура {} · переспросы {} · бюджет {}".format(
        rows.get("generation.temperature"), rows.get("output.max_attempts"),
        rows.get("budget.max_cost")))


@scenario("15", "сто агентов с разными конфигами")
def hundred(harness: Harness) -> Outcome:
    started = time.perf_counter()
    finished = subprocess.run([sys.executable, str(ROOT / "spawn_demo.py"),
                               "--agents", "100", "--rows", "3"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              env=harness.environment, timeout=TIMEOUT)
    elapsed = time.perf_counter() - started
    return Outcome("Ответили" in finished.stdout and finished.returncode in (0, 1),
                   "100 агентов за {:.2f} с".format(elapsed))


@scenario("сессии", "два запуска не делят память")
def isolation(harness: Harness) -> Outcome:
    first = harness.ask_json("Привет, меня зовут Влад")
    second = harness.ask_json("Как меня зовут?")
    separate = first["session_id"] != second["session_id"]
    fresh = second["usage"]["requests"] == 1
    if harness.live:
        knows = "лад" in second["text"]
        return Outcome(separate and fresh and not knows,
                       "сессии разные, второй ответ: «{}»".format(second["text"][:70]))
    return Outcome(separate and fresh,
                   "сессии {} и {}, у второй один запрос и пустая память".format(
                       first["session_id"], second["session_id"]))


@scenario("сессии", "возврат в разговор помнит сказанное")
def resume(harness: Harness) -> Outcome:
    harness.run(script="Привет, меня зовут Влад\n/exit\n", check=True)
    files = harness.sessions()
    if not files:
        return Outcome(False, "сессия не сохранилась")
    shown = flat(harness.run("--continue", script="/history\n/exit\n",
                             check=True).stdout)
    saved = json.loads(files[-1].read_text(encoding="utf-8"))
    return Outcome("Возвращаюсь в сессию" in shown and "Влад" in shown
                   and saved["config"]["api_key"] is None,
                   "сессия {} · ключа в файле нет · права {}".format(
                       saved["session_id"], oct(files[-1].stat().st_mode)[-3:]))


@scenario("сессии", "вопрос в контексте сохранённой сессии")
def continue_ask(harness: Harness) -> Outcome:
    harness.run(script="Привет, меня зовут Влад\n/exit\n", check=True)
    fresh = harness.ask_json("Как меня зовут?")
    carried = json.loads(harness.run("--continue", "--ask", "Как меня зовут?",
                                     "--json").stdout)
    return Outcome(carried["prompt_tokens"] > fresh["prompt_tokens"],
                   "во входе {} токенов против {} без памяти".format(
                       carried["prompt_tokens"], fresh["prompt_tokens"]))


@scenario("под-агент", "вызов под-агента из диалога и возврат итога")
def subagent(harness: Harness) -> Outcome:
    shown = flat(harness.run(script="/agent frugal max_tokens=120 -- Что такое хороший код\n"
                                    "/stats\n/exit\n", check=True).stdout)
    marks = ["⤷ Под-агент frugal" in shown,
             "своя память, отдельный процесс" in shown,
             "В памяти: итог под-агента" in shown or "принят в память" in shown,
             "под-агент —" in shown]
    return Outcome(all(marks), "панель, своя сессия, итог в памяти, "
                               "расход отдельной строкой: {}/4".format(sum(marks)))


@scenario("под-агент", "у под-агента своя сессия и свой конфиг")
def subagent_isolated(harness: Harness) -> Outcome:
    # Модель без провайдера сменить нельзя: конфиг frugal назван у OpenAI, и
    # модель другого провайдера он справедливо отвергнет.
    tweaks = "model=" + harness.other_model()
    if harness.live:
        tweaks = "provider={} ".format(harness.provider) + tweaks
    shown = flat(harness.run(
        script="/agent frugal {} -- Коротко о неизменяемости\n/exit\n".format(tweaks),
        check=True).stdout)
    ok = harness.other_model() in shown and "своя память" in shown
    return Outcome(ok, "под-агент отвечал моделью {}, заданной правкой".format(
        harness.other_model()) if ok else "правка не применилась: " + shown[-140:])


@scenario("бюджет", "потолок трат останавливает запрос")
def budget(harness: Harness) -> Outcome:
    shown = flat(harness.run("--set", "budget.max_requests=1",
                             script="раз\nдва\n/exit\n", check=True).stdout)
    return Outcome("лимит запросов" in shown, "второй запрос остановлен лимитом")


@scenario("бюджет", "потолок денег не даёт уйти даже первому запросу")
def cost_ceiling(harness: Harness) -> Outcome:
    """Потолок ниже цены запроса обязан защищать заранее, а не после траты."""
    if harness.free_tariff:
        return Outcome(None, "у бесплатного тарифа цена запроса нулевая, "
                             "и потолку трат нечего останавливать")
    payload = harness.ask_json("Привет", "--max-cost", "0.0000001", allow_failure=True)
    stopped = not payload["ok"] and "будет превышен" in payload.get("error", "")
    allowed = harness.ask_json("Привет", "--max-cost", "1.0")
    return Outcome(stopped and allowed["ok"],
                   "малый потолок остановил, щедрый пропустил (${:.6f})".format(
                       allowed.get("cost") or 0.0))


@scenario("параметры", "негодное значение отвергается с понятным текстом")
def bad_values(harness: Harness) -> Outcome:
    cases = [("generation.max_tokens=true", "логическое"),
             ("budget.max_requests=3.7", "целое число"),
             ("generation.temperature=жарко", "нужно число")]
    bad = []
    for assignment, marker in cases:
        finished = harness.run("--set", assignment, "--show-config")
        if finished.returncode != 2 or marker not in finished.stdout:
            bad.append(assignment)
    return Outcome(not bad, "проверено {} значений{}".format(
        len(cases), "" if not bad else ", подвели: " + ", ".join(bad)))


@scenario("параметры", "каждый именованный флаг доезжает до конфига")
def every_flag(harness: Harness) -> Outcome:
    other = harness.other_model()
    payload = harness.ask_json("Привет", "--model", other, "--max-tokens", "64",
                               "--name", "черновик", "--temperature", "0.1",
                               "--no-history", "--set", "attempts=2")
    marks = [payload["model"] == other, payload["agent"] == "черновик",
             payload["ok"], payload["usage"]["requests"] == 1]
    return Outcome(all(marks), "модель {} · агент {} · запросов {}".format(
        payload["model"], payload["agent"], payload["usage"]["requests"]))


@scenario("ошибки", "понятные отказы и коды возврата")
def failures(harness: Harness) -> Outcome:
    cases = [
        (["--set", "нетакого=1", "--show-config"], 2, "нет поля"),
        (["--model", "gpt-9", "--show-config"], 2, "нет модели"),
        (["--config", "{кривой", "--show-config"], 2, "не разобран"),
        (["--continue"], 2, "возвращаться некуда"),
        (["--resume", "нетакой"], 2, "нет сессии"),
        (["--rm-session", "нетакой"], 1, "Нет сессии"),
    ]
    bad = []
    for arguments, code, marker in cases:
        finished = harness.run(*arguments)
        if finished.returncode != code or marker not in finished.stdout:
            bad.append("{} → код {}".format(arguments[0], finished.returncode))
    return Outcome(not bad, "проверено {} отказов{}".format(
        len(cases), "" if not bad else ", подвели: " + ", ".join(bad)))


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def confirm_live(console: Console, count: int, provider, model: str) -> bool:
    """Спросить, если прогон стоит денег. Бесплатный тариф не спрашивает."""
    if provider.free:
        console.print(Panel(
            Text.from_markup(
                "Живой прогон: примерно [bold]{}[/] настоящих запросов к "
                "[bold]{}[/] ([green]бесплатный тариф[/]), модель {}.".format(
                    count, provider.name, model)),
            title="Живой прогон", border_style="green", box=box.ROUNDED))
        return True

    console.print(Panel(
        Text.from_markup(
            "Живой режим отправит примерно [bold]{}[/] настоящих запросов к "
            "[bold]{}[/], модель {} — и это [bold]платно[/].\n"
            "[dim]У бесплатного тарифа спрашивать не надо: "
            "--provider groq или --provider openrouter.[/]".format(
                count, provider.name, model)),
        title="⚠ Платный прогон", border_style="red", box=box.ROUNDED))
    try:
        return input("Продолжить? [y/N]: ").strip().lower() in ("y", "yes", "д", "да")
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Живая проверка агента")
    parser.add_argument("--live", action="store_true",
                        help="настоящие запросы к API")
    parser.add_argument("--provider", default="groq", metavar="КЛЮЧ",
                        help="провайдер живого прогона (по умолчанию groq — "
                             "бесплатный тариф)")
    parser.add_argument("--model", default="", metavar="ИМЯ",
                        help="модель живого прогона; по умолчанию штатная у провайдера")
    parser.add_argument("--only", default="", metavar="СЛОВО",
                        help="только сценарии, чьё имя или требование содержит слово")
    parser.add_argument("--yes", action="store_true", help="не спрашивать про --live")
    args = parser.parse_args(argv)

    chosen = [s for s in SCENARIOS
              if args.only.lower() in (s.title + " " + s.requirement).lower()]
    if not args.live:
        chosen = [s for s in chosen if not s.live_only]

    console = Console(highlight=False)
    provider_key, model_id = "", ""
    if args.live:
        from llmagent.transport import PROVIDERS

        provider = PROVIDERS.get(args.provider)
        if provider is None:
            console.print("[red]Неизвестный провайдер «{}»[/]".format(args.provider))
            return 2
        provider_key = provider.key
        model_id = args.model or provider.default_model.id
        if not args.yes and not confirm_live(console, len(chosen) * 3, provider, model_id):
            console.print("[dim]Отменено — денег не потрачено.[/]")
            return 0

    harness = Harness(args.live, console, provider_key, model_id)
    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("№", justify="right", style="dim", no_wrap=True)
    table.add_column("Треб.", no_wrap=True, style="dim")
    table.add_column("Сценарий")
    table.add_column("Итог", justify="center", no_wrap=True)
    table.add_column("Что увидели")

    passed = failed = skipped = 0
    started = time.perf_counter()
    try:
        for item in chosen:
            harness.fresh_home()
            with console.status("[bold]{}[/]".format(item.title), spinner="dots"):
                try:
                    outcome = item.run(harness)
                except Exception as exc:  # noqa: BLE001 — падение это тоже итог
                    outcome = Outcome(False, "{}: {}".format(type(exc).__name__, exc)[:110])
            if outcome.ok is None:
                mark, skipped = Text("—", style="yellow"), skipped + 1
            elif outcome.ok:
                mark, passed = Text("✓", style="green"), passed + 1
            else:
                mark, failed = Text("✗", style="red"), failed + 1
            table.add_row(str(item.number), item.requirement, item.title, mark,
                          Text(outcome.detail, style="" if outcome.ok else "yellow"))
    finally:
        harness.cleanup()

    elapsed = time.perf_counter() - started
    throttled = ("\n[dim]Отказов по частоте тарифа: {} — переждали и "
                 "повторили.[/]".format(harness.throttled) if harness.throttled else "")
    summary = Text.from_markup(
        "\n[green]{}[/] сошлось · [red]{}[/] не сошлось · [yellow]{}[/] неприменимо "
        "здесь · {:.1f} с{}\n[dim]Режим: {}[/]".format(
            passed, failed, skipped, elapsed, throttled,
            "настоящий API · {} · {}".format(provider_key, model_id) if args.live
            else "заглушка, денег не потрачено"))
    console.print(Panel(Group(table, summary), title="🔬 Живая проверка агента",
                        title_align="left",
                        border_style="green" if not failed else "red",
                        box=box.ROUNDED, padding=(0, 1)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
