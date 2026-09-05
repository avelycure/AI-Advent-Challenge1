#!/usr/bin/env python3
"""Ролик про управление ответом модели.

Показывает опыт целиком: один вопрос уходит в модель пятью запросами, в каждом
ровно один рычаг. По каждому шагу видно отправленный запрос, ответ, измерения
и проверку разбором.

Запуск:

    .venv-demo/bin/python demo/record_control_demo.py --key-file ~/.llm-test-key
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_renderer import FrameRenderer
from pipeline import build_video, record
from terminal_recorder import TerminalRecorder

COLS, ROWS = 132, 50
FPS = 10
MAX_SEGMENT = 1.7
FONT_SIZE = 15
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEP_PATTERN = re.compile(r"Шаг (\d) из 5")
ANSWER_TIMEOUT = 300

# Что говорится зрителю перед очередным шагом: какой рычаг сейчас добавляется.
NARRATION = {
    1: "Шаг 1 — ни одного рычага: смотрим, что модель отвечает сама по себе",
    2: "Шаг 2 — тот же вопрос плюс инструкция: ответ должен быть таблицей HTML",
    3: "Шаг 3 — формат задаём параметром API, схему в промпте не описываем",
    4: "Шаг 4 — условие завершения: промпт не меняется, добавляется только stop",
    5: "Шаг 5 — ограничение длины: max_tokens=120 и больше ничего",
}

CAPTIONS = {
    1: "Шаг 1 — точка отсчёта: голый вопрос, свободный ответ",
    2: "Шаг 2 — структуру задаёт инструкция в промпте",
    3: "Шаг 3 — структуру задаёт параметр запроса, а не текст",
    4: "Шаг 4 — тот же промпт, разница в одном параметре запроса",
    5: "Шаг 5 — лимит длины режет ответ на полуслове",
}

# Сколько держать экран, по виду приглашения. Ответ и проверки надо успеть
# прочитать, промежуточные переходы — нет.
HOLDS = [
    ("Enter — начать опыт", 11.0),
    ("Enter — продолжение ответа", 9.0),
    ("Enter — измерения и проверки", 11.0),
    ("Enter — следующий шаг", 5.0),
    ("Enter — выход", 14.0),
]


def build_scenario(rec: TerminalRecorder, key: str, provider: str, model: str) -> None:
    choose_provider_and_model(rec, key, provider, model)
    rec.wait_for("Условия опыта", 40)
    walk_steps(rec)


def choose_provider_and_model(rec: TerminalRecorder, key: str,
                              provider: str, model: str) -> None:
    rec.say("Реквизиты подхватываются из локального файла")
    rec.wait_for("Выберите LLM", 40)
    rec.wait_idle(0.6)
    rec.hold(2.4)
    rec.type_text(provider, settle=1.2)
    rec.enter()

    rec.say("Ключ уже сохранён — подтверждаем")
    rec.wait_for("Использовать|Вставьте", 20)
    rec.wait_idle(0.6)
    rec.hold(2.2)
    if rec.wait_for("Вставьте", 1):
        rec.type_text(key, echo=False)
    rec.enter()

    rec.say("Отвечает DeepSeek — на нём работают и stop, и response_format")
    rec.wait_for("Выберите модель", 60)
    rec.wait_idle(0.6)
    rec.hold(2.0)
    rec.type_text(model, settle=1.2)
    rec.enter()


def walk_steps(rec: TerminalRecorder) -> None:
    """Идти по экранам программы, пока она не дойдёт до выхода."""
    previous = ""
    while rec.alive:
        screen = settled_screen(rec, previous)
        if screen is None:
            return
        previous = head_of(screen)
        print("  шаг {} · {}".format(step_number(screen), prompt_of(screen)), flush=True)
        rec.say(caption_for(screen))
        rec.hold(hold_for(screen))
        if "Enter — выход" in screen:
            rec.enter()
            rec.pause(0.6)
            return
        announce(rec, screen)
        rec.enter()


def settled_screen(rec: TerminalRecorder, previous: str):
    """Дождаться нового экрана с приглашением.

    Приглашение остаётся на экране всё время, пока программа ждёт ответа модели,
    поэтому по одному и тому же кадру легко ответить дважды: подписи разъезжаются
    с картинкой, а лишний Enter уходит следующему шагу.
    """

    deadline = time.time() + ANSWER_TIMEOUT
    while rec.alive and time.time() < deadline:
        rec.wait_idle(1.0, 120)
        screen = rec.text()
        if head_of(screen) != previous and "Enter — " in screen:
            return screen
        time.sleep(0.2)
    return None


def head_of(screen: str) -> str:
    """Верх экрана — заголовок шага и запрос.

    Сравнивать экраны целиком нельзя: реплика зрителю набирается в строке ввода
    и оставляет после стирания след, из-за которого старый экран выглядит новым,
    а сценарий отвечает на одно приглашение дважды.
    """
    return "\n".join(screen.splitlines()[:14])


def prompt_of(screen: str) -> str:
    for marker, _ in HOLDS:
        if marker in screen:
            return marker
    return "?"


def announce(rec: TerminalRecorder, screen: str) -> None:
    """Перед каждым шагом — одна фраза о том, какой рычаг сейчас добавляется."""
    if "Enter — начать опыт" in screen:
        rec.narrate(NARRATION[1], 5.0)
        return
    if "Enter — следующий шаг" not in screen:
        return
    current = step_number(screen)
    if current is not None and current + 1 in NARRATION:
        rec.narrate(NARRATION[current + 1], 5.0)


def caption_for(screen: str) -> str:
    if "Условия опыта" in screen:
        return "Один вопрос, пять запросов, в каждом ровно один рычаг"
    if "Пять рычагов рядом" in screen:
        return "Итог: пять рычагов рядом, числа этого же прогона"
    number = step_number(screen)
    if number is None:
        return "Опыт идёт"
    if number == 4:
        return caption_for_stop(screen)
    if "Проверка разбором" in screen:
        return "Проверки считает код: разбор ответа, а не взгляд на него"
    return CAPTIONS[number]


def caption_for_stop(screen: str) -> str:
    if "Проверка разбором" in screen:
        return "Промпт совпадает до символа — разница только в параметре stop"
    if "4в" in screen:
        return "4в — договорный маркер: обрыв по протоколу, маркер срезан провайдером"
    if "4б" in screen:
        return "4б — тот же промпт со stop: список обрывается после третьего пункта"
    if "4а" in screen:
        return "4а — без условия завершения: модель заканчивает список сама"
    return CAPTIONS[4]


def hold_for(screen: str) -> float:
    for marker, seconds in HOLDS:
        if marker in screen:
            if marker == "Enter — измерения и проверки" and step_number(screen) == 4:
                return seconds + 3.0
            return seconds
    return 6.0


def step_number(screen: str):
    found = STEP_PATTERN.search(screen)
    return int(found.group(1)) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ролик про управление ответом модели")
    parser.add_argument("--key-file", default="~/.llm-test-key")
    parser.add_argument("--provider", default="1")
    parser.add_argument("--model", default="1")
    parser.add_argument("--output", default="~/Desktop/llm-control-demo.mp4")
    parser.add_argument("--demo", action="store_true",
                        help="прогон без сети, для проверки самого сценария")
    args = parser.parse_args()

    path = pathlib.Path(os.path.expanduser(args.key_file))
    if not path.exists():
        print("Не найден файл с ключом: {}".format(path))
        return 1
    key = path.read_text().strip()

    argv = ["./compare.sh", "--step"]
    if args.demo:
        argv.append("--demo")

    print("→ записываю сеанс…")
    frames = record(argv, PROJECT,
                    lambda rec: build_scenario(rec, key, args.provider, args.model),
                    COLS, ROWS, FPS)

    renderer = FrameRenderer(COLS, ROWS, font_size=FONT_SIZE)
    title = renderer.card("Управление ответом модели", [
        "один вопрос, пять запросов — в каждом ровно один рычаг, DeepSeek",
        "",
        "ничего · формат в промпте · формат через API · stop · max_tokens",
    ])
    outro = renderer.card("Что показал прогон", [
        "инструкция в промпте задаёт структуру, но держится на послушности модели",
        "response_format гарантирует синтаксис, а не схему — поля придумала модель",
        "stop обрывает генерацию у провайдера, но только по строке, которую модель напечатала",
        "max_tokens не сокращает ответ, а режет его на полуслове",
    ])
    build_video(frames, renderer, args.output, FPS, MAX_SEGMENT,
                title=title, outro=outro, title_seconds=2.8, outro_seconds=4.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
