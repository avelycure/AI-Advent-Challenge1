#!/usr/bin/env python3
"""Ролик про параметры генерации: один вопрос, разные настройки.

Один и тот же вопрос задаётся шесть раз подряд, между опытами меняются только
параметры. Видно, как меняется длина ответа, его предсказуемость и стиль —
и что делает программа, когда провайдер параметр принимает, но не применяет.

Перед каждым опытом история очищается командой /new: иначе модель видела бы
свои прошлые ответы и сравнение перестало бы быть честным.

Запуск:

    .venv-demo/bin/python demo/record_params_demo.py --key-file ~/.gigachat-key
    .venv-demo/bin/python demo/record_params_demo.py --fake     # без ключа
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_renderer import FrameRenderer
from pipeline import build_video, record
from terminal_recorder import TerminalRecorder

COLS, ROWS = 110, 38
FPS = 10
MAX_SEGMENT = 1.7
FONT_SIZE = 15
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUESTION = "Опиши язык программирования Rust."
QUESTION_JSON = QUESTION + ' Ответь только JSON: {"name":"...","year":0}'
ANSWER_TIMEOUT = 120

# Паузы подобраны под зрителя: набранную строку надо успеть прочитать до
# нажатия Enter, а ответ — рассмотреть, прежде чем экран сменится.
SETTLE = 1.3          # после набора строки, перед отправкой
SHOW_ANSWER = 5.5     # сколько держим ответ на экране
SHOW_PARAMS = 3.0     # сколько держим подтверждение смены параметров


def ask(rec: TerminalRecorder, caption: str, question: str = QUESTION,
        hold: float = SHOW_ANSWER) -> None:
    """Задать вопрос: набрать, дать прочитать, отправить, показать ответ."""
    rec.say(caption)
    rec.type_text(question, settle=SETTLE)
    rec.enter()
    rec.pause(0.3)
    rec.wait_idle(1.2, ANSWER_TIMEOUT)
    rec.hold(hold)


def set_params(rec: TerminalRecorder, command: str, caption: str,
               hold: float = SHOW_PARAMS) -> None:
    rec.say(caption)
    rec.type_text(command, settle=SETTLE)
    rec.enter()
    rec.wait_idle(0.8, 30)
    rec.hold(hold)


def fresh(rec: TerminalRecorder) -> None:
    """Очистить историю: иначе модель увидит свои прошлые ответы."""
    rec.type_text("/new", settle=0.6)
    rec.enter()
    rec.wait_idle(0.5, 20)
    rec.hold(0.5)


def build_scenario(rec: TerminalRecorder, key: str) -> None:
    rec.say("Реквизиты подхватываются из локального файла")
    rec.wait_for("Выберите LLM", 40)
    rec.wait_idle(0.6)
    rec.hold(2.5)
    rec.type_text("4", settle=1.0)
    rec.enter()
    rec.wait_for("Использовать|Вставьте", 20)
    rec.wait_idle(0.6)
    rec.hold(2.4)
    if rec.wait_for("Вставьте", 1):
        rec.type_text(key, echo=False)
    rec.enter()

    rec.say("Выбираем модель")
    rec.wait_for("Выберите модель", 60)
    rec.wait_idle(0.6)
    rec.hold(2.5)
    rec.type_text("1", settle=1.0)
    rec.enter()
    rec.wait_for("Диалог пуст", 15)
    rec.wait_idle(0.6)
    rec.hold(1.2)

    rec.say("Дальше — один и тот же вопрос при разных параметрах")
    rec.narrate("Задаю один вопрос снова и снова, меняя только параметры", read=3.0, cps=19)
    set_params(rec, "/change_llm_params",
               "Вот что можно менять прямо в диалоге", hold=5.5)

    # --- опыт 1: длина ----------------------------------------------------
    rec.narrate("Опыт 1: режу длину ответа до 40 токенов", read=3.0, cps=19)
    set_params(rec, "/change_llm_params max_tokens=40", "Ставим предел длины")
    ask(rec, "Тот же вопрос — ответ обрывается на полуслове", hold=7.0)

    # --- опыт 2: предсказуемость -----------------------------------------
    rec.say("Очищаем историю и возвращаем длину")
    fresh(rec)
    set_params(rec, "/change_llm_params max_tokens=200 temperature=0",
               "Опыт 2 — длину вернули, случайность убрали в ноль")
    ask(rec, "Тот же вопрос при temperature=0", hold=5.0)
    rec.say("Очищаем историю")
    fresh(rec)
    rec.narrate("Повторяю тот же вопрос — ответ должен совпасть", read=3.0, cps=19)
    ask(rec, "Ответ почти дословно тот же — вот что даёт нулевая температура", hold=7.0)

    # --- опыт 3: разброс --------------------------------------------------
    rec.say("Очищаем историю")
    fresh(rec)
    set_params(rec, "/change_llm_params temperature=1.8",
               "Опыт 3 — поднимаем случайность почти до предела")
    ask(rec, "Тот же вопрос — при 1.8 ответ разваливается в набор слов", hold=7.0)

    # --- опыт 4: формат параметром ---------------------------------------
    rec.say("Очищаем историю")
    fresh(rec)
    set_params(rec, "/change_llm_params max_tokens=150 temperature=0.3 response_format=json_object",
               "Опыт 4 — просим JSON параметром запроса")
    ask(rec, "GigaChat параметр принял, но не применил — ответ остался прозой", hold=7.0)

    # --- опыт 5: формат инструкцией ---------------------------------------
    rec.say("Очищаем историю")
    fresh(rec)
    rec.narrate("Параметр не сработал. Попрошу формат словами в вопросе", read=3.4, cps=19)
    ask(rec, "Тот же вопрос плюс требование формата — и вот он, JSON",
        question=QUESTION_JSON, hold=7.5)

    set_params(rec, "/reset_llm_params", "Возвращаем всё к значениям по умолчанию", hold=3.0)
    rec.type_text("/exit", settle=1.0)
    rec.enter()
    rec.wait_idle(0.8, 30)
    rec.hold(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ролик про параметры генерации")
    parser.add_argument("--key-file", default="~/.gigachat-key")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--output", default="~/Desktop/llm-params-demo.mp4")
    args = parser.parse_args()

    if args.fake:
        key = "demo-key"
    else:
        path = pathlib.Path(os.path.expanduser(args.key_file))
        if not path.exists():
            print("Не найден файл с ключом: {}".format(path))
            return 1
        key = path.read_text().strip()

    argv = ["./run.sh"] + (["--demo", "--ask-keys"] if args.fake else [])

    print("→ записываю сеанс…")
    frames = record(argv, PROJECT, lambda rec: build_scenario(rec, key), COLS, ROWS, FPS)

    renderer = FrameRenderer(COLS, ROWS, font_size=FONT_SIZE)
    title = renderer.card("Параметры генерации", [
        "один и тот же вопрос, разные настройки",
        "",
        "длина  ·  предсказуемость  ·  разброс  ·  формат",
    ])
    outro = renderer.card("Что показано", [
        "max_tokens режет длину, обрыв программа объясняет прямо",
        "temperature=0 делает ответ повторяемым, 1.8 — разрушает его",
        "response_format GigaChat принимает, но не применяет",
        "формат надёжно задаётся словами в самом вопросе",
        "изменённые параметры видны в счётчиках, /reset_llm_params возвращает всё",
    ])
    build_video(frames, renderer, args.output, FPS, MAX_SEGMENT,
                title=title, outro=outro, title_seconds=2.4, outro_seconds=3.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
