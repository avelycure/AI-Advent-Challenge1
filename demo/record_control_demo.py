#!/usr/bin/env python3
"""Ролик про управление ответом модели.

Показывает опыт целиком: один и тот же вопрос уходит в модель дважды — свободно
и с ограничениями, — а результаты сравниваются, причём соблюдение формата
проверяется разбором ответа.

Запуск:

    .venv-demo/bin/python demo/record_control_demo.py --key-file ~/.gigachat-key
    .venv-demo/bin/python demo/record_control_demo.py --fake     # без ключа
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

COLS, ROWS = 120, 40
FPS = 10
MAX_SEGMENT = 1.7
FONT_SIZE = 15
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_scenario(rec: TerminalRecorder, key: str) -> None:
    rec.say("Шаг 1 — выбор LLM")
    rec.wait_for("Выберите LLM", 40)
    rec.wait_idle(0.6)
    rec.hold(2.0)
    rec.type_text("4")
    rec.enter()

    rec.say("Шаг 2 — ключ авторизации, ввод скрыт")
    rec.wait_for("Ключ авторизации", 15)
    rec.wait_idle(0.6)
    rec.hold(1.8)
    rec.type_text(key, echo=False)
    rec.enter()

    rec.say("Шаг 3 — выбор модели")
    rec.wait_for("Выберите модель", 60)
    rec.wait_idle(0.6)
    rec.hold(1.6)
    rec.type_text("1")
    rec.enter()

    rec.say("Условия опыта: вопрос один, ограничения разные")
    rec.wait_for("Условия опыта", 20)
    rec.wait_idle(0.6)
    rec.hold(4.5)          # дать прочитать, что именно добавлено во второй запрос

    rec.say("Первый запрос уходит без ограничений")
    rec.wait_idle(1.2, 120)

    rec.say("Слева свободный ответ, справа — управляемый")
    rec.hold(5.0)
    rec.enter()

    rec.say("Формат проверяется разбором ответа, а не на глаз")
    rec.wait_for("Проверка формата", 30)
    rec.wait_idle(1.0, 30)
    rec.hold(7.0)          # таблицу проверок надо успеть прочитать
    rec.enter()
    rec.pause(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ролик про управление ответом")
    parser.add_argument("--key-file", default="~/.gigachat-key")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--format", default="json", choices=["json", "yaml", "md"])
    parser.add_argument("--output", default="~/Desktop/llm-control-demo.mp4")
    args = parser.parse_args()

    if args.fake:
        key = "demo-key"
    else:
        path = pathlib.Path(os.path.expanduser(args.key_file))
        if not path.exists():
            print("Не найден файл с ключом: {}".format(path))
            return 1
        key = path.read_text().strip()

    # --ask-keys: в ролике показываем ввод ключа, а не подстановку из файла.
    argv = ["./compare.sh", "--step", "--ask-keys", "--format", args.format]
    if args.fake:
        argv.append("--demo")

    print("→ записываю сеанс…")
    frames = record(argv, PROJECT, lambda rec: build_scenario(rec, key), COLS, ROWS, FPS)

    renderer = FrameRenderer(COLS, ROWS, font_size=FONT_SIZE)
    title = renderer.card("Управление ответом модели", [
        "один и тот же вопрос отправляется дважды",
        "",
        "формат ответа  ·  ограничение длины  ·  условие завершения",
    ])
    outro = renderer.card("Что показано", [
        "свободный ответ — текст, непригодный для машинной обработки",
        "управляемый — строгая схема, короче в разы",
        "формат задан инструкцией, response_format, max_tokens и stop",
        "соблюдение схемы проверено разбором ответа, а не на глаз",
    ])
    build_video(frames, renderer, args.output, FPS, MAX_SEGMENT,
                title=title, outro=outro, title_seconds=2.4, outro_seconds=3.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
