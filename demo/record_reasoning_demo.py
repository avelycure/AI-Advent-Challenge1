#!/usr/bin/env python3
"""Ролик про четыре способа поставить одну задачу.

Показывает опыт целиком: одна задача с заранее известным ответом решается
прямым вопросом, пошагово, по промпту от самой модели и от лица группы
экспертов, а затем результаты сравниваются.

Запуск:

    .venv-demo/bin/python demo/record_reasoning_demo.py \\
        --key-file ~/.openrouter-key --provider 6 --model 1 --judge gigachat
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

COLS, ROWS = 118, 40
FPS = 10
MAX_SEGMENT = 1.7
FONT_SIZE = 15
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOLVE_TIMEOUT = 900     # четыре способа с повторами считаются долго


def build_scenario(rec: TerminalRecorder, key: str, provider: str, model: str) -> None:
    rec.say("Реквизиты подхватываются из локального файла")
    rec.wait_for("Выберите LLM", 40)
    rec.wait_idle(0.6)
    rec.hold(2.2)
    rec.type_text(provider, settle=1.0)
    rec.enter()

    rec.say("Ключ уже сохранён — подтверждаем")
    rec.wait_for("Использовать|Вставьте", 20)
    rec.wait_idle(0.6)
    rec.hold(2.2)
    if rec.wait_for("Вставьте", 1):
        rec.type_text(key, echo=False)
    rec.enter()

    rec.say("Выбираем модель, которая будет решать")
    rec.wait_for("Выберите модель", 60)
    rec.wait_idle(0.6)
    rec.hold(1.8)
    rec.type_text(model, settle=1.0)
    rec.enter()

    rec.say("Задача с заранее известным ответом — иначе «точнее» не проверить")
    rec.wait_for("Условия опыта", 30)
    rec.wait_idle(0.8)
    rec.hold(7.0)

    rec.say("Четыре способа спросить одно и то же — модель решает")
    rec.wait_for("Enter — показать ответы", SOLVE_TIMEOUT)
    rec.wait_idle(1.0, 60)
    rec.hold(5.5)          # промпт, который модель написала себе сама
    rec.enter()

    captions = [
        "Способ 1 — прямой вопрос, без единой добавки",
        "Способ 2 — просьба решать пошагово",
        "Способ 3 — решение по промпту, который модель написала себе",
        "Способ 4 — аналитик, инженер и критик",
    ]
    for caption in captions:
        rec.say(caption)
        rec.wait_idle(0.8, 60)
        rec.hold(7.0)
        rec.enter()

    rec.say("Свод: верность итога считает код, качество оценивает другая модель")
    rec.wait_idle(1.0, 120)
    rec.hold(11.0)
    rec.enter()
    rec.pause(0.6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ролик про четыре способа спросить")
    parser.add_argument("--key-file", default="~/.openrouter-key")
    parser.add_argument("--provider", default="6")
    parser.add_argument("--model", default="1")
    parser.add_argument("--judge", default="gigachat")
    parser.add_argument("--runs", default="2")
    parser.add_argument("--label", default="MiniMax M3, оценивает GigaChat")
    parser.add_argument("--output", default="~/Desktop/llm-reasoning-demo.mp4")
    args = parser.parse_args()

    path = pathlib.Path(os.path.expanduser(args.key_file))
    if not path.exists():
        print("Не найден файл с ключом: {}".format(path))
        return 1
    key = path.read_text().strip()

    argv = ["./reasoning.sh", "--step", "--runs", args.runs]
    if args.judge:
        argv += ["--judge", args.judge]

    print("→ записываю сеанс…")
    frames = record(argv, PROJECT,
                    lambda rec: build_scenario(rec, key, args.provider, args.model),
                    COLS, ROWS, FPS)

    renderer = FrameRenderer(COLS, ROWS, font_size=FONT_SIZE)
    title = renderer.card("Четыре способа спросить", [
        "одна задача, четыре постановки промпта — {}".format(args.label),
        "",
        "прямой вопрос · пошагово · промпт от модели · группа экспертов",
    ])
    # Карточка описывает устройство опыта, а не исход: от прогона к прогону
    # побеждают разные способы, и обещать конкретный результат нельзя.
    outro = renderer.card("Что показано", [
        "одна задача, четыре постановки промпта",
        "верность итога считает код — у задачи есть заранее известный ответ",
        "качество объяснения оценивает другая модель, а не та, что решала",
        "итог из ответа достаёт модель, а не регулярное выражение",
        "разброс между прогонами больше, чем между способами",
    ])
    build_video(frames, renderer, args.output, FPS, MAX_SEGMENT,
                title=title, outro=outro, title_seconds=2.6, outro_seconds=3.4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
