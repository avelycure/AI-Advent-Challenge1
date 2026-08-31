#!/usr/bin/env python3
"""Запись демонстрационного ролика: проход по всему сценарию задания.

Ролик показывает то, что требовалось сделать: выбор LLM, запрос токена со ссылкой
на его получение, лоадер во время ожидания, тему диалога в шапке, счётчики
контекста и расхода токенов и — главное — что переписка помнит контекст.

Запуск:

    .venv-demo/bin/python demo/record_demo.py --key-file ~/.gigachat-key
    .venv-demo/bin/python demo/record_demo.py --fake      # без ключа, на заглушке

Зависимости ставятся отдельно от самого приложения:

    python3 -m venv .venv-demo
    .venv-demo/bin/pip install -r demo/requirements.txt
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio_ffmpeg
from frame_renderer import FrameRenderer
from terminal_recorder import Frame, TerminalRecorder, compress

COLS, ROWS = 100, 28
FPS = 10
MAX_SEGMENT = 1.7      # секунды: дольше одного вида экрана зритель не ждёт
TITLE_SECONDS = 2.2
OUTRO_SECONDS = 3.0
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_scenario(rec: TerminalRecorder, key: str, fake: bool) -> None:
    """Проход по флоу задания. Каждый шаг сопровождается подписью для зрителя."""

    rec.say("Шаг 1 — выбор LLM: поддерживаются четыре провайдера")
    rec.wait_for("Выберите LLM", 40)
    rec.wait_idle(0.6)
    rec.hold(3.0)          # дать прочитать список из четырёх провайдеров
    rec.type_text("4")           # GigaChat
    rec.enter()

    rec.say("Шаг 2 — токен: показана ссылка на выдачу, ввод скрыт")
    rec.wait_for("Ключ авторизации", 15)
    rec.wait_idle(0.6)
    rec.hold(2.8)          # дать разглядеть ссылку на получение ключа
    rec.type_text(key, echo=False)   # ключ не отображается и не попадает в кадр
    rec.enter()

    rec.say("Доступ проверяется до начала диалога")
    rec.wait_for("Выберите модель", 60)
    rec.wait_idle(0.6)
    rec.say("Шаг 3 — выбор модели")
    rec.pause(1.8)
    rec.type_text("1")
    rec.enter()
    rec.say("Диалог открыт: история пуста, счётчики на нуле")

    rec.wait_for("Диалог пуст", 15)
    rec.wait_idle(0.6)
    rec.pause(1.4)
    rec.say("Первый вопрос: просим запомнить два факта")
    rec.pause(0.6)
    rec.type_text("Запомни: число 47 и город Калининград. Ответь одним словом.")
    rec.enter()
    rec.pause(0.4)
    rec.say("Пока модель думает, крутится лоадер")
    rec.wait_idle(1.2, 120)

    rec.say("Тему в шапке сформулировала сама LLM, счётчики обновились")
    rec.hold(2.6)
    rec.say("Теперь проверим долгую историю: спрашиваем без подсказок")
    rec.pause(1.0)
    rec.type_text("Что я просил запомнить?")
    rec.enter()
    rec.say("Пока модель думает, крутится лоадер")
    rec.pause(0.4)
    rec.wait_idle(1.2, 120)

    rec.say("Модель помнит: вся переписка уходит в запрос целиком")
    rec.hold(3.4)

    rec.say("Команда /stats — полный учёт токенов")
    rec.type_text("/stats")
    rec.enter()
    rec.wait_idle(0.8, 30)
    rec.hold(4.0)          # таблицу статистики нужно успеть прочитать

    rec.say("Выход по команде /exit")
    rec.type_text("/exit")
    rec.enter()
    rec.wait_idle(0.8, 30)
    rec.hold(1.8)


def encode(images, path: str, fps: int) -> None:
    width, height = images[0].size
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "26", "-tune", "stillimage",
        "-movflags", "+faststart", path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(process.stderr.read().decode("utf-8", "replace")[-2000:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Запись демонстрационного ролика")
    parser.add_argument("--key-file", default="~/.gigachat-key",
                        help="файл с ключом авторизации GigaChat")
    parser.add_argument("--fake", action="store_true",
                        help="снимать на локальной заглушке, без обращения к API")
    parser.add_argument("--output", default="~/Desktop/llm-chat-demo.mp4")
    args = parser.parse_args()

    if args.fake:
        key = "demo-key"
    else:
        key_path = pathlib.Path(os.path.expanduser(args.key_file))
        if not key_path.exists():
            print("Не найден файл с ключом: {}".format(key_path)); return 1
        key = key_path.read_text().strip()

    print("→ записываю сеанс…")
    rec = TerminalRecorder(cols=COLS, rows=ROWS, fps=FPS)
    argv = ["./run.sh", "--demo"] if args.fake else ["./run.sh"]
    started = time.time()
    rec.start(argv, cwd=PROJECT)
    try:
        build_scenario(rec, key, args.fake)
    finally:
        rec.stop()
    print("  снято {} кадров за {:.0f} с".format(len(rec.frames), time.time() - started))

    frames: List[Frame] = compress(rec.frames, FPS, MAX_SEGMENT)
    print("  после сжатия пауз: {} кадров ({:.1f} с видео)".format(len(frames), len(frames) / FPS))

    print("→ рисую кадры…")
    renderer = FrameRenderer(COLS, ROWS)
    cache = {}
    images = []
    for frame in frames:
        key_ = (frame.snapshot, frame.caption)
        if key_ not in cache:
            cache[key_] = renderer.render(frame.snapshot, frame.caption)
        images.append(cache[key_])
    print("  уникальных кадров: {}".format(len(cache)))

    title = renderer.card("Терминальный чат с LLM", [
        "DeepSeek  ·  ChatGPT  ·  YandexGPT  ·  GigaChat",
        "",
        "долгая история диалога, счётчики токенов, тема разговора",
    ])
    outro = renderer.card("Что показано", [
        "выбор LLM и запрос токена со ссылкой на его получение",
        "лоадер во время ожидания ответа",
        "тема диалога в шапке, сформулированная самой LLM",
        "заполнение контекста, остаток и расход токенов",
        "память диалога: история пересылается в модель целиком",
    ])
    images = ([title] * int(TITLE_SECONDS * FPS) + images
              + [outro] * int(OUTRO_SECONDS * FPS))

    output = os.path.expanduser(args.output)
    print("→ кодирую…")
    encode(images, output, FPS)
    size = os.path.getsize(output)
    print("\nГотово: {}".format(output))
    print("  длительность {:.1f} с | размер {:.1f} МБ".format(len(images) / FPS, size / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
