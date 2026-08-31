"""Запись работы терминального приложения в видеоролик.

Алгоритм записи:

1. Приложение запускается под псевдотерминалом фиксированного размера, поэтому
   кадр не зависит от окна, в котором идёт запись.
2. Фоновый поток непрерывно читает вывод и скармливает его эмулятору терминала
   ``pyte``, который держит состояние экрана вместе с цветами.
3. Второй поток снимает состояние экрана с постоянной частотой и складывает
   снимки в ленту, помечая, шёл ли в этот момент ввод.
4. Сценарий выполняется шагами: дождаться текста, дождаться тишины в выводе,
   «напечатать» строку с имитацией набора, выдержать паузу.
5. Лента сжимается: подряд идущие почти одинаковые кадры без ввода прореживаются
   до заданной длительности. Так минуты ожидания ответа модели превращаются
   в пару секунд, а анимация спиннера при этом не застывает. Кадры, снятые во
   время набора текста, не трогаются — иначе набор перестанет быть виден.
6. Кадры рисуются в изображения и кодируются в H.264.
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import termios
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pyte

# Ячейка экрана: символ и его оформление.
Cell = Tuple[str, str, str, bool, bool]  # символ, цвет текста, цвет фона, жирный, инверсия
Snapshot = Tuple[Tuple[Cell, ...], ...]


@dataclass
class Frame:
    snapshot: Snapshot
    active: bool   # снят во время ввода или удержания — такие кадры не прореживаем
    caption: str   # подпись этапа, поясняющая зрителю происходящее


class TerminalRecorder:
    def __init__(self, cols: int = 100, rows: int = 30, fps: int = 10) -> None:
        self.cols, self.rows, self.fps = cols, rows, fps
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.frames: List[Frame] = []
        self._lock = threading.Lock()
        self._alive = False
        self._active = False
        self._caption = ""
        self._last_output = 0.0
        self.fd = -1
        self.pid = -1

    # --- запуск и потоки ---------------------------------------------
    def start(self, argv: Sequence[str], cwd: str, env: Optional[dict] = None) -> None:
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(cwd)
            os.environ["TERM"] = "xterm-256color"
            os.environ["COLUMNS"] = str(self.cols)
            os.environ["LINES"] = str(self.rows)
            os.environ["PYTHONUNBUFFERED"] = "1"
            # COLORTERM намеренно не ставим: без него rich выдаёт 256 цветов,
            # которые эмулятор разбирает точно.
            os.environ.pop("COLORTERM", None)
            for key, value in (env or {}).items():
                os.environ[key] = value
            os.execvp(argv[0], list(argv))
        self.pid, self.fd = pid, fd
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
        self._alive = True
        self._last_output = time.time()
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._sampler, daemon=True).start()

    def _reader(self) -> None:
        while self._alive:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                continue
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                break
            if not data:
                break
            with self._lock:
                self.stream.feed(data)
                self._last_output = time.time()
        self._alive = False

    def _sampler(self) -> None:
        period = 1.0 / self.fps
        next_at = time.time()
        while self._alive:
            now = time.time()
            if now < next_at:
                time.sleep(min(period / 4, next_at - now))
                continue
            next_at += period
            with self._lock:
                self.frames.append(Frame(self._snapshot(), self._active, self._caption))

    def _snapshot(self) -> Snapshot:
        rows = []
        for row in range(self.rows):
            line = self.screen.buffer[row]
            rows.append(tuple(
                (line[col].data, line[col].fg, line[col].bg, line[col].bold, line[col].reverse)
                for col in range(self.cols)
            ))
        return tuple(rows)

    def text(self) -> str:
        with self._lock:
            return "\n".join(self.screen.display)

    # --- шаги сценария ------------------------------------------------
    def wait_for(self, pattern: str, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        regex = re.compile(pattern)
        while time.time() < deadline and self._alive:
            if regex.search(self.text()):
                return True
            time.sleep(0.05)
        return False

    def wait_idle(self, quiet: float = 0.6, timeout: float = 90.0) -> bool:
        """Дождаться тишины в выводе: значит, приложение закончило рисовать."""
        deadline = time.time() + timeout
        while time.time() < deadline and self._alive:
            if time.time() - self._last_output >= quiet:
                return True
            time.sleep(0.05)
        return False

    def say(self, caption: str) -> None:
        """Подпись, которая будет видна на кадрах до следующего вызова."""
        self._caption = caption

    def pause(self, seconds: float) -> None:
        time.sleep(seconds)

    def hold(self, seconds: float) -> None:
        """Задержать кадр в ролике: такие кадры не прореживаются при сжатии.

        Нужно там, где зрителю надо успеть прочитать экран — таблицу провайдеров
        или статистику по токенам, — иначе сжатие пауз укоротит их до общего предела.
        """
        self._active = True
        time.sleep(seconds)
        self._active = False

    def type_text(self, text: str, cps: float = 32.0, echo: bool = True) -> None:
        """Напечатать строку по символу. echo=False — для скрытого ввода ключа."""
        self._active = echo
        delay = 1.0 / cps if echo else 0.0
        for ch in text:
            os.write(self.fd, ch.encode("utf-8"))
            if delay:
                time.sleep(delay)
        self._active = False

    def enter(self) -> None:
        os.write(self.fd, b"\n")

    def stop(self) -> None:
        self._alive = False
        time.sleep(0.2)
        try:
            os.close(self.fd)
        except OSError:
            pass


def diff_cells(a: Snapshot, b: Snapshot) -> int:
    changed = 0
    for row_a, row_b in zip(a, b):
        for cell_a, cell_b in zip(row_a, row_b):
            if cell_a != cell_b:
                changed += 1
                if changed > 16:
                    return changed
    return changed


def compress(frames: List[Frame], fps: int, max_segment: float,
             similar_cells: int = 6) -> List[Frame]:
    """Проредить длинные однообразные участки, не трогая кадры с вводом."""
    limit = max(1, int(round(max_segment * fps)))
    result: List[Frame] = []
    index = 0
    while index < len(frames):
        if frames[index].active:
            result.append(frames[index])
            index += 1
            continue
        start = index
        anchor = frames[index]
        index += 1
        while (index < len(frames) and not frames[index].active
               and frames[index].caption == anchor.caption
               and diff_cells(anchor.snapshot, frames[index].snapshot) <= similar_cells):
            index += 1
        run = frames[start:index]
        if len(run) <= limit:
            result.extend(run)
        else:
            # Равномерно выбираем кадры, чтобы спиннер продолжал вращаться.
            step = len(run) / float(limit)
            result.extend(run[int(i * step)] for i in range(limit))
    return result
