"""Ввод с клавиатуры через настоящий терминал.

Единственные тесты, которым нужен pty: всё остальное подаётся в канал, а канал
не терминал — вся работа с режимами терминала в нём просто не выполняется.
Именно поэтому дефект со склеиванием сообщений и дожил до жалобы человека.
"""
from __future__ import annotations

import os
import re
import select
import signal
import time

import pytest

pty = pytest.importorskip("pty")

from conftest import ROOT

PYTHON = str(ROOT / ".venv" / "bin" / "python")
ESCAPES = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
ASKED = re.compile(r"Вы спросили: «([^»]*)»")


class Terminal:
    """Программа, запущенная в настоящем терминале."""

    def __init__(self, home, *arguments: str) -> None:
        environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
                       "HOME": str(home), "COLUMNS": "100", "TERM": "xterm-256color"}
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            # Ветка дочернего процесса. Выйти из неё обязательно самому: если
            # execve сорвётся, ребёнок продолжит выполнять набор тестов вторым
            # процессом, и падать начнёт где угодно, кроме места ошибки.
            try:
                os.chdir(str(ROOT))
                os.execve(PYTHON, [PYTHON, "chat.py", *arguments], environment)
            except BaseException:
                os._exit(127)
            os._exit(127)
        self.seen: list = []

    def wait(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 8192)
            except OSError:
                return
            if not chunk:
                return
            self.seen.append(chunk.decode("utf-8", "replace"))

    def type(self, text: str) -> None:
        os.write(self.fd, text.encode("utf-8"))
        time.sleep(0.2)

    def close(self) -> None:
        """Закрыть терминал и непременно похоронить ребёнка.

        Закрыть один только описатель недостаточно: программа в терминале
        продолжает жить, и после набора тестов остаются брошенные процессы.
        """
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.kill(self.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            os.waitpid(self.pid, 0)
        except (OSError, ChildProcessError):
            pass

    @property
    def text(self) -> str:
        return ESCAPES.sub("", "".join(self.seen))

    @property
    def line(self) -> str:
        """Что сейчас видно в нижней строке экрана.

        Простейший эмулятор: возврат каретки, забой и очистка до конца строки.
        Больше ничего и не нужно — readline перерисовывает строку именно ими,
        а без такого разбора видно только поток вывода, а не экран, и стёртое
        в нём неотличимо от оставшегося.
        """
        return _replay("".join(self.seen))

    @property
    def questions(self) -> list:
        """Что заглушка получила: она повторяет заданный ей вопрос."""
        seen = []
        for found in ASKED.findall(self.text):
            if found not in seen:
                seen.append(found)
        return seen


# Управляющие последовательности, меняющие содержимое строки. Прочие для
# разбора неважны: цвет на то, что видно в строке, не влияет.
ERASE_TO_END = "\x1b[K"


def _replay(raw: str) -> str:
    line: list = []
    index = 0
    while index < len(raw):
        if raw.startswith(ERASE_TO_END, index):
            index += len(ERASE_TO_END)
            continue
        found = ESCAPES.match(raw, index)
        if found:
            index = found.end()
            continue
        char = raw[index]
        index += 1
        if char == "\r":
            line = []
        elif char == "\n":
            line = []
        elif char == "\b":
            if line:
                line.pop()
        else:
            line.append(char)
    return "".join(line).rstrip()


@pytest.fixture
def terminal(tmp_path):
    started = []

    def start(*arguments: str) -> Terminal:
        item = Terminal(tmp_path, "--demo", "--config", "demo", *arguments)
        started.append(item)
        item.wait(3.0)
        return item

    yield start
    for item in started:
        item.close()


def test_typing_ahead_does_not_get_swallowed(terminal):
    """Набранное, пока модель думает, должно остаться своим сообщением.

    Прежде вторая строка попадала в конец первого сообщения: программа считала
    вставкой всё, что оказалось во входной очереди сразу после Enter. Человек
    видел это как пропавшие слова и как чужие слова в своей строке.
    """
    session = terminal("--set", "transport.demo_delay=1.5")
    session.type("Первый вопрос\n")
    session.type("второй вопрос\n")     # модель ещё думает над первым
    session.wait(9.0)

    assert "Первый вопрос второй вопрос" not in session.text
    assert session.questions[:2] == ["Первый вопрос", "второй вопрос"]


def test_a_multiline_paste_stays_one_message(terminal):
    """Вставка в несколько строк — одно сообщение, а не по запросу на строку."""
    session = terminal("--set", "transport.demo_delay=0.2")
    session.type("\x1b[200~строка один\nстрока два\nстрока три\x1b[201~\n")
    session.wait(6.0)

    assert session.questions
    first = session.questions[0]
    assert "строка один" in first and "строка три" in first
    # Обрамление вставки в текст сообщения попадать не должно ни в каком виде.
    assert "200~" not in first and "201~" not in first and "00~" not in first


def test_an_ordinary_line_is_untouched(terminal):
    session = terminal("--set", "transport.demo_delay=0.2")
    session.type("Обычный вопрос без затей\n")
    session.wait(6.0)
    assert session.questions[:1] == ["Обычный вопрос без затей"]


def test_a_paste_does_not_corrupt_the_text_inside_it(terminal):
    """Обрамление срезается по краям, а не всюду: иначе адрес терял хвост."""
    session = terminal("--set", "transport.demo_delay=0.2")
    session.type('\x1b[200~curl "http://x/?v=201~2"\nвторая строка\x1b[201~\n')
    session.wait(6.0)
    assert session.questions
    assert "?v=201~2" in session.questions[0]


def test_erasing_a_word_does_not_wipe_the_prompt(terminal):
    """Подсказка «Вы ›» обязана пережить стирание набранного.

    Пока подсказку печатал rich, а input получал пустую строку, readline
    считал, что строка начинается с нулевой колонки, и, стирая слово, затирал
    подсказку вместе с ним. На экране оставалась пустота.
    """
    session = terminal()
    session.type("привет")
    session.wait(0.5)
    assert "Вы ›" in session.line and "привет" in session.line

    session.type("\x7f" * 6)
    session.wait(0.7)
    assert "Вы ›" in session.line
    assert "привет" not in session.line
