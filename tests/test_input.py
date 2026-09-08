"""Ввод с клавиатуры через настоящий терминал.

Единственные тесты, которым нужен pty: всё остальное подаётся в канал, а канал
не терминал — вся работа с режимами терминала в нём просто не выполняется.
Именно поэтому дефект со склеиванием сообщений и дожил до жалобы человека.
"""
from __future__ import annotations

import os
import re
import select
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
            os.chdir(str(ROOT))
            os.execve(PYTHON, [PYTHON, "chat.py", *arguments], environment)
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
        try:
            os.close(self.fd)
        except OSError:
            pass

    @property
    def text(self) -> str:
        return ESCAPES.sub("", "".join(self.seen))

    @property
    def questions(self) -> list:
        """Что заглушка получила: она повторяет заданный ей вопрос."""
        seen = []
        for found in ASKED.findall(self.text):
            if found not in seen:
                seen.append(found)
        return seen


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
