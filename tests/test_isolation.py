"""Изоляция коробки — требование ментора, проверяемое машиной, а не на глаз.

Слово «изолированный модуль» стоит ровно столько, сколько стоит проверка. Без
неё первый же удобный импорт ``rich`` внутрь агента прошёл бы незамеченным, и
коробка снова стала бы частью терминального приложения.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "llmagent"

# Чего в коробке быть не должно: интерфейс, его библиотека и всё, что читает
# стандартный ввод. Агент вызывают из кода, и спрашивать ему некого.
FORBIDDEN_MODULES = ("rich", "llmchat")


def test_importing_the_agent_pulls_in_neither_rich_nor_the_chat():
    """Проверяем в отдельном процессе: в текущем оба модуля уже загружены."""
    code = (
        "import sys, json; import llmagent; "
        "print(json.dumps([m for m in sys.modules "
        "if m.split('.')[0] in {}]))".format(list(FORBIDDEN_MODULES))
    )
    finished = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, check=True)
    assert finished.stdout.strip().endswith("[]"), finished.stdout


def test_no_source_file_of_the_agent_mentions_the_chat_or_rich():
    """Статическая проверка ловит и импорт внутри функции, который первый тест
    пропустил бы, пока такую функцию никто не вызвал."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN_MODULES:
                    offenders.append("{}:{} → {}".format(
                        path.relative_to(ROOT), node.lineno, name))
    assert offenders == [], "коробка потянула интерфейс: " + "; ".join(offenders)


def test_the_chat_depends_on_the_agent_and_not_the_other_way_round():
    """Зависимость должна идти в одну сторону, иначе изоляции нет."""
    chat_imports_agent = any(
        "llmagent" in path.read_text(encoding="utf-8")
        for path in (ROOT / "llmchat").glob("*.py"))
    assert chat_imports_agent


def test_the_agent_never_prints():
    """Печать в коробке означала бы, что у неё есть своё представление."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("print", "input")):
                offenders.append("{}:{} → {}".format(
                    path.relative_to(ROOT), node.lineno, node.func.id))
    assert offenders == [], "коробка разговаривает с человеком: " + "; ".join(offenders)
