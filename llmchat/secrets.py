"""Поиск реквизитов доступа: переменная окружения, локальный файл, ручной ввод.

Ключи не хранятся в репозитории и не записываются программой. Модуль только
читает то, что пользователь положил рядом сам, чтобы не вводить одно и то же
при каждом запуске.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SecretSource:
    """Откуда взято значение и что именно взято."""
    label: str
    value: str
    path: Optional[str] = None

    @property
    def warning(self) -> Optional[str]:
        return permission_warning(self.path) if self.path else None


def mask(value: str) -> str:
    """Показать, что нашли, не раскрывая ключ целиком."""
    if len(value) <= 10:
        return "*" * len(value)
    return "{}…{}".format(value[:6], value[-4:])


def shorten(path: str) -> str:
    home = os.path.expanduser("~")
    return path.replace(home, "~", 1) if path.startswith(home) else path


def read_secret_file(path: str) -> Optional[str]:
    """Прочитать первую содержательную строку файла.

    Пустые строки и строки, начинающиеся с решётки, пропускаются: так файл
    можно снабдить комментарием, не ломая чтение.
    """
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
            for line in handle:
                candidate = line.strip()
                if candidate and not candidate.startswith("#"):
                    return candidate
    except OSError:
        return None
    return None


def permission_warning(path: str) -> Optional[str]:
    """Предупредить, если файл с ключом доступен кому-то кроме владельца."""
    try:
        mode = os.stat(os.path.expanduser(path)).st_mode
    except OSError:
        return None
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        return "файл читают не только вы — стоит выполнить chmod 600 {}".format(shorten(path))
    return None


def find_sources(env_name: Optional[str], files: Sequence[str]) -> List[SecretSource]:
    """Собрать доступные источники значения в порядке предпочтения."""
    sources: List[SecretSource] = []

    if env_name:
        value = os.environ.get(env_name)
        if value and value.strip():
            sources.append(SecretSource("переменная {}".format(env_name), value.strip()))

    seen = set()
    for path in files:
        expanded = os.path.expanduser(path)
        if expanded in seen:
            continue
        seen.add(expanded)
        value = read_secret_file(expanded)
        if value:
            sources.append(SecretSource("файл {}".format(shorten(expanded)), value, expanded))

    return sources
