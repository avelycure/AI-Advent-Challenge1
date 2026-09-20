"""Явные слои памяти агента.

Краткосрочная память уже живёт в :mod:`llmagent.history`: это сообщения
текущего диалога. Здесь находятся два слоя, которые человек наполняет только
явно: рабочая память текущей задачи и долговременная память пользователя и
проекта.

Слои намеренно не пытаются угадать назначение текста. Обычная реплика всегда
остаётся краткосрочной; в рабочую или долговременную память значение попадает
только через ``Agent.remember`` (в чате — через ``/memory``).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import AgentError
from .profile import PROFILE_PROMPT, UserProfile


SHORT_TERM = "short"
WORKING = "working"
LONG_TERM = "long"
LAYERS = (SHORT_TERM, WORKING, LONG_TERM)

PROFILE = "profile"
RULES = "rules"
DECISIONS = "decisions"
KNOWLEDGE = "knowledge"
LONG_TERM_SECTIONS = (PROFILE, RULES, DECISIONS, KNOWLEDGE)

# Память попадает в файлы. Назвать секрет «api_key» — достаточно явная ошибка,
# чтобы остановить её до записи; произвольное значение надёжно распознать
# невозможно, поэтому документация дополнительно запрещает сохранять секреты.
SENSITIVE_KEYS = {
    "api_key", "apikey", "password", "passwd", "secret", "access_token",
    "auth_token", "credential", "credentials", "private_key", "authorization",
    "пароль", "секрет", "токен",
}
SENSITIVE_SUFFIXES = (
    "_api_key", "_apikey", "_token", "_password", "_passwd", "_secret",
    "_credential", "_credentials",
    "_private_key", "_пароль", "_секрет", "_токен",
)

WORKING_PREFIX = (
    "Рабочая память текущей задачи. Это явно сохранённые цель, ограничения, "
    "промежуточные данные и открытые вопросы. Учитывай их в ответе:\n\n"
)
# Старое публичное имя оставлено для совместимости импортов и расширений.
PROFILE_PREFIX = PROFILE_PROMPT
PROJECT_PREFIX = (
    "Долговременная память этого проекта. Соблюдай правила и запреты, учитывай "
    "принятые решения и устойчивые знания:\n\n"
)

FILE_MODE = 0o600
DIR_MODE = 0o700
FORMAT_VERSION = 1
DEFAULT_ROOT = pathlib.Path("~/.llm-agent/memory")


def _clean_key(value: str) -> str:
    return ".".join(part.strip() for part in str(value or "").split(".") if part.strip())


def _clean_value(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass
class WorkingMemory:
    """Данные одной текущей задачи, отделённые от реплик диалога."""

    items: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.items = _strings(self.items)

    def remember(self, key: str, value: str) -> bool:
        name = _clean_key(key)
        if not name:
            raise AgentError("у записи памяти должен быть ключ")
        _reject_secret_key(name)
        text = _clean_value(value)
        if not text:
            return self.forget(name)
        changed = self.items.get(name) != text
        self.items[name] = text
        return changed

    def forget(self, key: str) -> bool:
        name = _clean_key(key)
        if not name:
            raise AgentError("надо назвать ключ, который следует забыть")
        return self.items.pop(name, None) is not None

    def clear(self) -> None:
        self.items.clear()

    def prompt_message(self) -> Optional[Dict[str, str]]:
        items = _strings(self.items)
        if not items:
            return None
        return {"role": "user", "content": WORKING_PREFIX + _block(items)}

    def to_dict(self) -> Dict[str, str]:
        return _strings(self.items)

    @classmethod
    def restore(cls, payload: Any) -> "WorkingMemory":
        if not isinstance(payload, dict):
            return cls()
        # Читаем и новый вид {"items": {...}}, и компактный словарь первой
        # реализации: это делает формат терпимым к ранним файлам.
        raw = payload.get("items") if isinstance(payload.get("items"), dict) else payload
        return cls({_clean_key(key): _clean_value(value) for key, value in raw.items()
                    if _clean_key(key) and _clean_value(value)})


@dataclass
class LongTermMemory:
    """Профиль пользователя и память конкретного проекта между сессиями."""

    profile: UserProfile = field(default_factory=UserProfile)
    rules: Dict[str, str] = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)
    knowledge: Dict[str, str] = field(default_factory=dict)
    project_root: str = ""
    profile_revision: int = 0
    project_revision: int = 0

    def __post_init__(self) -> None:
        self.profile = UserProfile(_strings(self.profile))
        for name in (RULES, DECISIONS, KNOWLEDGE):
            setattr(self, name, _strings(getattr(self, name)))

    def section(self, name: str) -> Dict[str, str]:
        if name not in LONG_TERM_SECTIONS:
            raise AgentError(
                "неизвестный раздел долговременной памяти «{}»; доступны: {}".format(
                    name, ", ".join(LONG_TERM_SECTIONS)))
        return getattr(self, name)

    def remember(self, section: str, key: str, value: str) -> bool:
        bucket = self.section(section)
        name = _clean_key(key)
        if not name:
            raise AgentError("у записи памяти должен быть ключ")
        _reject_secret_key(name)
        text = _clean_value(value)
        if not text:
            return self.forget(section, name)
        changed = bucket.get(name) != text
        bucket[name] = text
        return changed

    def forget(self, section: str, key: str) -> bool:
        name = _clean_key(key)
        if not name:
            raise AgentError("надо назвать ключ, который следует забыть")
        return self.section(section).pop(name, None) is not None

    def profile_message(self, request: str = "") -> Optional[Dict[str, str]]:
        # Фильтруем ещё раз на границе промпта: даже прямое изменение mapping
        # в обход remember() не должно протащить секрет в запрос.
        return UserProfile(_strings(self.profile)).prompt_message(request)

    def prompt_messages(self, request: str = "") -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        profile = self.profile_message(request)
        if profile is not None:
            messages.append(profile)
        project = []
        for section in (RULES, DECISIONS, KNOWLEDGE):
            items = _strings(self.section(section))
            if items:
                project.append("{}:\n{}".format(section, _block(items)))
        if project:
            messages.append({"role": "user", "content": PROJECT_PREFIX + "\n\n".join(project)})
        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": _strings(self.profile),
            "rules": _strings(self.rules),
            "decisions": _strings(self.decisions),
            "knowledge": _strings(self.knowledge),
            "project_root": self.project_root,
        }


def _block(items: Dict[str, str]) -> str:
    return "\n".join("- {}: {}".format(key, value) for key, value in items.items())


class MemoryStore:
    """Отдельные защищённые файлы профиля и памяти текущего проекта."""

    def __init__(self, root: Optional[str] = None,
                 project_root: Optional[str] = None) -> None:
        self.root = pathlib.Path(os.path.expanduser(root or str(DEFAULT_ROOT)))
        self.project_root = _project_root(project_root)

    @property
    def profile_path(self) -> pathlib.Path:
        return self.root / "profile.json"

    @property
    def project_path(self) -> pathlib.Path:
        digest = hashlib.sha256(self.project_root.encode("utf-8")).hexdigest()[:16]
        return self.root / "projects" / (digest + ".json")

    @property
    def lock_path(self) -> pathlib.Path:
        return self.root / ".lock"

    def load(self) -> LongTermMemory:
        # Чтение не должно само создавать ~/.llm-agent/memory и lock-файл:
        # one-shot и под-агенты до первой явной записи остаются read-only.
        # Каждый data-файл заменяется атомарно, поэтому отдельное чтение
        # профиля и проекта не видит наполовину записанный JSON.
        return self._load_unlocked()

    def _load_unlocked(self) -> LongTermMemory:
        profile_payload = self._read(self.profile_path)
        profile = profile_payload.get("items") or {}
        project = self._read(self.project_path)
        return LongTermMemory(
            profile=_strings(profile),
            rules=_strings(project.get(RULES)),
            decisions=_strings(project.get(DECISIONS)),
            knowledge=_strings(project.get(KNOWLEDGE)),
            project_root=self.project_root,
            profile_revision=int(profile_payload.get("revision") or 0),
            project_revision=int(project.get("revision") or 0),
        )

    def remember(self, section: str, key: str, value: str):
        """Атомарно изменить один ключ поверх самого свежего снимка."""
        with self._locked():
            current = self._load_unlocked()
            changed = current.remember(section, key, value)
            if changed:
                if section == PROFILE:
                    current.profile_revision += 1
                    self._write_profile(current)
                else:
                    current.project_revision += 1
                    self._write_project(current)
            return current, changed

    def save(self, memory: LongTermMemory) -> None:
        """Записать полный снимок, отказав устаревшему вместо потери данных."""
        with self._locked():
            current = self._load_unlocked()
            if (memory.profile_revision != current.profile_revision
                    or memory.project_revision != current.project_revision):
                raise AgentError(
                    "долговременная память изменилась в другом процессе; "
                    "перечитайте её и повторите запись")
            memory.profile_revision += 1
            memory.project_revision += 1
            self._write_profile(memory)
            self._write_project(memory)

    def _write_profile(self, memory: LongTermMemory) -> None:
        self._write(self.profile_path, {
            "version": FORMAT_VERSION,
            "revision": memory.profile_revision,
            "items": _strings(memory.profile),
        })

    def _write_project(self, memory: LongTermMemory) -> None:
        self._write(self.project_path, {
            "version": FORMAT_VERSION,
            "revision": memory.project_revision,
            "project_root": self.project_root,
            RULES: _strings(memory.rules),
            DECISIONS: _strings(memory.decisions),
            KNOWLEDGE: _strings(memory.knowledge),
        })

    @contextmanager
    def _locked(self):
        """Согласовать профиль и проект между процессами одним lock-файлом."""
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, DIR_MODE)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            os.chmod(self.lock_path, FILE_MODE)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write(self, target: pathlib.Path, payload: Dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Закрываем и корень, и projects: в имени файла нет данных, но сам факт
        # существования проектной памяти тоже не должен быть общедоступным.
        os.chmod(self.root, DIR_MODE)
        os.chmod(target.parent, DIR_MODE)
        temporary = target.with_suffix(".{}.{}.tmp".format(
            os.getpid(), uuid.uuid4().hex[:8]))
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.chmod(temporary, FILE_MODE)
        os.replace(temporary, target)

    @staticmethod
    def _read(path: pathlib.Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise AgentError("файл памяти {} не прочитан: {}".format(path, exc)) from exc
        return payload if isinstance(payload, dict) else {}


def _strings(payload: Any) -> Dict[str, str]:
    if isinstance(payload, UserProfile):
        payload = payload.to_dict()
    if not isinstance(payload, dict):
        return {}
    return {_clean_key(key): _clean_value(value) for key, value in payload.items()
            if (_clean_key(key) and _clean_value(value)
                and not _is_sensitive_key(_clean_key(key)))}


def _reject_secret_key(key: str) -> None:
    if _is_sensitive_key(key):
        raise AgentError(
            "секреты нельзя сохранять в памяти; используйте хранилище ключей проекта")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(".", "_")
    return normalized in SENSITIVE_KEYS or normalized.endswith(SENSITIVE_SUFFIXES)


def _project_root(given: Optional[str]) -> str:
    """Стабильная область проектной памяти без зависимости от GitPython."""
    path = pathlib.Path(given or os.getcwd()).expanduser().resolve()
    for candidate in (path,) + tuple(path.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return str(path)
