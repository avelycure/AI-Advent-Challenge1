"""Сессии на диске: продолжить разговор после закрытия терминала.

Одна сессия — один файл. В файле лежит переписка целиком, счётчики расхода и
конфиг, с которым разговор шёл. Файл читается только владельцем, и лежит он до
тех пор, пока его не удалят: ``--sessions`` показывает список, ``--rm-session``
удаляет.

Реквизиты в файл не попадают. Ключ провайдера при сохранении обнуляется, а при
возврате в сессию берётся заново — из окружения или из файла рядом. Иначе копия
ключа расползлась бы по каталогу сессий, и удалять её пришлось бы вручную.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .errors import AgentError
from .history import Conversation
from .usage import UsageMeter

DEFAULT_ROOT = pathlib.Path("~/.llm-agent/sessions")
FILE_MODE = 0o600      # переписка — личное дело владельца
DIR_MODE = 0o700
FORMAT_VERSION = 1


class SessionNotFound(AgentError):
    """Сессии с таким идентификатором на диске нет."""


@dataclass
class SessionRecord:
    session_id: str
    name: str = ""
    topic: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    conversation: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self.conversation.get("messages") or [])

    @property
    def exchanges(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")

    @property
    def total_tokens(self) -> int:
        spent = (self.usage.get("by_kind") or {}).values()
        return sum(int(s.get("prompt_tokens", 0)) + int(s.get("completion_tokens", 0))
                   for s in spent)

    @property
    def costs(self) -> Dict[str, float]:
        return dict(self.usage.get("costs") or {})

    @property
    def model(self) -> str:
        return str(self.config.get("model", ""))

    def to_config(self) -> AgentConfig:
        return AgentConfig.from_dict(self.config)


class SessionStore:
    """Каталог сессий. Всё общение с диском — только здесь."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = pathlib.Path(os.path.expanduser(root or str(DEFAULT_ROOT)))

    # --- запись ---------------------------------------------------------
    def save(self, agent, topic: str = "") -> pathlib.Path:
        """Записать состояние сессии. Пустую сессию не сохраняем.

        Запись идёт во временный файл с переименованием: прерванная на середине
        запись не должна оставить обрубок вместо разговора.
        """
        if not agent.conversation.messages:
            return self.path_for(agent.session_id)

        target = self.path_for(agent.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, DIR_MODE)

        created = time.time()
        if target.exists():
            try:
                created = float(self._read(target).get("created_at", created))
            except (OSError, ValueError):
                pass

        payload = {
            "version": FORMAT_VERSION,
            "session_id": agent.session_id,
            "name": agent.config.name,
            "topic": topic,
            "created_at": created,
            "updated_at": time.time(),
            "config": _config_without_secrets(agent.config),
            "conversation": agent.conversation.to_dict(),
            "usage": agent.usage.to_dict(),
        }

        temporary = target.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.chmod(temporary, FILE_MODE)
        os.replace(temporary, target)
        return target

    # --- чтение ---------------------------------------------------------
    def path_for(self, session_id: str) -> pathlib.Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
        if not safe:
            raise AgentError("негодный идентификатор сессии: «{}»".format(session_id))
        return self.root / (safe + ".json")

    def load(self, session_id: str) -> SessionRecord:
        target = self.path_for(session_id)
        if not target.is_file():
            known = [record.session_id for record in self.recent(5)]
            raise SessionNotFound("нет сессии «{}»{}".format(
                session_id,
                ". Недавние: " + ", ".join(known) if known else
                "; сохранённых сессий пока нет"))
        return self._record(target)

    def latest(self) -> Optional[SessionRecord]:
        recent = self.recent(1)
        return recent[0] if recent else None

    def recent(self, limit: int = 20) -> List[SessionRecord]:
        """Сессии от свежих к старым. Испорченные файлы пропускаются."""
        if not self.root.is_dir():
            return []
        records: List[SessionRecord] = []
        for path in self.root.glob("*.json"):
            try:
                records.append(self._record(path))
            except (OSError, ValueError):
                continue
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[:limit]

    def _record(self, path: pathlib.Path) -> SessionRecord:
        payload = self._read(path)
        return SessionRecord(
            session_id=str(payload.get("session_id") or path.stem),
            name=str(payload.get("name") or ""),
            topic=str(payload.get("topic") or ""),
            created_at=float(payload.get("created_at") or 0.0),
            updated_at=float(payload.get("updated_at") or 0.0),
            config=dict(payload.get("config") or {}),
            conversation=dict(payload.get("conversation") or {}),
            usage=dict(payload.get("usage") or {}),
            path=str(path),
        )

    @staticmethod
    def _read(path: pathlib.Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("в файле сессии ожидался объект")
        return payload

    # --- удаление -------------------------------------------------------
    def remove(self, session_id: str) -> bool:
        target = self.path_for(session_id)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def clear(self) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink()
            removed += 1
        return removed


def restore_agent(record: SessionRecord, config: AgentConfig, **kwargs):
    """Собрать агента, вернув ему память и счётчики сохранённой сессии.

    Конфиг передаётся отдельно, а не берётся из записи: при возврате в сессию
    поверх сохранённого конфига могли лечь правки из командной строки.
    """
    from .agent import Agent

    agent = Agent(config, session_id=record.session_id, **kwargs)
    if record.conversation:
        agent.conversation = Conversation.restore(record.conversation)
        agent.conversation.keep_last_answer = config.history.keep_last_answer
        stored = record.config
        if (stored.get("provider"), stored.get("model")) != (config.provider, config.model):
            # Точный размер контекста измерен токенизатором прежней модели.
            # После смены он неверен, и полоса заполнения вместе с проверкой
            # «помещается ли ещё вопрос» соврали бы.
            agent.conversation.forget_exact()
    if record.usage:
        agent.usage = UsageMeter.restore(record.usage)
    return agent


def _config_without_secrets(config: AgentConfig) -> Dict[str, Any]:
    """Конфиг для записи: реквизиты обнулены, а не замаскированы звёздочками.

    Именно обнулены: строка «***» при возврате в сессию попала бы в заголовок
    запроса как настоящий ключ, и провайдер отказал бы с непонятной причиной.
    """
    payload = config.to_dict(with_secrets=False)
    for name in ("api_key", "api_extra"):
        payload[name] = None
    judge = payload.get("judge")
    if isinstance(judge, dict) and isinstance(judge.get("agent"), dict):
        for name in ("api_key", "api_extra"):
            judge["agent"][name] = None
    return payload
