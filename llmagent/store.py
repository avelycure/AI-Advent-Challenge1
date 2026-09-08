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
import uuid
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
class Saved:
    """Куда и под каким именем записалась сессия."""

    path: pathlib.Path
    session_id: str
    # Время этой записи. Вызывающий держит его при себе и передаёт обратно:
    # только так своя запись не принимается за чужую правку.
    updated_at: float = 0.0
    # Непусто, если запись отделилась от прежней сессии: её успел изменить
    # кто-то другой, и писать поверх значило бы потерять чужую работу.
    forked_from: str = ""


@dataclass
class SessionRecord:
    session_id: str
    # Имя самой сессии — то, по которому её узнаёт человек. Отдельно от
    # ``agent`` ниже: то имя принадлежит конфигу и говорит, каким агент был,
    # а это говорит, о чём был разговор.
    title: str = ""
    agent: str = ""
    topic: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    conversation: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    # Место в полном списке. Именно оно показывается человеку и принимается
    # командами: номер, зависящий от того, искали мы или нет, удалял бы не ту
    # сессию — проверено, и это худшее, что тут может случиться.
    number: int = 0

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

    @property
    def first_question(self) -> str:
        """Первый вопрос человека — по нему разговор узнаётся вернее всего."""
        for message in self.messages:
            if message.get("role") == "user" and not message.get("note"):
                return " ".join(str(message.get("content", "")).split())
        return ""

    @property
    def label(self) -> str:
        """Как называть сессию в списке. Имя, иначе тема, иначе первый вопрос."""
        return self.title or self.topic or self.first_question or "без имени"

    def mentions(self, needle: str) -> bool:
        """Встречается ли слово в имени, теме или самой переписке.

        Искать по переписке обязательно: через неделю человек помнит, о чём
        говорил, а не как назвал сессию.
        """
        wanted = needle.strip().lower()
        if not wanted:
            return True
        # Ни модель, ни идентификатор в поиск не входят: по слову «gpt»
        # находилось бы решительно всё, и список терял бы смысл.
        haystack = [self.title, self.topic]
        haystack += [str(message.get("content", "")) for message in self.messages]
        return any(wanted in str(item).lower() for item in haystack)

    def to_config(self) -> AgentConfig:
        return AgentConfig.from_dict(self.config)


class SessionStore:
    """Каталог сессий. Всё общение с диском — только здесь."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = pathlib.Path(os.path.expanduser(root or str(DEFAULT_ROOT)))

    # --- запись ---------------------------------------------------------
    def save(self, agent, topic: str = "", title: str = "",
             seen_at: Optional[float] = None) -> Saved:
        """Записать состояние сессии. Пустую сессию не сохраняем.

        Запись идёт во временный файл с переименованием: прерванная на середине
        запись не должна оставить обрубок вместо разговора.

        ``seen_at`` — время последней правки, известное вызывающему. Если на
        диске оказалось свежее, значит ту же сессию ведёт кто-то ещё: два
        терминала легко открыть одной командой. Писать поверх нельзя — чужой
        разговор пропал бы молча, — поэтому сессия отделяется под новым
        идентификатором, а вызывающему об этом сообщается.
        """
        if not agent.conversation.messages:
            return Saved(self.path_for(agent.session_id), agent.session_id, seen_at or 0.0)

        forked_from = ""
        if seen_at is not None and self._changed_since(agent.session_id, seen_at):
            forked_from = agent.session_id
            agent.session_id = uuid.uuid4().hex[:8]

        target = self.path_for(agent.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, DIR_MODE)

        created = time.time()
        if target.exists():
            try:
                created = float(self._read(target).get("created_at", created))
            except (OSError, ValueError):
                pass

        stamp = time.time()
        payload = {
            "version": FORMAT_VERSION,
            "session_id": agent.session_id,
            "title": title,
            "agent": agent.config.name,
            "topic": topic,
            "created_at": created,
            "updated_at": stamp,
            "config": _config_without_secrets(agent.config),
            "conversation": agent.conversation.to_dict(),
            "usage": agent.usage.to_dict(),
        }

        # Своё имя у каждого процесса: общий временный файл сводил бы на нет
        # всю затею с атомарной записью ровно там, где она нужнее всего.
        temporary = target.with_suffix(".{}.tmp".format(os.getpid()))
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.chmod(temporary, FILE_MODE)
        os.replace(temporary, target)
        return Saved(target, agent.session_id, stamp, forked_from)

    def _changed_since(self, session_id: str, seen_at: float) -> bool:
        """Изменился ли файл сессии после того, как его прочитали.

        Сравнение точное, без допуска. Допуск был бы дырой: чужая запись,
        случившаяся в те же полсекунды, проходила бы незамеченной и её работа
        пропадала бы. Своя предыдущая запись за чужую не принимается, потому
        что вызывающий берёт ``seen_at`` из её же итога.
        """
        try:
            target = self.path_for(session_id)
        except AgentError:
            return False
        if not target.is_file():
            return False
        try:
            stamp = float(self._read(target).get("updated_at") or 0.0)
        except (OSError, ValueError):
            return False
        return stamp > seen_at

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

    def find(self, wanted: str) -> SessionRecord:
        """Найти сессию по номеру из списка, имени или части идентификатора.

        Человек помнит имя или место в списке, а не восемь шестнадцатеричных
        знаков. Поэтому принимаются все три способа, а неоднозначность —
        ошибка с перечнем подходящих, а не молчаливый выбор первого.
        """
        asked = wanted.strip()
        if not asked:
            raise SessionNotFound("не названа сессия")

        recent = self.recent(limit=200)
        if not recent:
            raise SessionNotFound("сохранённых сессий нет")

        # Точное совпадение с идентификатором идёт раньше номера: у сессии
        # идентификатор может оказаться из одних цифр (примерно одна из
        # сорока), и она перестала бы открываться по тому самому номеру,
        # который ей показывают в шапке.
        exact = [r for r in recent if r.session_id == asked or r.title == asked]
        if len(exact) == 1:
            return exact[0]

        if asked.isdigit():
            number = int(asked)
            for record in recent:
                if record.number == number:
                    return record
            raise SessionNotFound("нет сессии под номером {}: в списке их {}".format(
                number, len(recent)))
        lowered = asked.lower()
        found = [r for r in recent if r.session_id.startswith(lowered)
                 or lowered in (r.title or "").lower()
                 or lowered in (r.topic or "").lower()]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise SessionNotFound(
                "«{}» подходит нескольким сессиям: {}. Уточните номером "
                "из --sessions".format(asked, ", ".join(
                    "{} ({})".format(r.session_id, r.label[:24]) for r in found[:4])))

        known = ", ".join("{} ({})".format(r.session_id, r.label[:20])
                          for r in recent[:3])
        raise SessionNotFound("нет сессии «{}»{}".format(
            asked, ". Недавние: " + known if known else ""))

    def matching(self, needle: str, limit: int = 50) -> List[SessionRecord]:
        """Сессии, где встречается слово — в имени, теме или переписке."""
        return [record for record in self.recent(limit=200)
                if record.mentions(needle)][:limit]

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
        # Порядок должен быть устойчивым: две сессии, записанные в одно
        # мгновение, иначе перескакивали бы в списке между запусками, а номер
        # из списка перестал бы значить одно и то же.
        records.sort(key=lambda record: (record.updated_at, record.created_at,
                                         record.session_id), reverse=True)
        for place, record in enumerate(records, 1):
            record.number = place
        return records[:limit]

    def _record(self, path: pathlib.Path) -> SessionRecord:
        payload = self._read(path)
        return SessionRecord(
            session_id=str(payload.get("session_id") or path.stem),
            title=str(payload.get("title") or ""),
            # Прежние файлы держали в «name» имя конфига: читаем и его, чтобы
            # сессии, записанные до этой правки, не потеряли подпись.
            agent=str(payload.get("agent") or payload.get("name") or ""),
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
        """Удалить сессию. Имя и номер годятся так же, как идентификатор."""
        target = self.path_for(session_id) if _looks_technical(session_id) else None
        if target is None or not target.is_file():
            try:
                target = pathlib.Path(self.find(session_id).path or "")
            except SessionNotFound as exc:
                # Про неоднозначность надо сказать, а про отсутствие — вернуть
                # False: вызывающий различает эти случаи по-разному.
                if "подходит нескольким" in str(exc):
                    raise
                return False
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


def _looks_technical(value: str) -> bool:
    """Похоже ли на идентификатор: короткая шестнадцатеричная строка."""
    stripped = value.strip()
    return bool(stripped) and all(ch in "0123456789abcdef" for ch in stripped.lower())


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
