"""Ветки разговора: два продолжения от одного места.

Смысл в том, чего нельзя сделать линейным диалогом. На середине сбора
требований выясняется, что стек можно выбрать двумя способами, и оба надо
обсудить. В линейном разговоре второй вариант обсуждается после первого —
и модель отвечает уже с оглядкой на него: «как мы решили выше». Ветка
отводится от того же места и не видит ничего, что было сказано в соседней.

Точка возврата хранит снимок переписки целиком, а не номер сообщения. Номер
казался дешевле, но он не выдерживает обрезки: стоило забыть начало
разговора ради места в окне, и «шестое сообщение» указывало уже на другое.
Снимок — это текст, он стоит недорого и не врёт.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import AgentError
from .history import Conversation
from .memory import WorkingMemory
from .task_state import TaskState

# Имя ветки, с которой начинается любой разговор. Она есть всегда, даже когда
# о ветках никто не спрашивал: так ветвление ничего не меняет в том, как
# выглядит обычный линейный диалог.
MAIN = "основная"


@dataclass
class Checkpoint:
    """Место в разговоре, от которого можно отвести ветку."""

    label: str
    # Где поставлена и сколько сообщений было к тому времени — для показа.
    branch: str = MAIN
    messages: int = 0
    created_at: float = 0.0
    # Снимок переписки на эту минуту: сообщения, пересказ и факты.
    conversation: Conversation = field(default_factory=Conversation)
    # Рабочая память относится к той же задаче и ветвится вместе с разговором.
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    task_state: TaskState = field(default_factory=TaskState)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "branch": self.branch,
                "messages": self.messages, "created_at": self.created_at,
                "conversation": self.conversation.to_dict(),
                "working_memory": self.working_memory.to_dict(),
                "task_state": self.task_state.to_dict()}

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "Checkpoint":
        return cls(
            label=str(payload.get("label") or ""),
            branch=str(payload.get("branch") or MAIN),
            messages=int(payload.get("messages") or 0),
            created_at=float(payload.get("created_at") or 0.0),
            conversation=Conversation.restore(payload.get("conversation") or {}),
            working_memory=WorkingMemory.restore(payload.get("working_memory") or {}),
            task_state=TaskState.restore(payload.get("task_state") or {}))


@dataclass
class Branch:
    """Одна ветка разговора: своя переписка и память о том, откуда отведена."""

    name: str
    conversation: Conversation = field(default_factory=Conversation)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    task_state: TaskState = field(default_factory=TaskState)
    # От какой ветки и от какой точки отведена. Пусто у основной: она ни от
    # чего не отведена, разговор с неё начался.
    parent: str = ""
    checkpoint: str = ""
    # Сколько сообщений унаследовано от родителя. По этому числу видно, где
    # начинается своя часть ветки, — и его же показывает ``/branches``.
    inherited: int = 0
    created_at: float = 0.0

    @property
    def own(self) -> int:
        """Сколько сообщений сказано уже в этой ветке, а не унаследовано."""
        return max(0, len(self.conversation.messages) - self.inherited)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "parent": self.parent,
                "checkpoint": self.checkpoint, "inherited": self.inherited,
                "created_at": self.created_at,
                "conversation": self.conversation.to_dict(),
                "working_memory": self.working_memory.to_dict(),
                "task_state": self.task_state.to_dict()}

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "Branch":
        return cls(
            name=str(payload.get("name") or MAIN),
            conversation=Conversation.restore(payload.get("conversation") or {}),
            working_memory=WorkingMemory.restore(payload.get("working_memory") or {}),
            task_state=TaskState.restore(payload.get("task_state") or {}),
            parent=str(payload.get("parent") or ""),
            checkpoint=str(payload.get("checkpoint") or ""),
            inherited=int(payload.get("inherited") or 0),
            created_at=float(payload.get("created_at") or 0.0))


class Branches:
    """Все ветки разговора и точки возврата. Активна ровно одна ветка."""

    def __init__(self, conversation: Optional[Conversation] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 task_state: Optional[TaskState] = None) -> None:
        root = Branch(MAIN, conversation if conversation is not None else Conversation(),
                      working_memory if working_memory is not None else WorkingMemory(),
                      task_state if task_state is not None else TaskState(),
                      created_at=time.time())
        self._branches: Dict[str, Branch] = {MAIN: root}
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._active = MAIN

    # --- чтение ---------------------------------------------------------
    @property
    def active(self) -> str:
        return self._active

    @property
    def conversation(self) -> Conversation:
        """Переписка активной ветки — та, с которой работает агент."""
        return self._branches[self._active].conversation

    @conversation.setter
    def conversation(self, value: Conversation) -> None:
        self._branches[self._active].conversation = value

    @property
    def working_memory(self) -> WorkingMemory:
        return self._branches[self._active].working_memory

    @working_memory.setter
    def working_memory(self, value: WorkingMemory) -> None:
        self._branches[self._active].working_memory = value

    @property
    def task_state(self) -> TaskState:
        return self._branches[self._active].task_state

    @task_state.setter
    def task_state(self, value: TaskState) -> None:
        self._branches[self._active].task_state = value

    @property
    def branched(self) -> bool:
        """Разветвлён ли разговор. Ложь — ветка одна, и о ветках можно молчать."""
        return len(self._branches) > 1

    def all(self) -> List[Branch]:
        return list(self._branches.values())

    def checkpoints(self) -> List[Checkpoint]:
        return list(self._checkpoints.values())

    def get(self, name: str) -> Branch:
        branch = self._branches.get(_clean(name))
        if branch is None:
            raise AgentError("нет ветки «{}»; есть: {}".format(
                name, ", ".join(self._branches)))
        return branch

    def find(self, wanted: str) -> Branch:
        """Ветка по имени или по номеру из списка ``/branches``."""
        name = _clean(wanted)
        if name in self._branches:
            return self._branches[name]
        if name.isdigit():
            order = self.all()
            number = int(name)
            if 1 <= number <= len(order):
                return order[number - 1]
        raise AgentError("нет ветки «{}»; есть: {}".format(
            wanted, ", ".join(self._branches)))

    def checkpoint(self, label: str) -> Checkpoint:
        point = self._checkpoints.get(_clean(label))
        if point is None:
            raise AgentError("нет точки «{}»{}".format(
                label, "; есть: " + ", ".join(self._checkpoints)
                if self._checkpoints else "; ни одной ещё не поставлено"))
        return point

    def latest_checkpoint(self) -> Optional[Checkpoint]:
        points = self.checkpoints()
        return points[-1] if points else None

    # --- изменение ------------------------------------------------------
    def mark(self, label: str = "") -> Checkpoint:
        """Запомнить текущее место разговора под именем.

        Имя придумывается само, если его не задали: точка без имени бесполезна,
        а по ней потом отводят ветку.
        """
        name = _clean(label) or "точка-{}".format(len(self._checkpoints) + 1)
        if name in self._checkpoints:
            raise AgentError("точка «{}» уже есть: выберите другое имя, "
                             "чтобы не потерять прежнюю".format(name))
        conversation = self.conversation
        point = Checkpoint(name, branch=self._active, messages=len(conversation.messages),
                           created_at=time.time(),
                           conversation=copy.deepcopy(conversation),
                           working_memory=copy.deepcopy(self.working_memory),
                           task_state=copy.deepcopy(self.task_state))
        self._checkpoints[name] = point
        return point

    def fork(self, name: str, label: str = "") -> Branch:
        """Отвести ветку от точки возврата и сделать её активной.

        Без точки ветка отводится от текущего места разговора — и точка под
        него ставится сама: иначе от того же места не отвести вторую ветку,
        а именно это в ветвлении и нужно.
        """
        wanted = _clean(name)
        if not wanted:
            raise AgentError("у ветки должно быть имя")
        if wanted in self._branches:
            raise AgentError("ветка «{}» уже есть".format(wanted))
        point = self.checkpoint(label) if label else (
            self.latest_checkpoint() or self.mark())
        branch = Branch(wanted, conversation=copy.deepcopy(point.conversation),
                        working_memory=copy.deepcopy(point.working_memory),
                        task_state=copy.deepcopy(point.task_state),
                        parent=point.branch, checkpoint=point.label,
                        inherited=len(point.conversation.messages),
                        created_at=time.time())
        self._branches[wanted] = branch
        self._active = wanted
        return branch

    def switch(self, wanted: str) -> Branch:
        branch = self.find(wanted)
        self._active = branch.name
        return branch

    def reset(self) -> None:
        """Оставить один разговор без ветвей: так ведёт себя ``/new``."""
        conversation = self.conversation
        conversation.reset()
        self._branches = {MAIN: Branch(MAIN, conversation, WorkingMemory(), TaskState(),
                                       created_at=time.time())}
        self._checkpoints = {}
        self._active = MAIN

    def each_conversation(self):
        """Все переписки: и ветвей, и снимков в точках возврата.

        Нужно тому, что относится ко всем сразу, — например к смене модели:
        измеренное её токенизатором неверно во всех ветвях, а не только
        в активной.
        """
        for branch in self._branches.values():
            yield branch.conversation
        for point in self._checkpoints.values():
            yield point.conversation

    # --- сохранение -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "branches": [branch.to_dict() for branch in self._branches.values()],
            "checkpoints": [point.to_dict() for point in self._checkpoints.values()],
        }

    @classmethod
    def restore(cls, payload: Dict[str, Any], conversation: Conversation,
                working_memory: Optional[WorkingMemory] = None,
                task_state: Optional[TaskState] = None) -> "Branches":
        """Собрать ветки из сохранённого вида.

        ``conversation`` — активная переписка, прочитанная обычным путём. Она
        и остаётся активной веткой: файл мог быть записан версией, которая про
        ветки не знала, и тогда ветка ровно одна.
        """
        branches = cls(conversation, working_memory, task_state)
        stored = payload.get("branches") if isinstance(payload, dict) else None
        if not stored:
            return branches
        active = _clean(str(payload.get("active") or MAIN))
        restored: Dict[str, Branch] = {}
        for item in stored:
            if not isinstance(item, dict):
                continue
            branch = Branch.restore(item)
            # Активную ветку берём из уже прочитанной переписки, а не из своей
            # копии: над ней вызывающий мог поработать — например, забыть
            # измеренное чужой моделью.
            if branch.name == active:
                branch.conversation = conversation
                branch.working_memory = working_memory or WorkingMemory()
                branch.task_state = task_state or TaskState()
            restored[branch.name] = branch
        if not restored:
            return branches
        branches._branches = restored
        branches._active = active if active in restored else next(iter(restored))
        for item in payload.get("checkpoints") or []:
            if isinstance(item, dict):
                point = Checkpoint.restore(item)
                if point.label:
                    branches._checkpoints[point.label] = point
        return branches


def _clean(name: str) -> str:
    return " ".join(str(name or "").split())
