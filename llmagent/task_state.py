"""Формализованное состояние текущей задачи.

Состояние хранит три вещи, которых недостаточно в обычной переписке:
этап, конкретный текущий шаг и действие, ожидаемое дальше. Пауза — отдельный
флаг, поэтому она не разрушает этап и после продолжения агент оказывается
ровно в той же точке.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .errors import AgentError

PLANNING = "planning"
EXECUTION = "execution"
VALIDATION = "validation"
DONE = "done"
TASK_STAGES = (PLANNING, EXECUTION, VALIDATION, DONE)

NEXT_STAGE = {
    PLANNING: EXECUTION,
    EXECUTION: VALIDATION,
    VALIDATION: DONE,
}

TASK_STATE_PREFIX = (
    "Формализованное состояние текущей задачи. Используй его как источник "
    "истины: не проси повторно цель и уже сохранённый контекст.\n\n"
)

TASK_STATE_PATTERN = re.compile(
    r"\s*\[\[TASK_STATE\s+(\{.*?\})\s*\]\]\s*$", re.DOTALL)
AUTOMATIC_CONTROL = (
    "\n\nТы сам управляешь конечным автоматом этой задачи. После содержательного "
    "ответа обязательно добавь последней строкой служебную директиву строго "
    "такого вида (она будет скрыта от пользователя):\n"
    '[[TASK_STATE {"stage":"execution","current_step":"что делать теперь",'
    '"expected_action":"следующее действие","paused":false}]]\n'
    "stage должен показывать фактически достигнутый после ответа этап цепочки "
    "planning → execution → validation → done. Программа проведёт переход "
    "через все промежуточные этапы. "
    "current_step описывает актуальный шаг после твоего ответа, а "
    "expected_action — конкретное действие следующим. paused=true ставь, "
    "только если без ответа или решения пользователя продолжать нельзя. "
    "Как только требуемое решение создано, переходи как минимум в validation; "
    "не оставайся в execution ради необязательных улучшений. После собственной "
    "проверки результата переходи в done."
)

STATE_REVIEW_MAX_CHARS = 8000


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass
class TaskState:
    """Конечный автомат ``planning → execution → validation → done``."""

    stage: str = PLANNING
    current_step: str = ""
    expected_action: str = ""
    paused: bool = False
    active: bool = False

    def __post_init__(self) -> None:
        if self.stage not in TASK_STAGES:
            raise AgentError("неизвестный этап задачи «{}»; доступны: {}".format(
                self.stage, ", ".join(TASK_STAGES)))
        self.current_step = _clean(self.current_step)
        self.expected_action = _clean(self.expected_action)
        self.paused = bool(self.paused) and bool(self.active)

    @property
    def status(self) -> str:
        if not self.active:
            return "inactive"
        return "paused" if self.paused else self.stage

    @property
    def next_stage(self) -> str:
        return NEXT_STAGE.get(self.stage, "")

    def start(self, current_step: str = "", expected_action: str = "") -> None:
        """Начать новую задачу с этапа планирования."""
        self.stage = PLANNING
        self.current_step = _clean(current_step)
        self.expected_action = _clean(expected_action)
        self.paused = False
        self.active = True

    def start_from_request(self, request: str) -> None:
        """Автоматически принять первую реплику как новую задачу."""
        task = _clean(request)
        if len(task) > 240:
            task = task[:237].rstrip() + "…"
        self.start(task or "Определить задачу",
                   "Составить план и приступить к выполнению")

    def update(self, current_step: Optional[str] = None,
               expected_action: Optional[str] = None) -> None:
        """Уточнить шаг внутри этапа, не выполняя переход."""
        self._require_active()
        if current_step is not None:
            self.current_step = _clean(current_step)
        if expected_action is not None:
            self.expected_action = _clean(expected_action)

    def transition(self, stage: str, current_step: Optional[str] = None,
                   expected_action: Optional[str] = None) -> None:
        """Перейти в тот же или строго следующий этап автомата."""
        self._require_active()
        wanted = _clean(stage).lower()
        if self.paused:
            raise AgentError("задача на паузе; сначала продолжите её")
        allowed = (self.stage, self.next_stage) if self.next_stage else (self.stage,)
        if wanted not in allowed:
            if not self.next_stage:
                raise AgentError("этап done завершающий; начните новую задачу")
            raise AgentError(
                "из {} можно перейти только в {}; пропускать этапы и идти "
                "назад нельзя".format(self.stage, self.next_stage))
        self.stage = wanted
        if current_step is not None:
            self.current_step = _clean(current_step)
        if expected_action is not None:
            self.expected_action = _clean(expected_action)

    def advance(self, current_step: str = "", expected_action: str = "") -> None:
        """Перейти в следующий этап без повторения его имени."""
        self._require_active()
        if not self.next_stage:
            raise AgentError("этап done завершающий; начните новую задачу")
        self.transition(self.next_stage, current_step, expected_action)

    def pause(self) -> None:
        """Приостановить задачу, сохранив этап, шаг и ожидаемое действие."""
        self._require_active()
        if self.paused:
            raise AgentError("задача уже на паузе")
        self.paused = True

    def resume(self) -> None:
        """Продолжить задачу с сохранённого места."""
        self._require_active()
        if not self.paused:
            raise AgentError("задача не на паузе")
        self.paused = False

    def reset(self) -> None:
        self.stage = PLANNING
        self.current_step = ""
        self.expected_action = ""
        self.paused = False
        self.active = False

    def prompt_message(self) -> Optional[Dict[str, str]]:
        """Служебное сообщение, которое сопровождает каждый следующий запрос."""
        if not self.active:
            return None
        lines = [
            "Этап: {}".format(self.stage),
            "Текущий шаг: {}".format(self.current_step or "не уточнён"),
            "Ожидаемое действие: {}".format(
                self.expected_action or "продолжить текущий шаг"),
            "Статус: {}".format("пауза" if self.paused else "в работе"),
        ]
        if self.paused:
            lines.append(
                "Задача приостановлена. Не выполняй следующий шаг, пока пауза не снята.")
        else:
            lines.append("Продолжай с этого места без повторного сбора уже известных данных.")
        return {"role": "user", "content": TASK_STATE_PREFIX + "\n".join(lines)}

    def absorb_model_update(self, payload: Dict[str, Any]) -> bool:
        """Принять решение модели, только если оно соблюдает автомат."""
        if not self.active or self.paused or not isinstance(payload, dict):
            return False
        wanted = _clean(payload.get("stage")).lower()
        if wanted not in TASK_STAGES:
            return False
        current_index = TASK_STAGES.index(self.stage)
        wanted_index = TASK_STAGES.index(wanted)
        if wanted_index < current_index:
            return False
        step = _clean(payload.get("current_step"))
        expected = _clean(payload.get("expected_action"))
        # Даже если один ответ закончил несколько фаз, внутри автомата они
        # проходятся последовательно — прямого ребра planning → done нет.
        while self.stage != wanted:
            self.transition(self.next_stage)
        self.update(step or self.current_step, expected or self.expected_action)
        if payload.get("paused") is True:
            self.pause()
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "stage": self.stage,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "paused": self.paused,
        }

    @classmethod
    def restore(cls, payload: Any) -> "TaskState":
        if not isinstance(payload, dict):
            return cls()
        stage = _clean(payload.get("stage")).lower() or PLANNING
        if stage not in TASK_STAGES:
            stage = PLANNING
        return cls(
            active=bool(payload.get("active")),
            stage=stage,
            current_step=_clean(payload.get("current_step")),
            expected_action=_clean(payload.get("expected_action")),
            paused=bool(payload.get("paused")),
        )

    def _require_active(self) -> None:
        if not self.active:
            raise AgentError("текущая задача не начата")


def split_task_directive(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Снять служебную директиву с конца ответа и разобрать её JSON."""
    match = TASK_STATE_PATTERN.search(text or "")
    if match is None:
        return (text or "").strip(), None
    visible = (text or "")[:match.start()].strip()
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError):
        return (text or "").strip(), None
    return visible, payload if isinstance(payload, dict) else None


def parse_task_update(text: str) -> Optional[Dict[str, Any]]:
    """Разобрать чистый JSON контролёра с допуском на markdown fence."""
    value = (text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def state_review_messages(state: TaskState, objective: str,
                          answer: str) -> list:
    """Запрос независимому контролёру, когда основной ответ не сменил этап."""
    clipped = (answer or "")[:STATE_REVIEW_MAX_CHARS]
    system = (
        "Ты контролёр конечного автомата задачи. Верни только один JSON-объект "
        "с полями stage, current_step, expected_action, paused. Определи этап, "
        "фактически достигнутый после ответа ассистента: planning — существует "
        "только план; execution — результат ещё создаётся; validation — "
        "запрошенный результат уже создан и теперь должен быть проверен; done — "
        "результат создан и проверен. Не оставляй execution ради необязательных "
        "улучшений. paused=true допустим только при обязательном вопросе к "
        "пользователю. Не добавляй markdown и пояснения.")
    user = (
        "Исходная задача:\n{}\n\nТекущее состояние:\n"
        "stage={}\ncurrent_step={}\nexpected_action={}\n\n"
        "Последний ответ ассистента:\n{}".format(
            _clean(objective) or state.current_step, state.stage,
            state.current_step, state.expected_action, clipped))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
