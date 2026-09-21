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
    '"expected_action":"следующее действие","paused":false,'
    '"validation_passed":false,"validation_evidence":""}]]\n'
    "stage должен показывать фактически достигнутый после ответа этап цепочки "
    "planning → execution → validation → done. За один ответ разрешён переход "
    "только в соседний этап. В planning подготовь план и попроси пользователя "
    "явно утвердить его; не реализуй план, пока в состоянии нет отметки "
    "«План утверждён: да». В validation действительно проверь результат. "
    "Переходя из validation в done, поставь validation_passed=true и кратко "
    "запиши выполненные проверки в validation_evidence. Без успешной проверки "
    "останься в validation. "
    "current_step описывает актуальный шаг после твоего ответа, а "
    "expected_action — конкретное действие следующим. paused=true ставь, "
    "только если без ответа или решения пользователя продолжать нельзя. "
    "Как только требуемое решение создано, переходи как минимум в validation; "
    "не оставайся в execution ради необязательных улучшений. После собственной "
    "проверки результата переходи в done."
)

STATE_REVIEW_MAX_CHARS = 8000

_PLAN_APPROVAL = re.compile(
    r"(?:\b(?:утверждаю|одобряю|согласовываю|принимаю)\s+"
    r"(?:(?:этот|предложенный)\s+)?план\b|"
    r"\bсоглас(?:ен|на)\s+с\s+(?:(?:этим|предложенным)\s+)?планом\b|"
    r"\bплан\s+(?:подходит|принимается)\b|"
    r"\bплан\s+(?:утвержд[её]н|одобрен|согласован)\b|"
    r"\b(?:approve|approved|accept)\s+(?:the\s+)?plan\b)",
    re.IGNORECASE,
)
_PLAN_REJECTION = re.compile(
    r"(?:\bне\s+(?:утверждаю|одобряю|согласовываю|принимаю)\b|"
    r"\bне\s+соглас(?:ен|на)\s+с\s+.*\bпланом\b|"
    r"\bплан\s+не\s+подходит\b|"
    r"\bплан\s+не\s+(?:утвержд[её]н|одобрен|согласован)\b|"
    r"\b(?:do\s+not|don't)\s+(?:approve|accept)\b)",
    re.IGNORECASE,
)


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
    plan_approved: bool = False
    validation_passed: bool = False
    validation_evidence: str = ""

    def __post_init__(self) -> None:
        if self.stage not in TASK_STAGES:
            raise AgentError("неизвестный этап задачи «{}»; доступны: {}".format(
                self.stage, ", ".join(TASK_STAGES)))
        self.current_step = _clean(self.current_step)
        self.expected_action = _clean(self.expected_action)
        self.validation_evidence = _clean(self.validation_evidence)
        self.paused = bool(self.paused) and bool(self.active)
        self.plan_approved = bool(self.plan_approved) and bool(self.active)
        self.validation_passed = bool(self.validation_passed) and bool(
            self.active and self.validation_evidence)

    @property
    def status(self) -> str:
        if not self.active:
            return "inactive"
        return "paused" if self.paused else self.stage

    @property
    def next_stage(self) -> str:
        return NEXT_STAGE.get(self.stage, "")

    @property
    def execution_allowed(self) -> bool:
        """Можно ли основному ответу выполнять действия и вызывать инструменты."""
        return not self.paused and (self.stage != PLANNING or self.plan_approved)

    def start(self, current_step: str = "", expected_action: str = "") -> None:
        """Начать новую задачу с этапа планирования."""
        self.stage = PLANNING
        self.current_step = _clean(current_step)
        self.expected_action = _clean(expected_action)
        self.paused = False
        self.active = True
        self.plan_approved = False
        self.validation_passed = False
        self.validation_evidence = ""

    def start_from_request(self, request: str) -> None:
        """Автоматически принять первую реплику как новую задачу."""
        task = _clean(request)
        if len(task) > 240:
            task = task[:237].rstrip() + "…"
        self.start(task or "Определить задачу",
                   "Составить план и получить его утверждение")

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
        if wanted == EXECUTION and self.stage == PLANNING and not self.plan_approved:
            raise AgentError(
                "нельзя перейти из planning в execution: план ещё не утверждён; "
                "сначала утвердите план")
        if wanted == DONE and self.stage == VALIDATION and not self.validation_passed:
            raise AgentError(
                "нельзя перейти из validation в done: успешная проверка результата "
                "ещё не зафиксирована")
        changed = self.stage != wanted
        self.stage = wanted
        if changed and wanted == VALIDATION:
            self.validation_passed = False
            self.validation_evidence = ""
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

    def approve_plan(self) -> None:
        """Открыть переход в execution явным утверждением пользователя."""
        self._require_active()
        if self.paused:
            raise AgentError("задача на паузе; сначала продолжите её")
        if self.stage != PLANNING:
            raise AgentError("план можно утвердить только на этапе planning")
        self.plan_approved = True

    def observe_user_message(self, text: str) -> bool:
        """Распознать только явное, неотрицательное утверждение плана."""
        if (not self.active or self.paused or self.stage != PLANNING
                or self.plan_approved):
            return False
        value = _clean(text)
        if _PLAN_REJECTION.search(value) or not _PLAN_APPROVAL.search(value):
            return False
        self.approve_plan()
        return True

    def pass_validation(self, evidence: str) -> None:
        """Открыть переход в done после успешной проверки с доказательством."""
        self._require_active()
        if self.paused:
            raise AgentError("задача на паузе; сначала продолжите её")
        if self.stage != VALIDATION:
            raise AgentError("результат можно подтвердить только на этапе validation")
        checked = _clean(evidence)
        if not checked:
            raise AgentError("для успешной проверки нужно указать её результат")
        self.validation_passed = True
        self.validation_evidence = checked

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
        self.plan_approved = False
        self.validation_passed = False
        self.validation_evidence = ""

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
            "План утверждён: {}".format("да" if self.plan_approved else "нет"),
            "Валидация пройдена: {}".format(
                "да — " + self.validation_evidence if self.validation_passed else "нет"),
        ]
        if self.paused:
            lines.append(
                "Задача приостановлена. Не выполняй следующий шаг, пока пауза не снята.")
        else:
            lines.append("Продолжай с этого места без повторного сбора уже известных данных.")
        return {"role": "user", "content": TASK_STATE_PREFIX + "\n".join(lines)}

    def absorb_model_update(self, payload: Dict[str, Any]) -> bool:
        """Принять решение модели, только если оно соблюдает автомат."""
        error = self.model_update_error(payload)
        if error:
            return False
        wanted = _clean(payload.get("stage")).lower()
        step = _clean(payload.get("current_step"))
        expected = _clean(payload.get("expected_action"))
        if payload.get("validation_passed") is True:
            self.pass_validation(_clean(payload.get("validation_evidence")))
        if self.stage != wanted:
            self.transition(wanted)
        self.update(step or self.current_step, expected or self.expected_action)
        if payload.get("paused") is True:
            self.pause()
        return True

    def model_update_error(self, payload: Any) -> str:
        """Объяснить, почему директива модели нарушает жизненный цикл."""
        if not self.active:
            return "текущая задача не начата"
        if self.paused:
            return "задача на паузе; сначала продолжите её"
        if not isinstance(payload, dict):
            return "директива состояния должна быть JSON-объектом"
        wanted = _clean(payload.get("stage")).lower()
        if wanted not in TASK_STAGES:
            return "неизвестный этап задачи «{}»; доступны: {}".format(
                wanted, ", ".join(TASK_STAGES))
        allowed = (self.stage, self.next_stage) if self.next_stage else (self.stage,)
        if wanted not in allowed:
            if not self.next_stage:
                return "этап done завершающий; начните новую задачу"
            return (
                "из {} можно перейти только в {}; пропускать этапы и идти назад "
                "нельзя".format(self.stage, self.next_stage))
        if self.stage == PLANNING and wanted == EXECUTION and not self.plan_approved:
            return (
                "нельзя перейти из planning в execution: план ещё не утверждён; "
                "сначала утвердите план")
        marks_validation = payload.get("validation_passed") is True
        evidence = _clean(payload.get("validation_evidence"))
        if marks_validation and self.stage != VALIDATION:
            return "успешную проверку можно фиксировать только на этапе validation"
        if marks_validation and not evidence:
            return "для успешной проверки нужно указать validation_evidence"
        if (self.stage == VALIDATION and wanted == DONE
                and not (self.validation_passed or (marks_validation and evidence))):
            return (
                "нельзя перейти из validation в done: успешная проверка результата "
                "ещё не зафиксирована")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "stage": self.stage,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "paused": self.paused,
            "plan_approved": self.plan_approved,
            "validation_passed": self.validation_passed,
            "validation_evidence": self.validation_evidence,
        }

    @classmethod
    def restore(cls, payload: Any) -> "TaskState":
        if not isinstance(payload, dict):
            return cls()
        stage = _clean(payload.get("stage")).lower() or PLANNING
        if stage not in TASK_STAGES:
            stage = PLANNING
        # Старые сессии не хранили ворота. Уже достигнутый этап служит
        # свидетельством ранее пройденного перехода и остаётся совместимым.
        stage_index = TASK_STAGES.index(stage)
        plan_approved = payload.get("plan_approved")
        if plan_approved is None:
            plan_approved = stage_index >= TASK_STAGES.index(EXECUTION)
        validation_evidence = _clean(payload.get("validation_evidence"))
        validation_passed = payload.get("validation_passed")
        if validation_passed is None and stage == DONE:
            validation_passed = True
            validation_evidence = validation_evidence or "восстановлено из завершённой сессии"
        return cls(
            active=bool(payload.get("active")),
            stage=stage,
            current_step=_clean(payload.get("current_step")),
            expected_action=_clean(payload.get("expected_action")),
            paused=bool(payload.get("paused")),
            plan_approved=bool(plan_approved),
            validation_passed=bool(validation_passed),
            validation_evidence=validation_evidence,
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


def transition_refusal(reason: str, state: TaskState) -> str:
    """Понятный пользователю ответ вместо результата запрещённого этапа."""
    return (
        "Переход состояния отклонён: {}. Текущий этап — {}. {}"
    ).format(
        reason,
        state.stage,
        "Утвердите предложенный план явной фразой, например «Утверждаю план»."
        if state.stage == PLANNING and not state.plan_approved
        else "Сначала выполните ожидаемое действие: {}.".format(
            state.expected_action or "завершить текущий этап"),
    )


def state_review_messages(state: TaskState, objective: str,
                          answer: str) -> list:
    """Запрос независимому контролёру, когда основной ответ не сменил этап."""
    clipped = (answer or "")[:STATE_REVIEW_MAX_CHARS]
    system = (
        "Ты контролёр конечного автомата задачи. Верни только один JSON-объект "
        "с полями stage, current_step, expected_action, paused, "
        "validation_passed и validation_evidence. Определи этап, "
        "фактически достигнутый после ответа ассистента: planning — существует "
        "только план; execution — результат ещё создаётся; validation — "
        "запрошенный результат уже создан и теперь должен быть проверен; done — "
        "результат создан и проверен. Разрешён только текущий или следующий "
        "этап. Из planning нельзя переходить дальше без уже утверждённого плана. "
        "Для done обязательны validation_passed=true и непустое описание реально "
        "выполненных проверок в validation_evidence. Не оставляй execution ради "
        "необязательных улучшений. paused=true допустим только при обязательном "
        "вопросе к пользователю. Не добавляй markdown и пояснения.")
    user = (
        "Исходная задача:\n{}\n\nТекущее состояние:\n"
        "stage={}\ncurrent_step={}\nexpected_action={}\nplan_approved={}\n"
        "validation_passed={}\nvalidation_evidence={}\n\n"
        "Последний ответ ассистента:\n{}".format(
            _clean(objective) or state.current_step, state.stage,
            state.current_step, state.expected_action, state.plan_approved,
            state.validation_passed, state.validation_evidence, clipped))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
