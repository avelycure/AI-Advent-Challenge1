"""Ошибки агента, отделённые от ошибок транспорта.

Транспортную ``LLMError`` агент не подменяет: она уже переведена в понятный
текст и вызывающему полезна как есть. Здесь только то, чего на уровне HTTP
не существует, — неверный конфиг, отсутствие реквизитов, нарушение политики
и упёршийся бюджет.
"""
from __future__ import annotations

from .transport import LLMError

__all__ = ["AgentError", "ConfigError", "MissingCredentials", "InputRejected",
           "OutputRejected", "BudgetExceeded", "ContextOverflow", "LLMError"]


class AgentError(Exception):
    """Общий предок ошибок агента."""


class ConfigError(AgentError):
    """Конфиг нельзя применить: неизвестный провайдер, модель или формат."""


class MissingCredentials(AgentError):
    """Реквизитов нет ни в конфиге, ни в окружении, ни в файлах."""


class InputRejected(AgentError):
    """Запрос не прошёл входную политику и в модель не отправлялся."""


class OutputRejected(AgentError):
    """Ответ не прошёл выходную политику, и переспросы не помогли."""


class BudgetExceeded(AgentError):
    """Лимит токенов или денег исчерпан; запрос не отправлен."""


class ContextOverflow(AgentError):
    """Диалог не помещается в окно модели; запрос не отправлен.

    Отдельно от ``BudgetExceeded``: там кончились деньги и помогает новый
    лимит, здесь кончилось место и помогает только более короткий разговор.
    """
