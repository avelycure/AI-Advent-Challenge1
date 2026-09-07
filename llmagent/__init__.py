"""Агент как отдельная сущность: запрос внутрь, разобранный ответ наружу.

Пакет намеренно не знает ничего об интерфейсе. Он не печатает, не спрашивает
и не импортирует ни ``rich``, ни ``llmchat`` — за этим следит отдельный тест.
Поэтому одного и того же агента одинаково вызывают терминальный чат, опыты
со сравнением промптов и массовый прогон на сотню конфигов.

    from llmagent import Agent, DEFAULT_CONFIG

    agent = Agent(DEFAULT_CONFIG)
    print(agent.ask("Привет!").text)
"""
from .agent import Agent
from .config import (
    DEFAULT_CONFIG,
    SYSTEM_PROMPT,
    AgentConfig,
    Budget,
    HistoryConfig,
    InputPolicy,
    JudgeConfig,
    OutputPolicy,
    Transport,
)
from .errors import (
    AgentError,
    BudgetExceeded,
    ConfigError,
    InputRejected,
    LLMError,
    MissingCredentials,
    OutputRejected,
)
from .formats import FORMATS, FREE_FORMAT, STOP_MARKER, Check, Field, Schema, Validation
from .history import Conversation, Message
from .params import GenerationParams
from .registry import SHARED, ClientRegistry, Credentials, resolve_credentials
from .result import AgentResult
from .spawn import build_agents, matrix, spawn, spawn_sync, summarize
from .usage import UsageMeter

__version__ = "2.0.0"

__all__ = [
    "Agent", "AgentConfig", "AgentResult", "DEFAULT_CONFIG", "SYSTEM_PROMPT",
    "InputPolicy", "OutputPolicy", "JudgeConfig", "HistoryConfig", "Budget", "Transport",
    "GenerationParams", "Schema", "Field", "Check", "Validation",
    "FORMATS", "FREE_FORMAT", "STOP_MARKER",
    "Conversation", "Message", "UsageMeter",
    "ClientRegistry", "Credentials", "SHARED", "resolve_credentials",
    "spawn", "spawn_sync", "build_agents", "matrix", "summarize",
    "AgentError", "ConfigError", "MissingCredentials", "InputRejected",
    "OutputRejected", "BudgetExceeded", "LLMError",
]
