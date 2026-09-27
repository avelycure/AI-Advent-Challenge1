"""Детерминированная демонстрация полного цикла Agent -> MCP -> GitHub."""
from __future__ import annotations

import json

from llmagent import Agent, AgentConfig, HistoryConfig, MCPConfig, ToolPolicy, Transport
from llmagent.transport import Completion, ToolCall, count_message_tokens, count_text_tokens
from llmchat.subagent import toolbox_for

TOOL_NAME = "github_repository_stats"


class StatisticsUsingClient:
    """Заменяет LLM в проверке, но проходит настоящий tool-loop агента."""

    def validate_key(self) -> None:
        return None

    def complete(self, model_ref, messages, max_tokens, temperature=0.7,
                 top_p=None, stop=None, response_format=None, tools=None):
        has_tool_result = any(
            message.get("role") == "tool" for message in messages)
        if tools and not has_tool_result:
            arguments = json.dumps({
                "visibility": "all",
                "includeForks": True,
                "includeArchived": True,
                "top": 3,
            })
            return Completion(
                "", count_message_tokens(messages), 5, elapsed=0.001,
                tool_calls=[ToolCall("github-stats-1", TOOL_NAME, arguments)],
            )

        tool_message = next(
            message for message in reversed(messages) if message.get("role") == "tool")
        try:
            stats = json.loads(tool_message["content"])
        except (TypeError, ValueError):
            answer = "GitHub MCP не вернул статистику: {}".format(
                tool_message.get("content") or "неизвестная ошибка")
            return Completion(
                answer,
                count_message_tokens(messages),
                count_text_tokens(answer),
                elapsed=0.001,
            )
        repositories = stats["repositories"]
        totals = stats["totals"]
        answer = (
            "GitHub MCP вернул {total} репозиториев: {public} публичных и "
            "{private} приватных. Всего звёзд: {stars}."
        ).format(
            total=repositories["total"],
            public=repositories["public"],
            private=repositories["private"],
            stars=totals["stars"],
        )
        return Completion(
            answer,
            count_message_tokens(messages),
            count_text_tokens(answer),
            elapsed=0.001,
        )


def main() -> None:
    config = AgentConfig(
        name="github-mcp-livecheck",
        transport=Transport(demo=True, demo_delay=0.0),
        tools=ToolPolicy(enabled=(TOOL_NAME,), max_calls=2),
        mcp=MCPConfig(enabled=True),
        history=HistoryConfig(enabled=False),
    )
    agent = Agent(
        config,
        client=StatisticsUsingClient(),
        toolbox=toolbox_for(config),
    )
    result = agent.ask("Собери статистику по всем моим репозиториям GitHub.")
    print(json.dumps({
        "answer": result.text,
        "tool_calls": [
            {"name": call["name"], "ok": call["ok"]}
            for call in result.tool_calls
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
