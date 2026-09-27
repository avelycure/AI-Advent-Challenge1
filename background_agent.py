#!/usr/bin/env python3
"""Day 18: запустить GitHub-агента поверх MCP scheduler в фоне."""
import sys

from llmchat.background import main


if __name__ == "__main__":
    sys.exit(main())
