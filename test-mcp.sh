#!/usr/bin/env bash
# Обязательный cross-process E2E: Agent -> Node MCP -> fake GitHub API.
set -euo pipefail
cd "$(dirname "$0")"

MCP_PROJECT="${MCP_PROJECT_DIR:-../AI-Advent-MCP/mcp1}"
if [ ! -f "$MCP_PROJECT/package.json" ]; then
  echo "→ MCP-проект не найден: $MCP_PROJECT" >&2
  exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "→ Сначала запустите ./run.sh — он создаст окружение." >&2
  exit 1
fi

(cd "$MCP_PROJECT" && npm install --ignore-scripts)
export MCP_CLIENT_SCRIPT="$MCP_PROJECT/src/client.js"
export MCP_E2E_REQUIRED=1
exec .venv/bin/python -m pytest -q tests/test_mcp_e2e.py
