#!/usr/bin/env bash
# Прогон тестов. Сеть не нужна, денег не тратится: всё на заглушках.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
[ -d "$VENV" ] || { echo "→ Сначала запустите ./run.sh — он создаст окружение."; exit 1; }
if [ ! -f "$VENV/.dev-deps-ok" ]; then
  echo "→ Ставлю зависимости тестов…"
  "$VENV/bin/pip" install -q -r requirements-dev.txt
  touch "$VENV/.dev-deps-ok"
fi

exec "$VENV/bin/python" -m pytest "$@"
