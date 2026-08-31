#!/usr/bin/env bash
# Готовит окружение и запускает чат. Повторные запуски используют готовый venv.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
  echo "→ Создаю виртуальное окружение…"
  "$PYTHON" -m venv "$VENV"
fi

if [ ! -f "$VENV/.deps-ok" ]; then
  echo "→ Ставлю зависимости…"
  "$VENV/bin/python" -m pip install --upgrade pip -q
  "$VENV/bin/pip" install -q -r requirements.txt
  # tiktoken делает оценку токенов точнее, но не обязателен.
  "$VENV/bin/pip" install -q tiktoken 2>/dev/null \
    || echo "  (tiktoken не установился — оценка неотправленных токенов будет приблизительной)"
  touch "$VENV/.deps-ok"
fi

exec "$VENV/bin/python" chat.py "$@"
