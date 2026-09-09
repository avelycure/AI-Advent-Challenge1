#!/usr/bin/env bash
# Три диалога рядом: короткий, длинный и переполненный. Без --live сеть не трогается.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "→ Сначала запустите ./run.sh — он создаст окружение."; exit 1; }
exec .venv/bin/python tokens.py "$@"
