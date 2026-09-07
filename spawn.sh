#!/usr/bin/env bash
# Массовый запуск агентов. Всё готовит run.sh; здесь только точка входа.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "→ Сначала запустите ./run.sh — он создаст окружение."; exit 1; }
exec .venv/bin/python spawn_demo.py "$@"
