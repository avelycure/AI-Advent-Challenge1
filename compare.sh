#!/usr/bin/env bash
# Запуск сравнения свободного и управляемого ответа.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || ./run.sh --help >/dev/null 2>&1 || true
exec .venv/bin/python compare.py "$@"
