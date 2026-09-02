#!/usr/bin/env bash
# Запуск сравнения четырёх способов постановки задачи.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python reasoning.py "$@"
