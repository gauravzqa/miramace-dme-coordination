#!/usr/bin/env bash
# Live end-to-end test of all flows. Run from anywhere:  bash ~/MiraMace/app/scripts/e2e_live.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "== syncing deps (incl. voice extra) =="
uv sync --extra voice || uv sync
echo "== running live e2e (progress also in e2e/<timestamp>/progress.log) =="
uv run --no-sync python scripts/e2e_live.py "$@"
