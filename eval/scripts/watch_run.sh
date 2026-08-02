#!/usr/bin/env bash
# Tail all per-model logs of a given run_id.
#   bash scripts/watch_run.sh           # uses 20260504_rq1
#   bash scripts/watch_run.sh 20260504_rq1_opt
set -euo pipefail
RUN_ID="${1:-${RUN_ID:-20260504_rq1}}"
LOG_DIR="${PROJECT_ROOT:-.}/eval/logs/$RUN_ID"
if [[ ! -d "$LOG_DIR" ]]; then
  echo "No such log dir: $LOG_DIR" >&2
  exit 1
fi
echo "Tailing $LOG_DIR/02_*.log  (Ctrl-C to stop)"
exec tail -F "$LOG_DIR"/02_*.log
