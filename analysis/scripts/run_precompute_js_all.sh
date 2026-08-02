#!/bin/bash
# Precompute per-token JS/KL parquets for all 12 (B, RL) × dataset pairs in
# parallel. Each pair is single-CPU; node has 64+ cores so we can run all 12
# at once.

set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave2_js_logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

PAIRS=(
    "B_q3 M_q3 aime"
    "B_q3 M_q3 ifeval"
    "B_q3 M_q3 ifbench"
    "B_q3 I_q3 aime"
    "B_q3 I_q3 ifeval"
    "B_q3 I_q3 ifbench"
    "B_q25m M_q25m aime"
    "B_q25m M_q25m ifeval"
    "B_q25m M_q25m ifbench"
    "B_q25m I_q25m aime"
    "B_q25m I_q25m ifeval"
    "B_q25m I_q25m ifbench"
)

echo "[precompute_js_all] launching ${#PAIRS[@]} parallel pair jobs"

worker_body='
PAIR="$1"
LOG_DIR="$2"
read BASE RL DS <<< "$PAIR"
NAME="${BASE}__${RL}__${DS}"
JOB_LOG=$LOG_DIR/${NAME}.log
python analysis/scripts/precompute_js.py \
    --base "$BASE" --rl "$RL" --dataset "$DS" \
    > "$JOB_LOG" 2>&1
EXIT=$?
echo "$EXIT" > "$LOG_DIR/${NAME}.exitcode"
'

for pair in "${PAIRS[@]}"; do
    setsid nohup bash -c "$worker_body" wave2_js_$(echo "$pair"|tr ' ' '_') \
        "$pair" "$LOG_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[precompute_js_all] launched. watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
