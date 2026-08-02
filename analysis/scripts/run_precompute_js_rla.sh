#!/bin/bash
# Parallel CPU-only precompute of RL-anchored JS for Track A.3.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave2_js_rla_logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

PAIRS=(
    "B_q3 M_q3 aime"
    "B_q3 I_q3 aime"
    "B_q3 I_q3 ifeval"
    "B_q25m M_q25m aime"
    "B_q25m I_q25m aime"
    "B_q25m I_q25m ifeval"
)

worker='
JOB=$1
LOG_DIR=$2
read BASE RL DS <<< "$JOB"
NAME="${BASE}__${RL}__${DS}"
python analysis/scripts/precompute_js_rl_anchored.py \
    --base "$BASE" --rl "$RL" --dataset "$DS" \
    > "$LOG_DIR/${NAME}.log" 2>&1
echo "$?" > "$LOG_DIR/${NAME}.exitcode"
'

for p in "${PAIRS[@]}"; do
    setsid nohup bash -c "$worker" rla_$(echo "$p"|tr ' ' '_') \
        "$p" "$LOG_DIR" </dev/null >/dev/null 2>&1 & disown
done

echo "[rla_all] launched ${#PAIRS[@]} parallel CPU jobs"
echo "[rla_all] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
