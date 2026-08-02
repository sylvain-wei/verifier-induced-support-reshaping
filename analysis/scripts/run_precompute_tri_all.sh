#!/bin/bash
# Parallel precompute of the 3-way JS triangles for d5.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave2_tri_logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

JOBS=(
    "B_q3 M_q3 I_q3 aime"
    "B_q3 M_q3 I_q3 ifeval"
    "B_q3 M_q3 I_q3 ifbench"
    "B_q25m M_q25m I_q25m aime"
    "B_q25m M_q25m I_q25m ifeval"
    "B_q25m M_q25m I_q25m ifbench"
)

worker='
JOB=$1
LOG_DIR=$2
read B M I DS <<< "$JOB"
NAME=${B}__${M}__${I}__${DS}
python analysis/scripts/precompute_js_triangle.py \
    --base "$B" --m "$M" --i "$I" --dataset "$DS" \
    > "$LOG_DIR/${NAME}.log" 2>&1
echo "$?" > "$LOG_DIR/${NAME}.exitcode"
'

for j in "${JOBS[@]}"; do
    setsid nohup bash -c "$worker" tri_$(echo "$j"|tr ' ' '_') \
        "$j" "$LOG_DIR" </dev/null >/dev/null 2>&1 & disown
done

echo "[tri] launched ${#JOBS[@]} parallel jobs"
echo "[tri] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
