#!/bin/bash
# Precompute per-token JS / KL for the 10 predictive routing-JS pairs:
#   B_q3 × {I_q3_step20_*5runs} × {aime, ifeval}
# CPU only; 10 parallel processes (each ~one core; numpy-bound).
#
# Output:
#   analysis/data/wave5_predictive_js/B_q3__{rl_tag}__{ds}.parquet
#
# Pre-req: analysis/data/wave5_predictive/{rl_tag}__{ds}.parquet must already
# exist (produced by run_predictive_routing_js.sh).
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave5_predictive_js_logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

WAVE1_DIR=analysis/data/wave1
WAVE5P_DIR=analysis/data/wave5_predictive
OUT_DIR=analysis/data/wave5_predictive_js

RUN_TAGS=(
    I_q3_step20_vanilla
    I_q3_step20_S1soft
    I_q3_step20_S2hard
    I_q3_step20_negDAI
    I_q3_step20_negRand
)
DATASETS=(aime ifeval)

JOBS=()
for TAG in "${RUN_TAGS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        JOBS+=("$TAG|$DS")
    done
done

echo "[precompute_pred_js] launching ${#JOBS[@]} parallel pair jobs"

worker_body='
SPEC="$1"
LOG_DIR="$2"
WAVE1="$3"
WAVE5P="$4"
OUT="$5"
IFS="|" read TAG DS <<< "$SPEC"
NAME="B_q3__${TAG}__${DS}"
JOB_LOG=$LOG_DIR/${NAME}.log
python analysis/scripts/precompute_js_predictive.py \
    --base_tag B_q3 \
    --base_parquet ${WAVE1}/B_q3__${DS}.parquet \
    --rl_tag ${TAG} \
    --rl_parquet ${WAVE5P}/${TAG}__${DS}.parquet \
    --dataset ${DS} \
    --out_dir ${OUT} \
    > "$JOB_LOG" 2>&1
EXIT=$?
echo "$EXIT" > "$LOG_DIR/${NAME}.exitcode"
'

for spec in "${JOBS[@]}"; do
    NAME=$(echo "$spec" | tr '|' '_')
    setsid nohup bash -c "$worker_body" wave5_pjs_${NAME} \
        "$spec" "$LOG_DIR" "$WAVE1_DIR" "$WAVE5P_DIR" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[precompute_pred_js] launched. watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
