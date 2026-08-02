#!/bin/bash
# Precompute per-token JS / KL for the 28 trajectory pairs (step ∈ {40,60,80,100}
# for vanilla/S1/S2; step 40 for negDAI/negRand). Mirrors
# run_precompute_predictive_js.sh's job-launch pattern.
#
# Output:  analysis/data/wave5_predictive_js/B_q3__{rl_tag}__{ds}.parquet
# Pre-req: analysis/data/wave5_predictive/{rl_tag}__{ds}.parquet must exist.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave5_traj_js_logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

WAVE1_DIR=analysis/data/wave1
WAVE5P_DIR=analysis/data/wave5_predictive
OUT_DIR=analysis/data/wave5_predictive_js

# (label, list of steps)
JOBS=()
for step in 40 60 80 100; do
    for label in vanilla S1soft S2hard; do
        for ds in aime ifeval; do
            JOBS+=("I_q3_step${step}_${label}|$ds")
        done
    done
done
# negctl only step 40
for label in negDAI negRand; do
    for ds in aime ifeval; do
        JOBS+=("I_q3_step40_${label}|$ds")
    done
done

echo "[precompute_traj_js] launching ${#JOBS[@]} parallel pair jobs"

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
    setsid nohup bash -c "$worker_body" wave5_tjs_${NAME} \
        "$spec" "$LOG_DIR" "$WAVE1_DIR" "$WAVE5P_DIR" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[precompute_traj_js] launched. watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l (target ${#JOBS[@]})"
