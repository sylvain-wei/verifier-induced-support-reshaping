#!/bin/bash
# Launch all MathIF/ReasonIF cells fully detached via setsid+nohup so the
# parent shell exiting does not signal the workers.
set -uo pipefail

ROOT=${PROJECT_ROOT:-.}
RUN_ID=${RUN_ID:-mathif_reasonif_k16_8k}
MODE=${MODE:-sampling_k16}
MODELS=${MODELS:-"base math_rlvr if_rlvr math_then_if"}
BENCHMARKS=${BENCHMARKS:-"mathif reasonif"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PLAN_YAML=${PLAN_YAML:-$ROOT/eval/configs/eval_plan_mathif_reasonif_k16_8k.yaml}
MODELS_YAML=${MODELS_YAML:-$ROOT/eval/configs/models_mathif_reasonif.yaml}
CHUNK_SIZE=${CHUNK_SIZE:-32}

LOG_DIR="$ROOT/analysis/scripts/mathif_reasonif/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
gpu_arr=($GPUS)
ngpu=${#gpu_arr[@]}

idx=0
for model in $MODELS; do
  for bench in $BENCHMARKS; do
    gpu=${gpu_arr[$((idx % ngpu))]}
    echo "[launch] gpu=$gpu model=$model bench=$bench"
    setsid nohup bash "$ROOT/analysis/scripts/mathif_reasonif/launch_one_cell.sh" \
      "$model" "$bench" "$gpu" "$RUN_ID" "$MODE" "$PLAN_YAML" "$MODELS_YAML" "$CHUNK_SIZE" \
      < /dev/null > /dev/null 2>&1 &
    disown || true
    idx=$((idx + 1))
    sleep 1
  done
done
echo "[launched] $idx cells; logs in $LOG_DIR"
