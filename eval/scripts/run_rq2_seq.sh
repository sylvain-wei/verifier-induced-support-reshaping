#!/usr/bin/env bash
# scripts/run_rq2_seq.sh
#
# Run the RQ2 sequential models (math_then_if, if_then_math) through the
# exact same 5-benchmark matrix used in the 20260504_rq1 run. Reusing the
# same benchmarks, prompts, decoding, and metric pipeline makes the
# sequential results directly comparable with base / math_rlvr / if_rlvr.
#
# Produces a separate run_id (default 20260509_rq2_seq) so the sequential
# responses/metrics sit in their own directory and can be cleanly joined
# with the RQ1 summary at analysis time.
#
# Default layout: 2 models, 2 GPUs, one model per GPU, serial tasks within
# each model. Expected wall time on 2× H20 parallel: ~65-75 min.

set -euo pipefail

export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export RUN_ID="${RUN_ID:-20260509_rq2_seq}"
GPUS="${GPUS:-0 1}"
export PYTHONPATH="${EVAL_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$EVAL_ROOT"

MODELS=(math_then_if if_then_math)

read -ra GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -lt ${#MODELS[@]} ]]; then
  echo "ERROR: need ${#MODELS[@]} GPUs, got ${#GPU_ARR[@]} from GPUS='$GPUS'" >&2
  exit 1
fi

TASKS=(
  "math500|sampling_k16"
  "aime24|sampling_k32"
  "gsm8k|greedy"
  "ifeval|greedy"
  "ifbench|greedy"
)

LOG_DIR="$EVAL_ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  RQ2 SEQUENTIAL RUN"
echo "------------------------------------------------------------"
echo "  run_id:    $RUN_ID"
echo "  models:    ${MODELS[*]}"
echo "  gpus:      $GPUS"
echo "  started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo "==> env check"
python scripts/check_env.py 2>&1 | tee "$LOG_DIR/00_env_check.log"

echo "==> prepare data (idempotent)"
python scripts/prepare_data.py --only math500 aime24 gsm8k ifeval ifbench 2>&1 | tee "$LOG_DIR/01_prepare_data.log"

launch_model() {
  local model=$1 gpu=$2 i=$3
  local log="$LOG_DIR/02_${model}.log"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    sleep $((i * 5))
    exec 1>>"$log" 2>&1
    echo "############################################################"
    echo "# MODEL=$model  GPU=$gpu  start=$(date '+%H:%M:%S')"
    echo "############################################################"
    for spec in "${TASKS[@]}"; do
      local bench="${spec%%|*}"
      local mode="${spec##*|}"
      echo
      echo "---- [$(date '+%H:%M:%S')] $model / $bench / $mode ----"
      python scripts/run_inference.py \
        --run-id "$RUN_ID" --model-id "$model" \
        --benchmark "$bench" --mode "$mode"
      python scripts/compute_metrics.py \
        --run-id "$RUN_ID" --model-id "$model" \
        --benchmark "$bench" --mode "$mode"
    done
    echo
    echo "############################################################"
    echo "# MODEL=$model DONE  end=$(date '+%H:%M:%S')"
    echo "############################################################"
  ) &
}

START_TS=$(date +%s)
PIDS=()
for i in "${!MODELS[@]}"; do
  launch_model "${MODELS[$i]}" "${GPU_ARR[$i]}" "$i"
  pid=$!
  PIDS+=("$pid")
  echo "  launched ${MODELS[$i]} on GPU ${GPU_ARR[$i]} (PID $pid)"
done

trap 'echo; echo "[INT] killing child PIDs ${PIDS[*]}"; kill "${PIDS[@]}" 2>/dev/null || true; exit 130' INT TERM

FAIL=0
FAILED_MODELS=()
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  model="${MODELS[$idx]}"
  if wait "$pid"; then
    echo "  [$(date '+%H:%M:%S')] $model finished OK"
  else
    echo "  [$(date '+%H:%M:%S')] $model FAILED (see $LOG_DIR/02_${model}.log)"
    FAILED_MODELS+=("$model")
    FAIL=$((FAIL + 1))
  fi
done

trap - INT TERM

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "  parallel phase: $((ELAPSED/60))m $((ELAPSED%60))s  (failed: $FAIL/${#MODELS[@]})"

echo "==> aggregate"
python scripts/aggregate_metrics.py --run-id "$RUN_ID" 2>&1 | tee "$LOG_DIR/03_aggregate.log"
python scripts/make_plot_data.py --run-id "$RUN_ID" 2>&1 | tee "$LOG_DIR/04_make_plot.log"

echo
echo "============================================================"
echo "  RQ2 SEQ RUN  — summary"
echo "  run_id:     $RUN_ID"
echo "  wall time:  $((ELAPSED/60))m $((ELAPSED%60))s"
echo "  failed:     ${FAILED_MODELS[*]:-(none)}"
echo "  main summary: $EVAL_ROOT/metrics/$RUN_ID/aggregate/rq1_main_summary.csv"
echo "============================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"

if [[ $FAIL -gt 0 ]]; then exit 1; fi
