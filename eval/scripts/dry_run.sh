#!/usr/bin/env bash
# Small-N smoke test: 2 samples per (model, benchmark) for required matrix.
# Produces responses & metrics under run_id 20260503_rq1_smoke.
#
# Uses --limit 2 on run_inference.py so each benchmark only processes 2 examples.
# Useful to verify the whole pipeline bolts together BEFORE launching the big run.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export RUN_ID="${RUN_ID:-20260503_rq1_smoke}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export PYTHONPATH="${EVAL_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

cd "$EVAL_ROOT"

echo "==> env check"
python scripts/check_env.py

echo "==> prepare data (idempotent)"
python scripts/prepare_data.py --only math500 aime24 gsm8k ifeval ifbench

MODELS=(base math_rlvr if_rlvr)

declare -a TASKS=(
  "math500|sampling_k16"
  "aime24|sampling_k32"
  "gsm8k|greedy"
  "ifeval|greedy"
  "ifbench|greedy"
)

for model in "${MODELS[@]}"; do
  for spec in "${TASKS[@]}"; do
    bench="${spec%%|*}"
    mode="${spec##*|}"
    echo "----- SMOKE inference: model=$model benchmark=$bench mode=$mode"
    python scripts/run_inference.py \
      --run-id "$RUN_ID" --model-id "$model" \
      --benchmark "$bench" --mode "$mode" \
      --limit 2
    python scripts/compute_metrics.py \
      --run-id "$RUN_ID" --model-id "$model" \
      --benchmark "$bench" --mode "$mode"
  done
done

python scripts/aggregate_metrics.py --run-id "$RUN_ID"
python scripts/make_plot_data.py --run-id "$RUN_ID"

echo "Smoke done. Check metrics/${RUN_ID}/aggregate/"
