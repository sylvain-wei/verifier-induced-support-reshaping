#!/usr/bin/env bash
# RQ1 required matrix: math500(K=16) + aime24(K=32) + ifeval(greedy) + ifbench(greedy)
# × base / math_rlvr / if_rlvr. Fail fast on error.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_rq1_required.sh
#
# Env knobs:
#   CUDA_VISIBLE_DEVICES   - which GPU(s) to use (default: 0)
#   TENSOR_PARALLEL_SIZE   - vLLM TP size (default: 1)
#   RUN_ID                 - run id (default: 20260503_rq1)
#   EVAL_ROOT              - repo root (default: ${PROJECT_ROOT:-.}/eval)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export RUN_ID="${RUN_ID:-20260503_rq1}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export PYTHONPATH="${EVAL_ROOT}:${PYTHONPATH:-}"
# vLLM engine subprocess MUST use spawn, otherwise CUDA re-init in forked child crashes.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

cd "$EVAL_ROOT"

echo "==> [1/5] Env check"
python scripts/check_env.py

echo "==> [2/5] Prepare data (idempotent)"
python scripts/prepare_data.py --only math500 aime24 gsm8k ifeval ifbench

MODELS=(base math_rlvr if_rlvr)

declare -a TASKS=(
  "math500|sampling_k16"
  "aime24|sampling_k32"
  "gsm8k|greedy"
  "ifeval|greedy"
  "ifbench|greedy"
)

echo "==> [3/5] Inference"
for model in "${MODELS[@]}"; do
  for spec in "${TASKS[@]}"; do
    bench="${spec%%|*}"
    mode="${spec##*|}"
    echo "----- inference: model=$model benchmark=$bench mode=$mode"
    python scripts/run_inference.py \
      --run-id "$RUN_ID" --model-id "$model" \
      --benchmark "$bench" --mode "$mode"
  done
done

echo "==> [4/5] Per-run metrics"
for model in "${MODELS[@]}"; do
  for spec in "${TASKS[@]}"; do
    bench="${spec%%|*}"
    mode="${spec##*|}"
    echo "----- metrics: model=$model benchmark=$bench mode=$mode"
    python scripts/compute_metrics.py \
      --run-id "$RUN_ID" --model-id "$model" \
      --benchmark "$bench" --mode "$mode"
  done
done

echo "==> [5/5] Aggregate + plot data"
python scripts/aggregate_metrics.py --run-id "$RUN_ID"
python scripts/make_plot_data.py --run-id "$RUN_ID"

echo "Done. See metrics/${RUN_ID}/aggregate/rq1_main_summary.csv"
