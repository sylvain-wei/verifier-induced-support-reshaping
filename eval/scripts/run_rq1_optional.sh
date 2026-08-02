#!/usr/bin/env bash
# RQ1 optional matrix: ifeval(K=8), ifbench(K=8), gsm8k(greedy), gsm8k_subset(K=8)
# × base / math_rlvr / if_rlvr.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export RUN_ID="${RUN_ID:-20260503_rq1_opt}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export PYTHONPATH="${EVAL_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

cd "$EVAL_ROOT"

echo "==> [1/5] Env check"
python scripts/check_env.py

echo "==> [2/5] Prepare data (idempotent; adds gsm8k_subset if not present)"
python scripts/prepare_data.py --only ifeval ifbench gsm8k

MODELS=(base math_rlvr if_rlvr)

declare -a TASKS=(
  "ifeval|sampling_k8"
  "ifbench|sampling_k8"
  "gsm8k_subset|sampling_k8"
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
    python scripts/compute_metrics.py \
      --run-id "$RUN_ID" --model-id "$model" \
      --benchmark "$bench" --mode "$mode"
  done
done

echo "==> [5/5] Aggregate + plot data"
python scripts/aggregate_metrics.py --run-id "$RUN_ID"
python scripts/make_plot_data.py --run-id "$RUN_ID"

echo "Done. See metrics/${RUN_ID}/aggregate/"
