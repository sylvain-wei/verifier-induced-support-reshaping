#!/bin/bash
# Wrapper for running MathIF/ReasonIF through eval/scripts/run_inference.py.
# Before using this, add mathif/reasonif entries to eval/configs/datasets.yaml,
# eval/configs/eval_plan.yaml, and the desired model/checkpoint entries to
# eval/configs/models.yaml (or pass alternate yaml files to run_inference.py).
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
EVAL_DIR="$ROOT/eval"
RUN_ID=${RUN_ID:-mathif_reasonif_smoke}
MODE=${MODE:-sampling_k16}
ENGINE=${ENGINE:-vllm}
MODELS=${MODELS:-"base math_rlvr if_rlvr math_then_if if_then_math"}
BENCHMARKS=${BENCHMARKS:-"mathif reasonif"}

for model in $MODELS; do
  for bench in $BENCHMARKS; do
    echo "[inference] model=$model benchmark=$bench mode=$MODE run_id=$RUN_ID"
    if [[ -n "${LIMIT:-}" ]]; then
      python "$EVAL_DIR/scripts/run_inference.py" \
        --model-id "$model" \
        --benchmark "$bench" \
        --mode "$MODE" \
        --run-id "$RUN_ID" \
        --engine "$ENGINE" \
        --limit "$LIMIT"
    else
      python "$EVAL_DIR/scripts/run_inference.py" \
        --model-id "$model" \
        --benchmark "$bench" \
        --mode "$MODE" \
        --run-id "$RUN_ID" \
        --engine "$ENGINE"
    fi

    resp="$EVAL_DIR/responses/$RUN_ID/$model/$bench/$MODE.jsonl"
    scored="$ROOT/analysis/tables/mathif_reasonif_scored_${RUN_ID}_${model}_${bench}_${MODE}.jsonl"
    echo "[score] $resp -> $scored"
    if [[ "$bench" == "mathif" ]]; then
      python "$ROOT/analysis/scripts/mathif_reasonif/score_mathif.py" --input "$resp" --output "$scored"
    else
      python "$ROOT/analysis/scripts/mathif_reasonif/score_reasonif.py" --input "$resp" --output "$scored"
    fi
  done
done
