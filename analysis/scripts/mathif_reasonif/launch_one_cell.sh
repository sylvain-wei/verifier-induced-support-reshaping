#!/bin/bash
# Single-cell launcher. Run each cell completely detached so that the parent
# shell (codebuddy or login session) terminating does not propagate signals.
set -uo pipefail

ROOT=${PROJECT_ROOT:-.}
EVAL_DIR="$ROOT/eval"

MODEL=$1
BENCH=$2
GPU=$3
RUN_ID=$4
MODE=$5
PLAN_YAML=$6
MODELS_YAML=$7
CHUNK_SIZE=${8:-32}

LOG_DIR="$ROOT/analysis/scripts/mathif_reasonif/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${MODEL}_${BENCH}_${MODE}.log"
RESP="$EVAL_DIR/responses/$RUN_ID/$MODEL/$BENCH/$MODE.jsonl"
SCORED="$ROOT/analysis/tables/mathif_reasonif_scored_${RUN_ID}_${MODEL}_${BENCH}_${MODE}.jsonl"

# Distinct distributed init port per cell so vLLM v1 engine cores never collide.
PORT=$((33000 + GPU * 100 + (RANDOM % 90)))

{
  echo "[start] $(date '+%F %T') pid=$$ ppid=$PPID model=$MODEL bench=$BENCH gpu=$GPU run_id=$RUN_ID port=$PORT chunk=$CHUNK_SIZE"
  export VLLM_HOST_IP=127.0.0.1
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$PORT
  export CUDA_VISIBLE_DEVICES=$GPU
  export TENSOR_PARALLEL_SIZE=1
  export TOKENIZERS_PARALLELISM=true
  export VLLM_LOGGING_LEVEL=WARN

  python "$EVAL_DIR/scripts/run_inference.py" \
    --model-id "$MODEL" \
    --benchmark "$BENCH" \
    --mode "$MODE" \
    --run-id "$RUN_ID" \
    --models-yaml "$MODELS_YAML" \
    --plan-yaml "$PLAN_YAML" \
    --engine vllm \
    --chunk-size "$CHUNK_SIZE"
  RC=$?
  echo "[inference_rc=$RC] $(date '+%F %T')"
  if [[ $RC -eq 0 && -f "$RESP" ]]; then
    if [[ "$BENCH" == "mathif" ]]; then
      python "$ROOT/analysis/scripts/mathif_reasonif/score_mathif.py" --input "$RESP" --output "$SCORED"
    else
      python "$ROOT/analysis/scripts/mathif_reasonif/score_reasonif.py" --input "$RESP" --output "$SCORED"
    fi
    echo "[done] $(date '+%F %T') model=$MODEL bench=$BENCH"
  fi
} >> "$LOG" 2>&1
