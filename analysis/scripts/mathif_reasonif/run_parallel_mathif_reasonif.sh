#!/bin/bash
# Run MathIF/ReasonIF cells in parallel across 8 H20 GPUs.
# Each job uses one visible GPU and tensor_parallel_size=1.
set -uo pipefail

ROOT=${PROJECT_ROOT:-.}
EVAL_DIR="$ROOT/eval"
RUN_ID=${RUN_ID:-mathif_reasonif_k16_fast}
MODE=${MODE:-sampling_k16}
PLAN_YAML=${PLAN_YAML:-$EVAL_DIR/configs/eval_plan_mathif_reasonif_k16_fast.yaml}
MODELS_YAML=${MODELS_YAML:-$EVAL_DIR/configs/models.yaml}
MODELS=${MODELS:-"base math_rlvr if_rlvr math_then_if if_then_math"}
BENCHMARKS=${BENCHMARKS:-"mathif reasonif"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
MAX_PARALLEL=${MAX_PARALLEL:-8}
ENGINE=${ENGINE:-vllm}
CHUNK_SIZE=${CHUNK_SIZE:-4}
LOG_DIR="$ROOT/analysis/scripts/mathif_reasonif/logs/$RUN_ID"
mkdir -p "$LOG_DIR" "$ROOT/analysis/tables"

job_items=()
for model in $MODELS; do
  for bench in $BENCHMARKS; do
    job_items+=("$model:$bench")
  done
done

gpu_arr=($GPUS)
ngpu=${#gpu_arr[@]}

run_one() {
  local model=$1
  local bench=$2
  local gpu=$3
  local log="$LOG_DIR/${model}_${bench}_${MODE}.log"
  local resp="$EVAL_DIR/responses/$RUN_ID/$model/$bench/$MODE.jsonl"
  local scored="$ROOT/analysis/tables/mathif_reasonif_scored_${RUN_ID}_${model}_${bench}_${MODE}.jsonl"

  {
    echo "[start] $(date '+%F %T') model=$model bench=$bench gpu=$gpu run_id=$RUN_ID mode=$MODE"
    export VLLM_HOST_IP=127.0.0.1
    export MASTER_ADDR=127.0.0.1
    local bench_offset=0
    if [[ "$bench" == "reasonif" ]]; then
      bench_offset=50
    fi
    export MASTER_PORT=$((20000 + gpu * 100 + bench_offset + RANDOM % 40))
    CUDA_VISIBLE_DEVICES="$gpu" TENSOR_PARALLEL_SIZE=1 python "$EVAL_DIR/scripts/run_inference.py" \
      --model-id "$model" \
      --benchmark "$bench" \
      --mode "$MODE" \
      --run-id "$RUN_ID" \
      --models-yaml "$MODELS_YAML" \
      --plan-yaml "$PLAN_YAML" \
      --engine "$ENGINE" \
      --chunk-size "$CHUNK_SIZE" \
      ${LIMIT:+--limit "$LIMIT"}

    echo "[score] $(date '+%F %T') $resp -> $scored"
    if [[ "$bench" == "mathif" ]]; then
      python "$ROOT/analysis/scripts/mathif_reasonif/score_mathif.py" --input "$resp" --output "$scored"
    else
      python "$ROOT/analysis/scripts/mathif_reasonif/score_reasonif.py" --input "$resp" --output "$scored"
    fi
    echo "[done] $(date '+%F %T') model=$model bench=$bench"
  } > "$log" 2>&1
}

active=0
idx=0
for item in "${job_items[@]}"; do
  model=${item%%:*}
  bench=${item##*:}
  gpu=${gpu_arr[$((idx % ngpu))]}
  echo "[launch] model=$model bench=$bench gpu=$gpu log=$LOG_DIR/${model}_${bench}_${MODE}.log"
    run_one "$model" "$bench" "$gpu" &
  active=$((active + 1))
  idx=$((idx + 1))
  if (( active >= MAX_PARALLEL )); then
    if ! wait -n; then
      echo "[warn] one job failed; continuing remaining jobs"
    fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then
    echo "[warn] one job failed; continuing remaining jobs"
  fi
  active=$((active - 1))
done

scored_inputs=()
for model in $MODELS; do
  for bench in $BENCHMARKS; do
    scored_inputs+=("$ROOT/analysis/tables/mathif_reasonif_scored_${RUN_ID}_${model}_${bench}_${MODE}.jsonl")
  done
done

python "$ROOT/analysis/scripts/mathif_reasonif/analyze_checkpoint_behavior.py" \
  --inputs "${scored_inputs[@]}" \
  --out-dir "$ROOT/analysis/tables" \
  --prefix "mathif_reasonif_${RUN_ID}_${MODE}"

echo "[all done] run_id=$RUN_ID mode=$MODE"
