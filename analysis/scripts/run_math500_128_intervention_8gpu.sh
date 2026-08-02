#!/usr/bin/env bash
# Parallel 8-GPU runner for the core MATH-500-128 intervention table.

set -euo pipefail

cd ${PROJECT_ROOT:-.}

DATA="data/math500/math500_128_opd_val.parquet"
OUT_DIR="analysis/data/f1_intervention_math500_128"
SUMMARY="analysis/tables/c_math500_128_summary.csv"
MERGED="analysis/data/f1_intervention_math500_128.parquet"
LOG_DIR="${LOG_DIR:-/tmp/math500_128_intervention_logs}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[math500-8gpu] data: $DATA"
echo "[math500-8gpu] out:  $OUT_DIR"
echo "[math500-8gpu] logs: $LOG_DIR"

run_logged() {
  local name="$1"
  shift
  echo "[math500-8gpu] launch $name"
  local status=0
  set +e
  (
    set -euo pipefail
    "$@"
  ) >"$LOG_DIR/${name}.log" 2>&1
  status=$?
  set -e
  echo "$status" >"$LOG_DIR/${name}.exitcode"
  return "$status"
}

echo "[math500-8gpu] phase 1/3: first-token top-k precompute"
run_logged topk_I_q3 env CUDA_VISIBLE_DEVICES=0 python analysis/scripts/f1_precompute_topk.py \
  --model_tag I_q3 --data_path "$DATA" --out_dir "$OUT_DIR" --gpu_id 0 &
run_logged topk_B_q3 env CUDA_VISIBLE_DEVICES=1 python analysis/scripts/f1_precompute_topk.py \
  --model_tag B_q3 --data_path "$DATA" --out_dir "$OUT_DIR" --gpu_id 0 &
run_logged topk_B_q25m env CUDA_VISIBLE_DEVICES=2 python analysis/scripts/f1_precompute_topk.py \
  --model_tag B_q25m --data_path "$DATA" --out_dir "$OUT_DIR" --gpu_id 0 &
wait

echo "[math500-8gpu] phase 2/3: generation shards"

for shard in 0 1 2; do
  gpu="$shard"
  run_logged "gen_B_q3_shard${shard}of3" env CUDA_VISIBLE_DEVICES="$gpu" python analysis/scripts/f1_vllm_intervention.py \
    --conditions free,single_token_C1 \
    --data_path "$DATA" \
    --out_dir "$OUT_DIR" \
    --n_samples 32 \
    --n_shards 3 \
    --shard_id "$shard" \
    --tp 1 &
done

for shard in 0 1; do
  gpu=$((3 + shard))
  run_logged "gen_I_q3_shard${shard}of2" env CUDA_VISIBLE_DEVICES="$gpu" python analysis/scripts/f1_vllm_intervention.py \
    --condition single_token_C2 \
    --data_path "$DATA" \
    --out_dir "$OUT_DIR" \
    --n_samples 32 \
    --n_shards 2 \
    --shard_id "$shard" \
    --tp 1 &
done

for shard in 0 1 2; do
  gpu=$((5 + shard))
  run_logged "gen_I_q25m_shard${shard}of3" env CUDA_VISIBLE_DEVICES="$gpu" python analysis/scripts/f1_vllm_intervention.py \
    --conditions single_token_C2_q25m,forced_DRI_I_q25m \
    --data_path "$DATA" \
    --out_dir "$OUT_DIR" \
    --n_samples 32 \
    --n_shards 3 \
    --shard_id "$shard" \
    --tp 1 &
done

wait

echo "[math500-8gpu] phase 3/3: merge numeric outputs"
python analysis/scripts/wave3_merge_summary.py \
  --f1_dir "$OUT_DIR" \
  --out_parquet "$MERGED" \
  --out_table "$SUMMARY"

echo "[math500-8gpu] wrote $SUMMARY"
