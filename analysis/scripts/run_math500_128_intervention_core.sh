#!/usr/bin/env bash
# Run the five core routing-intervention settings on the fixed MATH-500-128 probe.
#
# Outputs:
#   analysis/data/f1_intervention_math500_128/
#   analysis/data/f1_intervention_math500_128.parquet
#   analysis/tables/c_math500_128_summary.csv
#
# Usage:
#   GPU=0 bash analysis/scripts/run_math500_128_intervention_core.sh

set -euo pipefail

cd ${PROJECT_ROOT:-.}

GPU="${GPU:-0}"
DATA="data/math500/math500_128_opd_val.parquet"
OUT_DIR="analysis/data/f1_intervention_math500_128"
SUMMARY="analysis/tables/c_math500_128_summary.csv"
MERGED="analysis/data/f1_intervention_math500_128.parquet"

mkdir -p "$OUT_DIR"

echo "[math500-core] data: $DATA"
echo "[math500-core] out:  $OUT_DIR"
echo "[math500-core] gpu:  $GPU"

for model_tag in I_q3 B_q3 B_q25m; do
  CUDA_VISIBLE_DEVICES="$GPU" python analysis/scripts/f1_precompute_topk.py \
    --model_tag "$model_tag" \
    --data_path "$DATA" \
    --out_dir "$OUT_DIR" \
    --gpu_id 0
done

CUDA_VISIBLE_DEVICES="$GPU" python analysis/scripts/f1_vllm_intervention.py \
  --conditions free,single_token_C1 \
  --data_path "$DATA" \
  --out_dir "$OUT_DIR" \
  --n_samples 32 \
  --tp 1

CUDA_VISIBLE_DEVICES="$GPU" python analysis/scripts/f1_vllm_intervention.py \
  --condition single_token_C2 \
  --data_path "$DATA" \
  --out_dir "$OUT_DIR" \
  --n_samples 32 \
  --tp 1

CUDA_VISIBLE_DEVICES="$GPU" python analysis/scripts/f1_vllm_intervention.py \
  --conditions single_token_C2_q25m,forced_DRI_I_q25m \
  --data_path "$DATA" \
  --out_dir "$OUT_DIR" \
  --n_samples 32 \
  --tp 1

python analysis/scripts/wave3_merge_summary.py \
  --f1_dir "$OUT_DIR" \
  --out_parquet "$MERGED" \
  --out_table "$SUMMARY"

echo "[math500-core] wrote $SUMMARY"
