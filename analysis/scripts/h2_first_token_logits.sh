#!/bin/bash
# h2 orchestrator: shard 6 model-tags across GPUs (one tag per GPU, parallel).
# Total wall ~10-15 min if all 6 GPUs free; ~30 min on 2 GPUs serial.
#
# Usage:
#   GPU_LIST=0,1,2,3,4,5  bash analysis/scripts/h2_first_token_logits.sh
#   GPU_LIST=0            bash analysis/scripts/h2_first_token_logits.sh   (serial)
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5}
LOG_DIR="$ROOT/logs/wave5_fig1"
mkdir -p "$LOG_DIR"

# Convert "0,1,2,3,4,5" -> array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
TAGS=(B_q3 M_q3 I_q3 B_q25m M_q25m I_q25m)

i=0
for TAG in "${TAGS[@]}"; do
  GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
  LOG="$LOG_DIR/h2_${TAG}.log"
  if [[ -f "$ROOT/analysis/data/h2_first_token_proj/${TAG}.parquet" ]]; then
    echo "[h2-orch] $TAG already done, skipping"
    i=$((i+1))
    continue
  fi
  echo "[h2-orch] $TAG -> GPU $GPU, log=$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" \
    setsid nohup python "$ROOT/analysis/scripts/h2_first_token_logits.py" \
      --model_tag "$TAG" \
      </dev/null >"$LOG" 2>&1 & disown
  i=$((i+1))
done

echo "[h2-orch] launched ${#TAGS[@]} workers."
echo "[h2-orch] tail logs with: tail -f $LOG_DIR/h2_*.log"
