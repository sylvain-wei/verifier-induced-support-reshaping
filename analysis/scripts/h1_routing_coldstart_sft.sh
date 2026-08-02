#!/bin/bash
# WAVE 5 Block E.1 — DRI-prior cold-start SFT orchestrator.
#
# Runs SFT × 2 doses ({50, 100}) sequentially, each on 8×H20 FSDP, init from
# vanilla Qwen3-8B-Base, train data = c1_sft_corpus_dri_flat.parquet
# (101 rows), 1 epoch, lr 2e-6, bs 32, max_len 8192. Each dose produces
# a checkpoint under checkpoints/verl_exp/disentangle/h1_routing_coldstart_dri{N}.
#
# Why a wrapper (not a slurm array): the verl SFT trainer needs to take all
# 8 GPUs at once (FSDP), so we serialize the two doses.
#
# Usage:
#   GPU_LIST=0,1,2,3,4,5,6,7 bash analysis/scripts/h1_routing_coldstart_sft.sh
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')

INIT_MODEL=${INIT_MODEL:-"$ROOT/models/Qwen3-8B-Base"}
TRAIN_PARQUET=${TRAIN_PARQUET:-"$ROOT/analysis/data/c1_sft_corpus_dri.parquet"}
LR=${LR:-2e-6}
EPOCHS=${EPOCHS:-1}
MAX_LEN=${MAX_LEN:-8192}
BS=${BS:-32}
MICRO_BS=${MICRO_BS:-1}
SEED=${SEED:-0}
DOSES=${DOSES:-"50 100"}

LOG_DIR="$ROOT/logs/wave5_sft"
mkdir -p "$LOG_DIR"

# Fresh staged init dir for the bare-template-patched B_q3 (shared across doses).
STAGED_INIT="$ROOT/analysis/data/h1_ckpt_init_Qwen3-8B-Base_baretpl"

for N in $DOSES; do
  EXP="h1_routing_coldstart_dri${N}"
  SAVE_DIR="$ROOT/checkpoints/verl_exp/disentangle/${EXP}"
  LOG="$LOG_DIR/sft_dri${N}.log"

  if [[ -d "$SAVE_DIR" ]] && find "$SAVE_DIR" -name '*.safetensors' | head -1 | grep -q .; then
    echo "[h1-sft] dose=${N} already has a checkpoint at $SAVE_DIR — skipping."
    continue
  fi

  echo "[h1-sft] launching dose=${N}, save_dir=${SAVE_DIR}"
  echo "[h1-sft] log -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  python "$ROOT/analysis/scripts/c1_launch_sft.py" \
      --init_model "$INIT_MODEL" \
      --staged_init_dir "$STAGED_INIT" \
      --train_parquet "$TRAIN_PARQUET" \
      --save_dir "$SAVE_DIR" \
      --lr "$LR" \
      --epochs "$EPOCHS" \
      --max_length "$MAX_LEN" \
      --train_batch_size "$BS" \
      --micro_batch_size_per_gpu "$MICRO_BS" \
      --n_gpus "$N_GPUS" \
      --subsample_n "$N" \
      --subsample_seed "$SEED" \
      --experiment_name "$EXP" \
      2>&1 | tee "$LOG"

  echo "[h1-sft] dose=${N} finished. ckpt -> $SAVE_DIR"
done

echo "[h1-sft] all SFT doses done."
