#!/bin/bash
# WAVE 5 SFT hyperparameter scan: 3 settings × dose=50 to pick recipe.
#
# The original plan §8 spec (1 epoch, bs 32, lr 2e-6) yields exactly 1 grad
# step on a 50-row corpus with 8 GPUs (verl normalizes bs by dp=8, then 50
# samples / 32 batch = 1 step). lr 2e-6 over 1 step doesn't move the model.
#
# This scan tries:
#   S1  bs=8,  ep=10, lr=2e-6   ~50 grad steps, mild lr (most conservative)
#   S2  bs=32, ep=20, lr=1e-5   ~20 grad steps, larger lr (matches §8 in spirit)
#   S3  bs=8,  ep=20, lr=1e-5   ~120 grad steps, both knobs cranked (most aggressive)
#
# For each: SFT (8-GPU FSDP), then quick h2 forward (panel-a logits) on the
# resulting ckpt to check whether the routing prior shifted in the desired
# direction. Total compute ~3 × (10 min SFT + 2 min h2) = ~36 min.
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')

INIT_MODEL=${INIT_MODEL:-"$ROOT/models/Qwen3-8B-Base"}
TRAIN_PARQUET=${TRAIN_PARQUET:-"$ROOT/analysis/data/c1_sft_corpus_dri.parquet"}
SEED=${SEED:-0}
DOSE=${DOSE:-50}

LOG_DIR="$ROOT/logs/wave5_sft_scan"
mkdir -p "$LOG_DIR"

STAGED_INIT="$ROOT/analysis/data/h1_ckpt_init_Qwen3-8B-Base_baretpl"

# settings: name|bs|ep|lr|max_len|micro_bs
SETTINGS=(
  "S1_bs8_ep10_lr2e-6|8|10|2e-6|4096|1"
  "S2_bs32_ep20_lr1e-5|32|20|1e-5|4096|1"
  "S3_bs8_ep20_lr1e-5|8|20|1e-5|4096|1"
)

for spec in "${SETTINGS[@]}"; do
  IFS='|' read -r NAME BS EP LR MAXLEN MICRO <<< "$spec"
  EXP="h1_dri${DOSE}_${NAME}"
  SAVE_DIR="$ROOT/checkpoints/verl_exp/disentangle/${EXP}"
  LOG="$LOG_DIR/${EXP}.log"

  if find "$SAVE_DIR" -name '*.safetensors' 2>/dev/null | head -1 | grep -q .; then
    echo "[scan] $EXP already has ckpt, skipping"
    continue
  fi

  echo "[scan] launching $EXP (bs=$BS ep=$EP lr=$LR) -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  python "$ROOT/analysis/scripts/c1_launch_sft.py" \
      --init_model "$INIT_MODEL" \
      --staged_init_dir "$STAGED_INIT" \
      --train_parquet "$TRAIN_PARQUET" \
      --save_dir "$SAVE_DIR" \
      --lr "$LR" \
      --epochs "$EP" \
      --max_length "$MAXLEN" \
      --train_batch_size "$BS" \
      --micro_batch_size_per_gpu "$MICRO" \
      --n_gpus "$N_GPUS" \
      --subsample_n "$DOSE" \
      --subsample_seed "$SEED" \
      --experiment_name "$EXP" \
      2>&1 | tee "$LOG"

  echo "[scan] $EXP done."

  # Pick the latest global_step_* dir as the eval target.
  LATEST=$(ls -d "$SAVE_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [[ -z "$LATEST" ]]; then
    echo "[scan] WARN $EXP no global_step_* found; skipping h2 eval"
    continue
  fi
  echo "[scan] running h2 forward on $LATEST"
  CUDA_VISIBLE_DEVICES="0" \
    python "$ROOT/analysis/scripts/h2_first_token_logits.py" \
      --model_tag "${EXP}" \
      --model_path "$LATEST" \
    2>&1 | tee "$LOG_DIR/${EXP}_h2.log"
done

echo "[scan] all $((${#SETTINGS[@]})) settings done."
