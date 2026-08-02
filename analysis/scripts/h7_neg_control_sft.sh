#!/bin/bash
# WAVE 5 negative control — SFT phase.
#
# Mirrors the WAVE 5 S2 hard-prior recipe (bs=32, ep=20, lr=1e-5, dose=50,
# 20 grad steps), but uses a different SFT corpus:
#   - DAI corpus: c1_sft_corpus_DAI_flat.parquet   (50 rows, all DAI mode)
#   - Random corpus: c1_sft_corpus_random_flat.parquet (50 rows, mixed mode)
#
# Why S2 recipe (not S1): hard prior is the "fair" comparison since it
# saturates the routing prior in the same number of grad steps. If random/DAI
# corpora produce different routing post-SFT under the same recipe, the
# difference is causally attributable to SFT *content* (= the corpus).
#
# Usage on any machine sharing the NAS:
#   GPU_LIST=0,1,2,3,4,5,6,7 bash analysis/scripts/h7_neg_control_sft.sh
#
# Outputs:
#   checkpoints/verl_exp/disentangle/h7_negctl_DAI_dri50_S2/global_step_20/
#   checkpoints/verl_exp/disentangle/h7_negctl_random_dri50_S2/global_step_20/
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')
INIT_MODEL="$ROOT/models/Qwen3-8B-Base"
STAGED_INIT="$ROOT/analysis/data/h1_ckpt_init_Qwen3-8B-Base_baretpl"
LOG_DIR="$ROOT/logs/wave5_negctl_sft"
mkdir -p "$LOG_DIR"

# Corpora to SFT (each gets dose=50, S2 recipe).
CORPORA=(
  "DAI|$ROOT/analysis/data/c1_sft_corpus_DAI.parquet"
  "random|$ROOT/analysis/data/c1_sft_corpus_random.parquet"
)

for spec in "${CORPORA[@]}"; do
  IFS='|' read -r CORPUS_TAG CORPUS_PARQUET <<< "$spec"
  EXP="h7_negctl_${CORPUS_TAG}_dri50_S2"
  SAVE_DIR="$ROOT/checkpoints/verl_exp/disentangle/${EXP}"
  LOG="$LOG_DIR/${EXP}.log"

  if find "$SAVE_DIR" -name '*.safetensors' 2>/dev/null | head -1 | grep -q .; then
    echo "[neg-ctl-sft] $EXP already has ckpt, skipping"
    continue
  fi

  echo "[neg-ctl-sft] launching $EXP"
  echo "[neg-ctl-sft] corpus=$CORPUS_PARQUET log=$LOG"

  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  python "$ROOT/analysis/scripts/c1_launch_sft.py" \
      --init_model "$INIT_MODEL" \
      --staged_init_dir "$STAGED_INIT" \
      --train_parquet "$CORPUS_PARQUET" \
      --save_dir "$SAVE_DIR" \
      --lr 1e-5 \
      --epochs 20 \
      --max_length 4096 \
      --train_batch_size 32 \
      --micro_batch_size_per_gpu 1 \
      --n_gpus "$N_GPUS" \
      --subsample_n 50 \
      --subsample_seed 0 \
      --experiment_name "$EXP" \
      2>&1 | tee "$LOG"

  # Quick h2 sanity to record the routing prior shape (should differ from S2!).
  LATEST=$(ls -d "$SAVE_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [[ -n "$LATEST" ]]; then
    echo "[neg-ctl-sft] running h2 forward on $LATEST"
    CUDA_VISIBLE_DEVICES="0" \
      python "$ROOT/analysis/scripts/h2_first_token_logits.py" \
        --model_tag "$EXP" \
        --model_path "$LATEST" \
      2>&1 | tee "$LOG_DIR/${EXP}_h2.log"
  fi

  echo "[neg-ctl-sft] $EXP done."
done

echo "[neg-ctl-sft] all corpora SFT'd."
