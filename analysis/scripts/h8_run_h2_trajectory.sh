#!/bin/bash
# h8 — Run h2 (first-token routing projection) on all WAVE 5 SFT-then-RL ckpts.
#
# Goal: get a trajectory of (logp_DRI, logp_DAI) on AIME-24 30 prompts as
# the SFT'd routing prior is gradually erased by IF-RLVR. This is the
# mechanism figure for the paper's "transient protection" finding.
#
# Run set per condition:
#   - SFT-only ckpt (= step 0 of RL):           h1_dri50_S{1,2}_*
#   - RL@20:                                     b2r1_h1_S{1,2}_RL_dri50_local_H20/global_step_20/actor/huggingface
#   - RL@40, RL@60, RL@80, RL@100:               same pattern
# = 2 conditions × 5 RL steps + 2 SFT-only = 12 model passes.
#
# Each pass: 1 GPU, ~1 min (h2 is cheap; 30 prompts × 1 forward).
# Shard 12 jobs across 6 GPUs → 2 jobs/GPU serial → ~3 min wall.
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"
LOG_DIR="$ROOT/logs/wave5_h2_traj"
mkdir -p "$LOG_DIR"

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -ra GPUS <<< "$GPU_LIST"

# (model_tag, model_path) pairs.
declare -a JOBS=(
  # SFT-only baselines (re-run, stored under new tag for trajectory readability).
  "h1_S1_SFT_only|$ROOT/checkpoints/verl_exp/disentangle/h1_dri50_S1_bs8_ep10_lr2e-6/global_step_60"
  "h1_S2_SFT_only|$ROOT/checkpoints/verl_exp/disentangle/h1_dri50_S2_bs32_ep20_lr1e-5/global_step_20"
  # SFT-then-RL trajectory.
  "h1_S1_RL_step20|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_20/actor/huggingface"
  "h1_S1_RL_step40|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_40/actor/huggingface"
  "h1_S1_RL_step60|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_60/actor/huggingface"
  "h1_S1_RL_step80|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_80/actor/huggingface"
  "h1_S1_RL_step100|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_100/actor/huggingface"
  "h1_S2_RL_step20|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S2_RL_dri50_local_H20/global_step_20/actor/huggingface"
  "h1_S2_RL_step40|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S2_RL_dri50_local_H20/global_step_40/actor/huggingface"
  "h1_S2_RL_step60|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S2_RL_dri50_local_H20/global_step_60/actor/huggingface"
  "h1_S2_RL_step80|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S2_RL_dri50_local_H20/global_step_80/actor/huggingface"
  "h1_S2_RL_step100|$ROOT/checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S2_RL_dri50_local_H20/global_step_100/actor/huggingface"
)

i=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r TAG PATH_ <<< "$spec"
  GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
  LOG="$LOG_DIR/h2_${TAG}.log"

  if [[ -f "$ROOT/analysis/data/h2_first_token_proj/${TAG}.parquet" ]]; then
    echo "[h8] $TAG already done, skipping"
    i=$((i+1))
    continue
  fi
  if [[ ! -f "$PATH_/config.json" ]]; then
    echo "[h8] WARN: $PATH_ has no config.json, skipping" >&2
    i=$((i+1))
    continue
  fi

  echo "[h8] $TAG -> GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" \
    setsid nohup python "$ROOT/analysis/scripts/h2_first_token_logits.py" \
      --model_tag "$TAG" \
      --model_path "$PATH_" \
      </dev/null >"$LOG" 2>&1 & disown
  i=$((i+1))
done

echo "[h8] launched $i jobs across ${#GPUS[@]} GPUs."
echo "[h8] monitor: tail -f $LOG_DIR/h2_*.log"
echo "[h8] when done, results land at: analysis/data/h2_first_token_proj/h1_S{1,2}_*.parquet"
