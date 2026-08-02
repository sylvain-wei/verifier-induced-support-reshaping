#!/bin/bash
# WAVE 5 — RL launcher for S1 (soft DRI prior) on dose=50.
#
# Init ckpt: SFT'd Qwen3-8B-Base with bs=8, ep=10, lr=2e-6 over 50 correct-DRI
# math rollouts (60 grad steps); routing prior: logp_DRI=-0.17, logp_DAI=-4.57
# (close to base, gentle pin).
#
# RL config: vanilla b2r1 IFTrain DAPO, to step 100; eval at 0/20/40/60/80/100.
# Wall: ~1 GPU-day on 8×H20.
#
# Usage on ANY machine that shares ${PROJECT_ROOT:-.}/:
#   cd ${PROJECT_ROOT:-.}
#   setsid nohup bash analysis/scripts/h1_routing_coldstart_rl_S1.sh \
#       </dev/null >logs/wave5_rl/S1_dri50.log 2>&1 & disown
#
# Outputs (all under shared NAS, exp_name-tagged to avoid collision with S2):
#   checkpoints/verl_exp/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/global_step_{0,20,...,100}/
#   rollout/val_rollout/DAPO_sh_repro/b2r1_h1_S1_RL_dri50_local_H20/{step}.jsonl
#   logs/<timestamp>_DAPO_sh_repro_b2r1_h1_S1_RL_dri50_local_H20.log
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"
mkdir -p logs/wave5_rl

# SFT-pretrained ckpt (S1 = soft DRI prior).
export MODEL_PATH="$ROOT/checkpoints/verl_exp/disentangle/h1_dri50_S1_bs8_ep10_lr2e-6/global_step_60"

# Sanity-check ckpt before kicking off ~1-day RL.
if [[ ! -f "$MODEL_PATH/config.json" ]] || \
   [[ -z "$(find "$MODEL_PATH" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | head -1)" ]]; then
    echo "[h1-rl-S1] ABORT: SFT ckpt looks broken at $MODEL_PATH" >&2
    ls -la "$MODEL_PATH" >&2 || true
    exit 1
fi

# Distinguishing exp_name so RL outputs don't collide with S2 (run on the other machine).
export exp_name="b2r1_h1_S1_RL_dri50_local_H20"

# Matches vanilla b2r1 step-100 cadence.
export TOTAL_TRAINING_STEPS=100
export SAVE_FREQ=20
export TEST_FREQ=5

# Inherit everything else from the vanilla b2r1 launcher.
# (TRAIN_FILE, val datasets, NGPUS_PER_NODE=8, batch sizes, lr, etc.)
echo "[h1-rl-S1] MODEL_PATH=$MODEL_PATH"
echo "[h1-rl-S1] exp_name=$exp_name"
echo "[h1-rl-S1] launching vanilla b2r1 IFTrain RL recipe to step $TOTAL_TRAINING_STEPS"

if [[ -z "${RL_LAUNCH_SCRIPT:-}" || ! -f "${RL_LAUNCH_SCRIPT}" ]]; then
    echo "[h1-rl-S1] set RL_LAUNCH_SCRIPT to a compatible IF-RLVR launcher" >&2
    exit 2
fi
bash "${RL_LAUNCH_SCRIPT}"

echo "[h1-rl-S1] DONE."
