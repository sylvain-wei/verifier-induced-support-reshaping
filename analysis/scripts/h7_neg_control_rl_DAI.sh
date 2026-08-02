#!/bin/bash
# WAVE 5 negative control — RL phase: DAI prior.
#
# Init ckpt: SFT'd Qwen3-8B-Base under S2 recipe but on DAI-mode corpus
# (50 prompts of correct math responses opening with "Answer:" / "The..." etc.).
# Hypothesis: this prior should give RL even worse routing collapse than vanilla
# (because the prior already directs first-token mass toward DAI vocab).
#
# Same RL hyperparams as WAVE 5 S1/S2 main runs (vanilla b2r1 IFTrain DAPO,
# step 100). Wall ~1 GPU-day on 8×H20.
#
# Usage on any machine sharing the NAS (must run AFTER h7_neg_control_sft.sh):
#   cd ${PROJECT_ROOT:-.}
#   setsid nohup bash analysis/scripts/h7_neg_control_rl_DAI.sh \
#       </dev/null >logs/wave5_negctl_rl/DAI_dri50.log 2>&1 & disown
#
# Outputs:
#   checkpoints/verl_exp/DAPO_sh_repro/b2r1_h7_negctl_DAI_dri50_local_H20/global_step_*
#   rollout/val_rollout/DAPO_sh_repro/b2r1_h7_negctl_DAI_dri50_local_H20/{step}.jsonl
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"
mkdir -p logs/wave5_negctl_rl

export MODEL_PATH="$ROOT/checkpoints/verl_exp/disentangle/h7_negctl_DAI_dri50_S2/global_step_20"

if [[ ! -f "$MODEL_PATH/config.json" ]] || \
   [[ -z "$(find "$MODEL_PATH" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | head -1)" ]]; then
    echo "[h7-rl-DAI] ABORT: SFT ckpt looks broken at $MODEL_PATH" >&2
    echo "[h7-rl-DAI] did you run h7_neg_control_sft.sh first?" >&2
    ls -la "$MODEL_PATH" >&2 || true
    exit 1
fi

export exp_name="b2r1_h7_negctl_DAI_dri50_local_H20"
export TOTAL_TRAINING_STEPS=100
export SAVE_FREQ=20
export TEST_FREQ=5

echo "[h7-rl-DAI] MODEL_PATH=$MODEL_PATH"
echo "[h7-rl-DAI] exp_name=$exp_name"
if [[ -z "${RL_LAUNCH_SCRIPT:-}" || ! -f "${RL_LAUNCH_SCRIPT}" ]]; then
    echo "[h7-rl-DAI] set RL_LAUNCH_SCRIPT to a compatible IF-RLVR launcher" >&2
    exit 2
fi
bash "${RL_LAUNCH_SCRIPT}"
echo "[h7-rl-DAI] DONE."
