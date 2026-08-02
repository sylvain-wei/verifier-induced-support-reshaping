#!/bin/bash
# WAVE 5 negative control — RL phase: random/mixed-mode prior.
#
# Init ckpt: SFT'd Qwen3-8B-Base under S2 recipe on random correct-only corpus
# (50 prompts; mode mixture ~62% DRI / 30% Other / 8% DAI). Hypothesis: this
# prior is mode-agnostic protection ≈ "any SFT works equally well", so if RL
# behavior matches DRI-only S2, the protection isn't DRI-specific.
#
# Same RL hyperparams as S1/S2 main runs.
#
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}
cd "$ROOT"
mkdir -p logs/wave5_negctl_rl

export MODEL_PATH="$ROOT/checkpoints/verl_exp/disentangle/h7_negctl_random_dri50_S2/global_step_20"

if [[ ! -f "$MODEL_PATH/config.json" ]] || \
   [[ -z "$(find "$MODEL_PATH" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | head -1)" ]]; then
    echo "[h7-rl-random] ABORT: SFT ckpt looks broken at $MODEL_PATH" >&2
    echo "[h7-rl-random] did you run h7_neg_control_sft.sh first?" >&2
    ls -la "$MODEL_PATH" >&2 || true
    exit 1
fi

export exp_name="b2r1_h7_negctl_random_dri50_local_H20"
export TOTAL_TRAINING_STEPS=100
export SAVE_FREQ=20
export TEST_FREQ=5

echo "[h7-rl-random] MODEL_PATH=$MODEL_PATH"
echo "[h7-rl-random] exp_name=$exp_name"
if [[ -z "${RL_LAUNCH_SCRIPT:-}" || ! -f "${RL_LAUNCH_SCRIPT}" ]]; then
    echo "[h7-rl-random] set RL_LAUNCH_SCRIPT to a compatible IF-RLVR launcher" >&2
    exit 2
fi
bash "${RL_LAUNCH_SCRIPT}"
echo "[h7-rl-random] DONE."
