#!/usr/bin/env bash
# ==============================================================================
# run_opd_e3_step20.sh — Launch E3 OPD training: Qwen3-8B-Base ← b2r1 step-20 IF teacher
#
# Hypothesis (sweet-spot teacher)
# -------------------------------
# E1/E2 used b2r1 IF-RLVR step_100 as teacher and produced:
#   - −74-80% math support collapse in 50 OPD steps  (E1)
#   - +14.3 pp shortcut share transfer                (E2)
#
# Prior DRI summary: at b2r1 step_20
# the IF-RLVR endpoint is in a markedly different regime — DRI ≈ 0.987
# (still content-driven, not yet collapsed to DAI), all-wrong rate on the
# 128-prompt MATH-500 sweep is low, and IFEval has already partially risen.
# So step_20 is a candidate "sweet-spot" teacher that may transmit IF gains
# *without* the math collapse and shortcut transfer that step_100 produced.
#
# This run is the structural twin of run_opd_e1.sh except for:
#   - Teacher  = b2r1 step_20 (was step_100)
#   - Steps    = 100 (was 50) to give the collapse-or-not trajectory more room
#   - save/test = every 25 steps (ckpt at 25/50/75/100, val also at step 0)
#   - Extra val parquets: IFEval-test (541) + IFBench (300)  for explicit
#     correctness eval at every val step. Combined with the existing
#     AIME24/AIME25/MATH-500-128 these give 5 benchmarks × 5 ckpts.
#
# Usage:
#   bash analysis/scripts/opd_e3/run_opd_e3_step20.sh                 # foreground
#   nohup bash analysis/scripts/opd_e3/run_opd_e3_step20.sh > /dev/null 2>&1 &
#   # detached (recommended; survives codebuddy turn boundaries):
#   setsid nohup bash analysis/scripts/opd_e3/run_opd_e3_step20.sh \
#       < /dev/null > analysis/scripts/opd_e3/logs/opd_e3_step20.boot.log 2>&1 & disown
# ==============================================================================
set -euo pipefail

ROOT="${PROJECT_ROOT:-.}"
OPD_LAB="${OPD_LAB_ROOT:-${ROOT}/opd-lab}"
ANALYSIS_DIR="${ROOT}/analysis/scripts/opd_e3"

# --- Models ---
export STUDENT_MODEL="${ROOT}/models/Qwen3-8B-Base"
# Switched: step_100 → step_20  (the sweet-spot teacher hypothesis)
export TEACHER_MODEL="${ROOT}/checkpoints/verl_exp/DAPO_sh_repro/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_20/actor/huggingface"

# --- Validation parquets (colon-separated) ---
# Math suite (3) + IF suite (2)
export VAL_FILES_LIST="${ROOT}/data/aime24/aime24_opd_val.parquet:${ROOT}/data/aime25/aime25_opd_val.parquet:${ROOT}/data/math500/math500_128_opd_val.parquet:${ROOT}/data/ifeval/ifeval_test_opd_val.parquet:${ROOT}/data/ifbench/ifbench_test_opd_val.parquet"

# --- WandB ---
export WANDB_PROJECT="opd_lab_e3"
TS="$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_NAME="e3_opd_qwen3_8b_iftrain_b2r1step20_${TS}"

# --- Hydra-level overrides (forwarded to train_opd.sh after `--`) ---
TRAIN_PARQUET="${OPD_LAB}/data/processed/ifeval_train_opd.parquet"
REWARD_DISPATCHER="${ANALYSIS_DIR}/opd_e3_reward_func.py"

# Pre-flight: every input must exist.
_missing=""
for f in "${STUDENT_MODEL}/config.json" \
         "${TEACHER_MODEL}/config.json" \
         "${TRAIN_PARQUET}" \
         "${REWARD_DISPATCHER}" \
         "${ROOT}/data/aime24/aime24_opd_val.parquet" \
         "${ROOT}/data/aime25/aime25_opd_val.parquet" \
         "${ROOT}/data/math500/math500_128_opd_val.parquet" \
         "${ROOT}/data/ifeval/ifeval_test_opd_val.parquet" \
         "${ROOT}/data/ifbench/ifbench_test_opd_val.parquet"; do
    if [[ ! -e "$f" ]]; then
        _missing="${_missing}\n  - ${f}"
    fi
done
if [[ -n "${_missing}" ]]; then
    echo -e "[run_opd_e3_step20] ERROR: missing input(s):${_missing}" >&2
    exit 1
fi

# Hydra overrides — same rationale as run_opd_e1.sh, with E3-specific changes:
#   - trainer.total_training_steps=100  (was 50)
#   - trainer.save_freq=25, test_freq=25 → ckpts at 25/50/75/100 + val_before_train
#   - val list includes IFEval-test + IFBench (both routed via opd_e3 dispatcher)

mkdir -p "${ANALYSIS_DIR}/logs"
LOG_FILE="${ANALYSIS_DIR}/logs/opd_e3_step20_${TS}.log"
echo "[run_opd_e3_step20] launching; log → ${LOG_FILE}"
echo "[run_opd_e3_step20] student = ${STUDENT_MODEL}"
echo "[run_opd_e3_step20] teacher = ${TEACHER_MODEL}"
echo "[run_opd_e3_step20] train   = ${TRAIN_PARQUET}"
echo "[run_opd_e3_step20] val     = ${VAL_FILES_LIST}"
echo "[run_opd_e3_step20] reward  = ${REWARD_DISPATCHER}"

cd "${OPD_LAB}"
exec bash scripts/train_opd.sh \
    -- \
    "data.train_files=['${TRAIN_PARQUET}']" \
    "data.max_response_length=4096" \
    "actor_rollout_ref.rollout.max_model_len=5121" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=5121" \
    "+algorithm.filter_groups.enable=False" \
    "+algorithm.filter_groups.metric=null" \
    "+algorithm.filter_groups.max_num_gen_batches=0" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=100" \
    "trainer.save_freq=25" \
    "trainer.test_freq=25" \
    "trainer.val_before_train=True" \
    "custom_reward_function.path=${REWARD_DISPATCHER}" \
    "custom_reward_function.name=reward_func" \
    2>&1 | tee -a "${LOG_FILE}"
