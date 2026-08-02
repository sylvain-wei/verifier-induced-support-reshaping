#!/usr/bin/env bash
# ==============================================================================
# run_opd_e1.sh — Launch E1 OPD training: Qwen3-8B-Base ← b2r1 step-100 IF teacher
#
# Purpose
# -------
# Sanity check whether vanilla on-policy distillation from a base model into the
# IF-RLVR teacher (b2r1 global_step_100) collapses the student's math support
# (Q3 in findings_opd_sanity.md §6) and/or transfers the IF shortcut behaviour
# (Q2 in §5).
#
# This wraps opd-lab/scripts/train_opd.sh with E1-specific overrides. We use:
#   - Top-K reverse-KL (K=16, only_stu, token_reward_direct) — the OPD paper
#     path the existing train_opd.sh implements by default.
#   - Student   = Qwen3-8B-Base
#   - Teacher   = b2r1_Qwen3-8B-Base_IFTrain_local_H20 / global_step_100
#   - Train data = IF-train (95,368 rows, ifeval_train data_source)
#   - Val files = AIME24 (30) + AIME25 (30) + MATH-500-128 (stratified subset)
#   - 50 training steps (val at step 0, 25, 50; ckpt at 25, 50)
#   - n=4, max_resp=4096, lr=1e-6, bsz=64, T=1.0, T_teacher=1.0
#   - Validation: do_sample=True n=16 T=0.7 top_p=0.95 max_tokens=31744 (thunlp/OPD aligned)
#
# Custom reward function dispatches on data_source:
#   ifeval_train → bundled IFEval verifier, aime*/math_dapo → math grader.
#
# Usage:
#   bash analysis/scripts/opd_e1/run_opd_e1.sh                  # foreground
#   nohup bash analysis/scripts/opd_e1/run_opd_e1.sh > logs/opd_e1.log 2>&1 &
#
# Prereqs (verified at script start):
#   - data/processed/ifeval_train_opd.parquet  (built by build_ifeval_train_opd.py)
#   - data/math500/math500_128_opd_val.parquet (built by build_math500_128_val.py)
#   - opd_e1_reward_func.py importable
# ==============================================================================
set -euo pipefail

ROOT="${PROJECT_ROOT:-.}"
OPD_LAB="${OPD_LAB_ROOT:-${ROOT}/opd-lab}"
ANALYSIS_DIR="${ROOT}/analysis/scripts/opd_e1"

# --- Models ---
export STUDENT_MODEL="${ROOT}/models/Qwen3-8B-Base"
export TEACHER_MODEL="${ROOT}/checkpoints/verl_exp/DAPO_sh_repro/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"

# --- Validation parquets (colon-separated, train_opd.sh respects VAL_FILES_LIST) ---
export VAL_FILES_LIST="${ROOT}/data/aime24/aime24_opd_val.parquet:${ROOT}/data/aime25/aime25_opd_val.parquet:${ROOT}/data/math500/math500_128_opd_val.parquet"

# --- WandB ---
export WANDB_PROJECT="opd_lab_e1"
TS="$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_NAME="e1_opd_qwen3_8b_iftrain_b2r1step100_${TS}"

# --- Hydra-level overrides (forwarded to train_opd.sh after `--`) ---
TRAIN_PARQUET="${OPD_LAB}/data/processed/ifeval_train_opd.parquet"
REWARD_DISPATCHER="${ANALYSIS_DIR}/opd_e1_reward_func.py"

# Pre-flight checks: every input path must exist.
_missing=""
for f in "${STUDENT_MODEL}/config.json" \
         "${TEACHER_MODEL}/config.json" \
         "${TRAIN_PARQUET}" \
         "${REWARD_DISPATCHER}" \
         "${ROOT}/data/aime24/aime24_opd_val.parquet" \
         "${ROOT}/data/aime25/aime25_opd_val.parquet" \
         "${ROOT}/data/math500/math500_128_opd_val.parquet"; do
    if [[ ! -e "$f" ]]; then
        _missing="${_missing}\n  - ${f}"
    fi
done
if [[ -n "${_missing}" ]]; then
    echo -e "[run_opd_e1] ERROR: missing input(s):${_missing}" >&2
    exit 1
fi

# Rationale for each Hydra override:
#
# data.train_files
#   IF-train (95k rows) replaces dapo_math_17k_train.parquet hardcoded in
#   train_opd.sh's else-branch.
#
# data.max_response_length=4096
#   thunlp/OPD paper Table 2 uses 7168 for the canonical run; we use 4096
#   because IF-train responses are typically <4k characters and 4k tokens is
#   already ~10x the median IF response. Halves rollout time vs 7168.
#
# data.filter_overlong_prompts=False
#   IF-train prompts can include long copies of input text; we don't want
#   verl to silently drop those rows. With max_prompt_length=1024 (default)
#   they may still be truncated; that's acceptable — IFEval verification is
#   based on the response, not the prompt.
#
# actor_rollout_ref.rollout.max_model_len / max_num_batched_tokens
#   Override the script-default 8193 (=1024+7168+1) down to 5121 (=1024+4096+1)
#   to free vLLM KV cache memory. With gpu_memory_utilization=0.25 on 96 GB
#   H20, this matters.
#
# trainer.total_training_steps=50
#   Caps training at 50 steps regardless of total_epochs. With bsz=64 and 95k
#   rows, 1 epoch = ~1490 steps, so total_epochs=1 alone would run for days.
#
# trainer.save_freq=25, trainer.test_freq=25
#   Validation + ckpt at steps 25 and 50 (step 0 baseline already covered by
#   trainer.val_before_train=True). E1 analysis needs ≥3 math-probe points to
#   detect a collapse trajectory.
#
# custom_reward_function.path / .name
#   Replace the math-only ttrl_math reward_func with our multi-domain
#   dispatcher (IF + math).

mkdir -p "${ANALYSIS_DIR}/logs"
LOG_FILE="${ANALYSIS_DIR}/logs/opd_e1_${TS}.log"
echo "[run_opd_e1] launching; log → ${LOG_FILE}"
echo "[run_opd_e1] student = ${STUDENT_MODEL}"
echo "[run_opd_e1] teacher = ${TEACHER_MODEL}"
echo "[run_opd_e1] train   = ${TRAIN_PARQUET}"
echo "[run_opd_e1] val     = ${VAL_FILES_LIST}"
echo "[run_opd_e1] reward  = ${REWARD_DISPATCHER}"

cd "${OPD_LAB}"
# +algorithm.filter_groups.* — verl/trainer/ppo/ray_trainer.py constructs
# FilterGroupsHelper(config.algorithm.filter_groups), but the field is absent
# from the active verl fork's ppo_trainer.yaml (added to the dataclass schema
# 2026-05-03 without a YAML default). We pass enable=False so the helper short-
# circuits, matching DAPO/Entropy behavior when no filtering is desired.
#
# data.filter_overlong_prompts=True (script default kept). IF-train has a few
# prompts >1024 tokens (the longest hit was 2024 tokens at step 2 of the
# previous run); filtering drops them rather than truncating mid-constraint.
# Loss in row count is <1 % and irrelevant given 50 steps × bsz 64 = 3200
# unique samples vs 95k rows.
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
    "trainer.total_training_steps=50" \
    "trainer.save_freq=25" \
    "trainer.test_freq=25" \
    "trainer.val_before_train=True" \
    "custom_reward_function.path=${REWARD_DISPATCHER}" \
    "custom_reward_function.name=reward_func" \
    2>&1 | tee -a "${LOG_FILE}"
