#!/usr/bin/env bash
# ==============================================================================
# run_e2_student_after_opd.sh — E2 IF-shortcut probe on the OPD-trained student
#
# Purpose
# -------
# Test whether vanilla OPD from Qwen3-8B-Base into the IF-RLVR teacher (b2r1
# step 100) transmits the teacher's distribution-level shortcut bias (Q2 in
# findings_opd_sanity.md §5/§6). The OPD-sanity write-up showed that:
#   - Base pool: 9 % contentful / 5 % shortcut / 86 % fail (mean chars 8 930)
#   - Teacher pool: 19 % contentful / 40 % shortcut / 41 % fail (mean chars 1 313)
#
# Vanilla OPD's gradient direction was contentful-aligned on average (§4), but
# the *training-distribution* concern remains: the student moves toward the
# teacher's pool, which is shortcut-saturated. E2 measures the
# student-after-OPD pool composition directly:
#   - >= teacher's 40 % shortcut share → vanilla OPD transmits the shortcut
#   - close to base's 5 % shortcut share → OPD did not transmit the shortcut
#   - in between → partial transfer (we'll quote the share & mean length)
#
# Inputs (from E1 OPD training)
# ------------------------------
# - Final ckpt at: opd-lab/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_<TS>/global_step_50/actor/huggingface/
#   (We auto-pick the latest run by mtime.)
# - Same 200 stratified IF-train prompts as the OPD sanity check at
#   analysis/data/opd_sanity/selected_prompts.parquet
#
# Pipeline
# --------
# 1. Locate the latest E1 OPD output dir (or accept --ckpt-dir override).
# 2. Run sample_responses.py with --model_tag student_after_opd:
#      vLLM, T=1.0 top_p=0.7 top_k=-1 max_resp=8192 n=32 seed=1234 — IDENTICAL
#      to the base / teacher_if pools so label fractions are commensurable.
# 3. Run label_responses.py to get DeepSeek 3-class labels on all 6 400 rows.
#
# Usage:
#   bash analysis/scripts/opd_e1/run_e2_student_after_opd.sh
#   bash analysis/scripts/opd_e1/run_e2_student_after_opd.sh \
#         --ckpt-dir "${CHECKPOINT_DIR}"
# ==============================================================================
set -euo pipefail

ROOT="${PROJECT_ROOT:-.}"
SANITY_DIR="${ROOT}/analysis/scripts/opd_sanity"
OPD_LAB_OUT="${OPD_OUTPUT_ROOT:-${ROOT}/opd-lab/outputs}"

# --- Args ---
CKPT_DIR=""
STEP="50"  # which OPD step's ckpt to use
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        -*) echo "unknown flag: $1"; exit 1 ;;
        *) shift ;;
    esac
done

# --- Locate ckpt ---
if [[ -z "${CKPT_DIR}" ]]; then
    # Pick the latest run dir matching the E1 EXP_NAME prefix.
    LATEST_RUN="$(ls -dt ${OPD_LAB_OUT}/baseline_opd_topk_reverse_kl_k16_tch_huggingface_* 2>/dev/null | head -1)"
    if [[ -z "${LATEST_RUN}" ]]; then
        echo "[run_e2] ERROR: no E1 run dir found under ${OPD_LAB_OUT}/" >&2
        echo "[run_e2]        pass --ckpt-dir manually if your run was renamed." >&2
        exit 1
    fi
    CKPT_DIR="${LATEST_RUN}/global_step_${STEP}/actor/huggingface"
fi
if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
    echo "[run_e2] ERROR: no config.json at ${CKPT_DIR}" >&2
    echo "[run_e2]        confirm E1 finished and the step ${STEP} ckpt was saved." >&2
    exit 1
fi
echo "[run_e2] ckpt = ${CKPT_DIR}"

# --- Optionally activate a caller-selected Conda environment. ---
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

# --- Load DEEPSEEK_API_KEY from ${ROOT}/.env if not already in env ---
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${ROOT}/.env" ]]; then
    set -a; source "${ROOT}/.env"; set +a
fi

# --- 1. Rollouts ---
ROLLOUT_OUT="${ROOT}/rollout/opd_sanity/student_after_opd"
ROLLOUT_JSONL="${ROLLOUT_OUT}/responses.jsonl"
if [[ -f "${ROLLOUT_JSONL}" ]]; then
    echo "[run_e2] ${ROLLOUT_JSONL} exists — skipping rollouts (delete to re-run)"
else
    echo "[run_e2] sampling rollouts from ${CKPT_DIR}..."
    cd "${ROOT}"
    python3 "${SANITY_DIR}/sample_responses.py" \
        --model_tag student_after_opd \
        --model_path "${CKPT_DIR}" \
        --n 32 \
        --temperature 1.0 \
        --top_p 0.7 \
        --top_k -1 \
        --response_length 8192 \
        --tp 4 \
        --gpu_mem_util 0.90 \
        --seed 1234
fi

# --- 2. DeepSeek 3-class labels ---
LABEL_OUT="${ROOT}/analysis/data/opd_sanity/responses_labelled_student_after_opd.parquet"
if [[ -f "${LABEL_OUT}" ]]; then
    echo "[run_e2] ${LABEL_OUT} exists — skipping labels (delete to re-run)"
else
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo "[run_e2] WARNING: DEEPSEEK_API_KEY not set — labelling will fail. " >&2
        echo "[run_e2]          export DEEPSEEK_API_KEY=... and re-run this step." >&2
        exit 1
    fi
    echo "[run_e2] labelling 6 400 student_after_opd rows via DeepSeek..."
    python3 "${SANITY_DIR}/label_responses.py" \
        --model_tag student_after_opd \
        --workers 8
fi

echo "[run_e2] done."
echo "[run_e2] outputs:"
echo "  - ${ROLLOUT_JSONL}"
echo "  - ${LABEL_OUT}"
