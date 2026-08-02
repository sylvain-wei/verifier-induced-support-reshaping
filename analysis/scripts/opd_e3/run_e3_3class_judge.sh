#!/usr/bin/env bash
# ==============================================================================
# run_e3_3class_judge.sh — E2-style 3-class judge on the 4 E3 step-100 students.
#
# Why this script
# ---------------
# The unified E3 sweet-spot scan (findings_opd_E3_sweetspot_scan.md) showed that
# IFEval transfer is hump-shaped — peaks at b2r1 step_40 teacher (IFEval=0.814),
# not at the latest teacher. That hump could mean either:
#   (a) genuine IF improvement that just decays with more-broken teachers, or
#   (b) shortcut-style verifier compliance that the step_40 teacher transmits
#       more strongly than the others.
# E2 already showed the original (step_100 teacher) student is +14.3 pp shortcut
# vs base. We need the corresponding shortcut share for each of the 4 sweet-spot
# students to disambiguate (a) vs (b).
#
# Pipeline
# --------
# Each E3 student timestamp → step_100 actor/huggingface, sampled identically to
# E2 (200 prompts × 32 rollouts; T=1.0 top_p=0.7 max_resp=8192 seed=1234) and
# labelled by DeepSeek's 3-class judge {contentful, shortcut, fail}.
#
#   E3 student            timestamp           model_tag (rollout dir)
#   ─────────────────     ─────────────────   ───────────────────────────
#   b2r1 step_20 teacher  20260522_141515     student_e3_step20
#   b2r1 step_40 teacher  20260523_030754     student_e3_step40
#   b2r1 step_60 teacher  20260523_055520     student_e3_step60
#   b2r1 step_80 teacher  20260523_090845     student_e3_step80
#
# Outputs
# -------
#   rollout/opd_sanity/student_e3_step{20,40,60,80}/responses.jsonl
#   analysis/data/opd_sanity/responses_labelled_student_e3_step{20,40,60,80}.parquet
#   (existing labels_cache.jsonl appended to)
#
# Sequential GPU usage: each student takes one TP=8 vLLM session. ~20-30 min per
# rollout × 4 students ≈ 1.5-2 h GPU. Then DeepSeek labelling is ~30 min for
# 25.6k label calls @ 8 workers (cached, so re-runs are free).
#
# Usage:
#   bash analysis/scripts/opd_e3/run_e3_3class_judge.sh
# ==============================================================================
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
SANITY_DIR=${ROOT}/analysis/scripts/opd_sanity
OPD_OUT=${OPD_OUTPUT_ROOT:-${ROOT}/opd-lab/outputs}
E3_DIR=${ROOT}/analysis/scripts/opd_e3
LOG_DIR=${E3_DIR}/logs
mkdir -p "$LOG_DIR"

# E3 timestamp → student tag map (matches eval_if_offline.sh whitelist).
declare -A STUDENTS=(
    ["20260522_141515"]="student_e3_step20"
    ["20260523_030754"]="student_e3_step40"
    ["20260523_055520"]="student_e3_step60"
    ["20260523_090845"]="student_e3_step80"
)

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi
export VLLM_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Load DeepSeek key from .env.
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${ROOT}/.env" ]]; then
    set -a; source "${ROOT}/.env"; set +a
fi

cd "$ROOT"

# --- Phase 1: rollouts (sequential, TP=8 each) ---
for ts in "20260522_141515" "20260523_030754" "20260523_055520" "20260523_090845"; do
    tag=${STUDENTS[$ts]}
    ckpt_dir=${OPD_OUT}/baseline_opd_topk_reverse_kl_k16_tch_huggingface_${ts}/global_step_100/actor/huggingface
    out_jsonl=${ROOT}/rollout/opd_sanity/${tag}/responses.jsonl

    if [[ -f "$out_jsonl" ]]; then
        echo "[3class] ${tag}: rollouts already exist (${out_jsonl}), skipping sample"
        continue
    fi

    if ! ls "$ckpt_dir"/*.safetensors >/dev/null 2>&1; then
        echo "[3class] ERROR: no safetensors at $ckpt_dir for $tag" >&2
        exit 1
    fi

    log=${LOG_DIR}/3class_sample_${tag}.log
    echo "[3class] sampling ${tag} from ${ckpt_dir}  (log: ${log})"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 "${SANITY_DIR}/sample_responses.py" \
        --model_tag "${tag}" \
        --model_path "${ckpt_dir}" \
        --n 32 \
        --temperature 1.0 \
        --top_p 0.7 \
        --top_k -1 \
        --response_length 8192 \
        --tp 8 \
        --gpu_mem_util 0.85 \
        --seed 1234 \
        > "$log" 2>&1
    echo "[3class] ${tag}: rollouts done"
done

# --- Phase 2: DeepSeek 3-class labelling (sequential, 8 workers) ---
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[3class] WARNING: DEEPSEEK_API_KEY not set — labelling skipped" >&2
    exit 1
fi
for ts in "20260522_141515" "20260523_030754" "20260523_055520" "20260523_090845"; do
    tag=${STUDENTS[$ts]}
    out_parquet=${ROOT}/analysis/data/opd_sanity/responses_labelled_${tag}.parquet
    if [[ -f "$out_parquet" ]]; then
        echo "[3class] ${tag}: labels parquet already exists, skipping label"
        continue
    fi
    log=${LOG_DIR}/3class_label_${tag}.log
    echo "[3class] labelling ${tag}  (log: ${log})"
    python3 "${SANITY_DIR}/label_responses.py" \
        --model_tag "${tag}" \
        --workers 24 \
        > "$log" 2>&1
    echo "[3class] ${tag}: labels done"
done

echo "[3class] all rollouts and labels are ready"
