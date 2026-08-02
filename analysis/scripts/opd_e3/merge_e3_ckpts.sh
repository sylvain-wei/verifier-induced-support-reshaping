#!/usr/bin/env bash
# merge_e3_ckpts.sh — convert all E3 FSDP ckpts to HF safetensors.
#
# verl saves FSDP shards (model_world_size_8_rank_*.pt). vLLM/HF cannot
# load these directly; verl ships scripts/legacy_model_merger.py which
# stitches them into safetensors files alongside the existing config.json.
#
# This script walks the 4 E3 output dirs and for every global_step_{25,50,
# 75,100}/actor that doesn't already have safetensors, runs the merger.
# Idempotent: skips ckpts that already have weights.
#
# Each merge takes ~2 min on a single CPU. Total: 16 ckpts × 2 min ≈ 35 min.
#
# Usage:
#   bash analysis/scripts/opd_e3/merge_e3_ckpts.sh
#   # detached:
#   setsid nohup bash analysis/scripts/opd_e3/merge_e3_ckpts.sh \
#       < /dev/null > analysis/scripts/opd_e3/logs/merge_e3.log 2>&1 & disown
set -euo pipefail

ROOT="${PROJECT_ROOT:-.}"
OPD_LAB="${OPD_LAB_ROOT:-${ROOT}/opd-lab}"
VERL_MERGER="${ROOT}/verl/scripts/legacy_model_merger.py"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

if [[ ! -f "${VERL_MERGER}" ]]; then
    echo "ERROR: verl merger not found at ${VERL_MERGER}" >&2
    exit 1
fi

# E3 output directories (whitelist)
declare -A E3_DIRS
E3_DIRS["20260522_141515"]="step20"
E3_DIRS["20260523_030754"]="step40"
E3_DIRS["20260523_055520"]="step60"
E3_DIRS["20260523_090845"]="step80"

mkdir -p "${ROOT}/analysis/scripts/opd_e3/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/analysis/scripts/opd_e3/logs/merge_e3_${TS}.log"

echo "[merge_e3] starting; log → ${LOG}"
echo "[merge_e3] verl_merger=${VERL_MERGER}"

merge_one() {
    local stamp="$1"
    local step="$2"
    local label="${E3_DIRS[$stamp]}"
    local actor_dir="${OPD_LAB}/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_${stamp}/global_step_${step}/actor"
    local hf_dir="${actor_dir}/huggingface"

    if ! ls "${hf_dir}"/*.safetensors >/dev/null 2>&1; then :; else
        echo "[merge_e3] skip ${label} step ${step}: safetensors already present" | tee -a "${LOG}"
        return 0
    fi

    if [[ ! -d "${actor_dir}" ]]; then
        echo "[merge_e3] skip ${label} step ${step}: actor dir missing (${actor_dir})" | tee -a "${LOG}"
        return 0
    fi

    local n_shards
    n_shards=$(ls "${actor_dir}"/model_world_size_8_rank_*.pt 2>/dev/null | wc -l)
    if [[ "${n_shards}" -ne 8 ]]; then
        echo "[merge_e3] skip ${label} step ${step}: expected 8 FSDP shards, got ${n_shards}" | tee -a "${LOG}"
        return 0
    fi

    echo "" | tee -a "${LOG}"
    echo "=== merging ${label} step ${step} ===" | tee -a "${LOG}"
    echo "  src: ${actor_dir}" | tee -a "${LOG}"
    echo "  dst: ${hf_dir}" | tee -a "${LOG}"
    local t0=$SECONDS
    if python3 "${VERL_MERGER}" merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${hf_dir}" 2>&1 | tee -a "${LOG}"; then
        local elapsed=$((SECONDS - t0))
        echo "[merge_e3] ✓ ${label} step ${step} merged in ${elapsed}s" | tee -a "${LOG}"
    else
        echo "[merge_e3] ✗ ${label} step ${step} MERGE FAILED" | tee -a "${LOG}"
        return 1
    fi
}

# Sort by run label then step for predictable output (step20 → step40 → step60 → step80)
for label_target in step20 step40 step60 step80; do
    for stamp in "${!E3_DIRS[@]}"; do
        if [[ "${E3_DIRS[$stamp]}" == "${label_target}" ]]; then
            for step in 25 50 75 100; do
                merge_one "${stamp}" "${step}" || true
            done
        fi
    done
done

echo "" | tee -a "${LOG}"
echo "[merge_e3] done." | tee -a "${LOG}"
