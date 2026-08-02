#!/usr/bin/env bash
# eval_if_offline.sh — auto-discover E3 student ckpts and run offline IF eval.
#
# Walks `opd-lab/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_*/`,
# infers the run label from each output dir's WandB run name, and for every
# global_step_{25,50,75,100}/actor/huggingface ckpt runs eval_if_offline.py
# (idempotent — appends to e3_val_metrics.csv only if (run,step,dataset) not
# already there).
#
# Usage:
#   bash analysis/scripts/opd_e3/eval_if_offline.sh                # all
#   RUN_FILTER="step20" bash analysis/scripts/opd_e3/eval_if_offline.sh
#   STEPS="50 100" bash analysis/scripts/opd_e3/eval_if_offline.sh
#
# Detached:
#   setsid nohup bash analysis/scripts/opd_e3/eval_if_offline.sh \
#       < /dev/null > analysis/scripts/opd_e3/logs/eval_if_offline.log 2>&1 & disown
set -euo pipefail

ROOT="${PROJECT_ROOT:-.}"
OPD_LAB="${OPD_LAB_ROOT:-${ROOT}/opd-lab}"
ANALYSIS_DIR="${ROOT}/analysis/scripts/opd_e3"
LOG_DIR="${ANALYSIS_DIR}/logs"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

# Defaults
RUN_FILTER="${RUN_FILTER:-}"            # e.g. step20 — substring match
STEPS="${STEPS:-25 50 75 100}"          # which ckpt steps to evaluate
DATASETS="${DATASETS:-ifeval_test,ifbench_test}"

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_if_offline_${TS}.log"
echo "[eval_if_offline] starting; log → ${LOG_FILE}"
echo "[eval_if_offline] RUN_FILTER='${RUN_FILTER}' STEPS='${STEPS}' DATASETS='${DATASETS}'"

# Whitelist of E3 output directories. Manually curated because the broader
# baseline_opd_topk_reverse_kl_k16_tch_huggingface_* glob also matches:
#   - E1's run (20260519_222832, teacher=b2r1 step_100) — wandb run name
#     "b2r1step100" parses as run_label="step100" but is NOT part of E3
#   - empty leftover dirs from the first E3 attempt that crashed in val
#     (20260522_120605, 20260522_130609, 20260522_185049, 20260522_203556,
#      20260522_222921) — no ckpts saved
# Only these 4 directories correspond to the successful E3 reruns:
declare -A E3_WHITELIST
E3_WHITELIST["20260522_141515"]="step20"   # b2r1step20, 100 OPD steps
E3_WHITELIST["20260523_030754"]="step40"   # b2r1step40, 100 OPD steps (rerun after NCCL fix)
E3_WHITELIST["20260523_055520"]="step60"   # b2r1step60, 100 OPD steps (rerun)
E3_WHITELIST["20260523_090845"]="step80"   # b2r1step80, 100 OPD steps (rerun)

OUTPUT_DIRS=()
for stamp in "${!E3_WHITELIST[@]}"; do
    d="${OPD_LAB}/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_${stamp}"
    if [[ -d "$d" ]]; then
        OUTPUT_DIRS+=("$d")
    fi
done
if [[ ${#OUTPUT_DIRS[@]} -eq 0 ]]; then
    echo "[eval_if_offline] no whitelisted E3 output dirs found" | tee -a "${LOG_FILE}"
    exit 0
fi
echo "[eval_if_offline] discovered ${#OUTPUT_DIRS[@]} E3 dirs"

# Resolve run label from whitelist instead of wandb metadata.
infer_run_label() {
    local out_dir="$1"
    local stamp
    stamp=$(basename "${out_dir}" | sed -E 's/.*_huggingface_//')
    echo "${E3_WHITELIST[${stamp}]:-}"
}

for out_dir in "${OUTPUT_DIRS[@]}"; do
    run_label=$(infer_run_label "${out_dir}" || true)
    if [[ -z "${run_label}" ]]; then
        echo "[eval_if_offline] skip ${out_dir}: cannot infer run label" | tee -a "${LOG_FILE}"
        continue
    fi
    if [[ -n "${RUN_FILTER}" && "${run_label}" != *"${RUN_FILTER}"* ]]; then
        continue
    fi

    for step in ${STEPS}; do
        ckpt="${out_dir}/global_step_${step}/actor/huggingface"
        if [[ ! -f "${ckpt}/config.json" ]]; then
            echo "[eval_if_offline] skip ${run_label} step ${step}: ckpt not found at ${ckpt}" \
                | tee -a "${LOG_FILE}"
            continue
        fi
        echo "" | tee -a "${LOG_FILE}"
        echo "=== ${run_label} step ${step} ===" | tee -a "${LOG_FILE}"
        echo "  ckpt = ${ckpt}" | tee -a "${LOG_FILE}"

        # Each ckpt eval is its own python process: vLLM+CUDA cleanup between
        # ckpts is more reliable that way.
        python3 "${ANALYSIS_DIR}/eval_if_offline.py" \
            --run "${run_label}" \
            --step "${step}" \
            --ckpt "${ckpt}" \
            --datasets "${DATASETS}" \
            2>&1 | tee -a "${LOG_FILE}"
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "[eval_if_offline] done." | tee -a "${LOG_FILE}"
