#!/bin/bash
# Offline IFBench n=32 evaluation for both b2r1 experiments.
#
# For each experiment:
#   step 0  -> use the corresponding base model
#   step N  -> use checkpoints/.../global_step_${N}/actor/huggingface
#
# Outputs:
#   ${PROJECT_ROOT:-.}/rollout/val_rollout_ifbench/${exp}/${step}.jsonl
#   ${PROJECT_ROOT:-.}/eval_results/${exp}_ifbench_pass32_L8192/step_${step}.json
#
# Sampling params match the b2r1 training-time val (n=32, T=1.0, top_p=0.7,
# top_k=-1, max_resp=8192, max_prompt=2048).
#
# 8 GPUs, TP=4 (matches the proven eval_pipeline.sh template).
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
ROOT_CKPT=${ROOT}/checkpoints/verl_exp/DAPO_sh_repro
ROOT_OUT=${ROOT}/rollout/val_rollout_ifbench
ROOT_AGG=${ROOT}/eval_results
DATA=${ROOT}/data/ifbench/test.parquet
SCRIPT=${ROOT}/scripts_eval/eval_one_ifbench.py
LOG_DIR=${ROOT}/eval_results/logs/ifbench
mkdir -p "$LOG_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

export VLLM_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=true

# Two experiments (exp_name|base_model_path).
EXPS=(
  "b2r1_Qwen3-8B-Base_IFTrain_local_H20|${ROOT}/models/Qwen3-8B-Base"
  "b2r1_Qwen2.5-math-7B_IFTrain_local_H20|${ROOT}/models/Qwen2.5-Math-7B"
)

run_one() {
    local exp=$1
    local step=$2
    local model_path=$3
    local out_rollout=${ROOT_OUT}/${exp}/${step}.jsonl
    local out_json=${ROOT_AGG}/${exp}_ifbench_pass32_L8192/step_${step}.json
    local log=${LOG_DIR}/${exp}_step_${step}.log

    if [[ -f "$out_rollout" ]]; then
        echo "[ifbench] ${exp} step ${step}: rollout jsonl exists, skip"
        return
    fi

    mkdir -p "$(dirname "$out_rollout")" "$(dirname "$out_json")"
    echo "[ifbench] ${exp} step ${step} <- ${model_path}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$SCRIPT" \
        --model_path "$model_path" \
        --data_path "$DATA" \
        --out_json "$out_json" \
        --out_rollout_jsonl "$out_rollout" \
        --step "$step" \
        --n 32 \
        --temperature 1.0 \
        --top_p 0.7 \
        --top_k -1 \
        --max_prompt_length 2048 \
        --response_length 8192 \
        --tp 4 \
        --gpu_mem_util 0.90 \
        --seed 1234 \
        > "$log" 2>&1 || { echo "  [ERR] see $log"; tail -30 "$log"; return; }
    if [[ -f "$out_rollout" ]]; then
        tail -n 2 "$log" | sed "s/^/  /"
        wc -l "$out_rollout" | sed "s/^/  /"
    fi
}

for entry in "${EXPS[@]}"; do
    exp=${entry%%|*}
    base_model=${entry##*|}
    echo "=========================="
    echo "[ifbench] Experiment: ${exp}"
    echo "[ifbench] Base model: ${base_model}"
    echo "=========================="

    # step 0 = base model
    run_one "$exp" 0 "$base_model"

    # iterate global_step_* under this experiment, sorted ascending by step number
    EXP_DIR=${ROOT_CKPT}/${exp}
    if [[ ! -d "$EXP_DIR" ]]; then
        echo "[ifbench] ${exp}: no ckpt dir at $EXP_DIR, skipping"
        continue
    fi
    for step_dir in $(ls -d "$EXP_DIR"/global_step_* 2>/dev/null | \
                       awk -F'global_step_' '{print $NF, $0}' | sort -n | awk '{print $2}'); do
        step=$(basename "$step_dir" | sed 's/global_step_//')
        hf="$step_dir/actor/huggingface"
        if ! ls "$hf"/*.safetensors >/dev/null 2>&1; then
            echo "[ifbench] ${exp} step ${step}: no merged safetensors at $hf, skip"
            continue
        fi
        run_one "$exp" "$step" "$hf"
    done
done

echo "[ifbench] all done"
