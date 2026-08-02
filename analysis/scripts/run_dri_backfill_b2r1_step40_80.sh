#!/bin/bash
# Backfill DRI rollouts for b2r1 step_40 and step_80 (originally only 0/20/60/100
# were in math_support_dri_per_ckpt.csv).
#
# Runs the two cells in parallel — each with TP=4 (4 GPUs):
#   step_40 → CUDA_VISIBLE_DEVICES=0,1,2,3
#   step_80 → CUDA_VISIBLE_DEVICES=4,5,6,7
#
# Sampling matches the original sweep (eval_pipeline_math_support.sh):
#   n=16, T=1.0, top_p=0.7, top_k=-1, max_resp=8192, max_prompt=2048, seed=1234
#
# Outputs:
#   rollout/math_support_probe/b2r1_Qwen3-8B-Base_IFTrain_local_H20/step_{40,80}.jsonl
#   eval_results/b2r1_Qwen3-8B-Base_IFTrain_local_H20_math_support_probe/step_{40,80}.json
#
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
ROOT_CKPT=${ROOT}/checkpoints/verl_exp/DAPO_sh_repro
ROOT_OUT=${ROOT}/rollout/math_support_probe
ROOT_AGG=${ROOT}/eval_results
DATA=${ROOT}/analysis/data/math_support_probe_128.parquet
SCRIPT=${ROOT}/analysis/scripts/eval_one_math_support.py
LOG_DIR=${ROOT}/eval_results/logs/math_support_probe
mkdir -p "$LOG_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi
export VLLM_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=true

EXP=b2r1_Qwen3-8B-Base_IFTrain_local_H20

run_one() {
    local step=$1
    local gpus=$2
    local model_path=${ROOT_CKPT}/${EXP}/global_step_${step}/actor/huggingface
    local out_rollout=${ROOT_OUT}/${EXP}/step_${step}.jsonl
    local out_json=${ROOT_AGG}/${EXP}_math_support_probe/step_${step}.json
    local log=${LOG_DIR}/${EXP}_step_${step}_backfill.log

    if [[ -f "$out_json" ]]; then
        echo "[backfill] step ${step}: out_json exists, skip"
        return
    fi
    if ! ls "$model_path"/*.safetensors >/dev/null 2>&1; then
        echo "[backfill] step ${step}: no safetensors at $model_path, skip"
        return
    fi

    mkdir -p "$(dirname "$out_rollout")" "$(dirname "$out_json")"
    echo "[backfill] step ${step} on GPUs ${gpus} <- ${model_path}"
    CUDA_VISIBLE_DEVICES=${gpus} python "$SCRIPT" \
        --model_path "$model_path" \
        --data_path "$DATA" \
        --out_json "$out_json" \
        --out_rollout_jsonl "$out_rollout" \
        --step "$step" \
        --run_name "$EXP" \
        --n 16 \
        --temperature 1.0 \
        --top_p 0.7 \
        --top_k -1 \
        --max_prompt_length 2048 \
        --response_length 8192 \
        --tp 4 \
        --gpu_mem_util 0.90 \
        --seed 1234 \
        > "$log" 2>&1
    echo "[backfill] step ${step} done"
    grep -E "^\[probe\] (all_wrong|wrote)" "$log" | sed "s/^/  /"
}

# Launch both in parallel.
run_one 40 "0,1,2,3" &
PID40=$!
run_one 80 "4,5,6,7" &
PID80=$!

echo "[backfill] launched step_40 (pid $PID40) + step_80 (pid $PID80) in parallel"
wait $PID40 || echo "[backfill] step_40 FAILED"
wait $PID80 || echo "[backfill] step_80 FAILED"

echo "[backfill] all requested rollouts are complete"
