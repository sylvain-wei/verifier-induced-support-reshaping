#!/bin/bash
# Loop over base + all merged HF checkpoints, run eval_one.py for each, collect summaries.
set -euo pipefail

EXP_DIR=${PROJECT_ROOT:-.}/checkpoints/verl_exp/RLVR_MATH_VIF_BALANCE/Qwen3-8B-Base_Base2Math_AddDynaSampl
BASE_MODEL=${PROJECT_ROOT:-.}/models/Qwen3-8B-Base
DATA=${PROJECT_ROOT:-.}/data/aime24/test.parquet
OUT_DIR=${PROJECT_ROOT:-.}/eval_results/aime24_pass32_L12288
LOG_DIR=${PROJECT_ROOT:-.}/eval_results/logs
SCRIPT=${PROJECT_ROOT:-.}/scripts_eval/eval_one.py

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

run_one() {
    local step=$1
    local model_path=$2
    local out="$OUT_DIR/step_${step}.json"
    local log="$LOG_DIR/eval_step_${step}.log"
    if [[ -f "$out" ]]; then
        echo "[eval] step $step: already done ($out)"
        return
    fi
    echo "[eval] step $step <- $model_path"
    python "$SCRIPT" \
        --model_path "$model_path" \
        --data_path "$DATA" \
        --out_json "$out" \
        --n 32 \
        --temperature 1.0 \
        --top_p 0.7 \
        --max_prompt_length 1024 \
        --response_length 12288 \
        --tp 2 \
        --gpu_mem_util 0.85 \
        --seed 1234 \
        > "$log" 2>&1
    tail -n 3 "$log" | sed "s/^/  /"
}

# step 0 = base
run_one 0 "$BASE_MODEL"

# then all merged steps in ascending order
for step_dir in $(ls -d "$EXP_DIR"/global_step_* | sort -t_ -k3 -n); do
    step=$(basename "$step_dir" | sed 's/global_step_//')
    hf="$step_dir/actor/huggingface"
    if ! ls "$hf"/*.safetensors >/dev/null 2>&1; then
        echo "[eval] step $step: no safetensors, skip"
        continue
    fi
    run_one "$step" "$hf"
done

echo "[eval] all done"
