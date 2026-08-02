#!/bin/bash
# Pipelined: wait for each merged HF ckpt, immediately evaluate it, then move to the next.
# Uses all 8 GPUs with TP=4 (optimal for 8B bf16 with 13k context).
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

# Controls for vLLM
export VLLM_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=true

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
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$SCRIPT" \
        --model_path "$model_path" \
        --data_path "$DATA" \
        --out_json "$out" \
        --n 32 \
        --temperature 1.0 \
        --top_p 0.7 \
        --max_prompt_length 1024 \
        --response_length 12288 \
        --tp 4 \
        --gpu_mem_util 0.90 \
        --seed 1234 \
        > "$log" 2>&1 || { echo "  [ERR] see $log"; tail -30 "$log"; }
    if [[ -f "$out" ]]; then
        tail -n 2 "$log" | sed "s/^/  /"
    fi
}

# Step 0: base model
run_one 0 "$BASE_MODEL"

# For each global_step, if it's not merged yet, wait for it; then eval.
for step_dir in $(ls -d "$EXP_DIR"/global_step_* | while read p; do echo "$(basename $p | sed 's/global_step_//') $p"; done | sort -n | awk '{print $2}'); do
    step=$(basename "$step_dir" | sed 's/global_step_//')
    hf="$step_dir/actor/huggingface"
    out="$OUT_DIR/step_${step}.json"
    if [[ -f "$out" ]]; then
        echo "[eval] step $step: already done"
        continue
    fi
    # wait for safetensors to appear (merge may still be in progress)
    waited=0
    while ! ls "$hf"/*.safetensors >/dev/null 2>&1; do
        if (( waited == 0 )); then
            echo "[eval] step $step: waiting for merge..."
        fi
        sleep 20
        waited=$((waited+20))
        if (( waited > 1800 )); then
            echo "[eval] step $step: gave up waiting after 30 min"
            break
        fi
    done
    if ls "$hf"/*.safetensors >/dev/null 2>&1; then
        run_one "$step" "$hf"
    fi
done

echo "[eval] all done"
