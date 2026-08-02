#!/bin/bash
# Math-7.5k support probe sweep — measure unfiltered (all-wrong, all-correct,
# mixed) group-rate trajectories to support F4 (mode lock-in vs all-wrong
# dominance).
#
# Coverage (per user decision): focus on b4r1 + b1r1-Q3-8B early window +
# b2r1-Q3-8B endpoint. ~12 ckpts.
#
# Each (run, step) cell produces:
#   rollout/math_support_probe/${exp}/step_${step}.jsonl  (128*16=2048 rows)
#   eval_results/${exp}_math_support_probe/step_${step}.json (with all_wrong_rate etc.)
#
# Sampling matches DAPO training: n=16, T=1.0, top_p=0.7, top_k=-1,
# max_resp=8192, max_prompt=2048, seed=1234.
#
# 8 GPUs, TP=4. Sequential over (run, step) — vLLM owns the GPUs at a time.

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

# (exp_name, step, model_path)
# Step 0 always = base model.
JOBS=(
  # b4r1 starting policy + 1 train ckpt — the scenario that motivates the sweep.
  "b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20|0|${ROOT_CKPT}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"
  "b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20|20|${ROOT_CKPT}/b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20/global_step_20/actor/huggingface"

  # b1r1 Q3-8B early window — the contrast case (math-RLVR from cold base
  # successfully reaches DRI mode and shrinks all-wrong rate).
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|0|${ROOT}/models/Qwen3-8B-Base"
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|20|${ROOT_CKPT}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_20/actor/huggingface"
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|40|${ROOT_CKPT}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_40/actor/huggingface"
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|60|${ROOT_CKPT}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_60/actor/huggingface"
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|80|${ROOT_CKPT}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_80/actor/huggingface"
  "b1r1_Qwen3-8B-Base_math7.5k_local_H20|100|${ROOT_CKPT}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_100/actor/huggingface"

  # b2r1 Q3-8B step 100 — equals b4r1 starting policy, but logged under its own
  # exp name to make trajectory plots cleaner. (Same model file as b4r1 step 0
  # above; we still re-evaluate so output JSON files line up with run names.)
  "b2r1_Qwen3-8B-Base_IFTrain_local_H20|0|${ROOT}/models/Qwen3-8B-Base"
  "b2r1_Qwen3-8B-Base_IFTrain_local_H20|20|${ROOT_CKPT}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_20/actor/huggingface"
  "b2r1_Qwen3-8B-Base_IFTrain_local_H20|60|${ROOT_CKPT}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_60/actor/huggingface"
  "b2r1_Qwen3-8B-Base_IFTrain_local_H20|100|${ROOT_CKPT}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"
)

run_one() {
    local exp=$1
    local step=$2
    local model_path=$3
    local out_rollout=${ROOT_OUT}/${exp}/step_${step}.jsonl
    local out_json=${ROOT_AGG}/${exp}_math_support_probe/step_${step}.json
    local log=${LOG_DIR}/${exp}_step_${step}.log

    if [[ -f "$out_json" ]]; then
        echo "[probe] ${exp} step ${step}: out_json exists, skip"
        return
    fi

    if ! ls "$model_path"/*.safetensors >/dev/null 2>&1; then
        echo "[probe] ${exp} step ${step}: no safetensors at $model_path, skip"
        return
    fi

    mkdir -p "$(dirname "$out_rollout")" "$(dirname "$out_json")"
    echo "[probe] ${exp} step ${step} <- ${model_path}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$SCRIPT" \
        --model_path "$model_path" \
        --data_path "$DATA" \
        --out_json "$out_json" \
        --out_rollout_jsonl "$out_rollout" \
        --step "$step" \
        --run_name "$exp" \
        --n 16 \
        --temperature 1.0 \
        --top_p 0.7 \
        --top_k -1 \
        --max_prompt_length 2048 \
        --response_length 8192 \
        --tp 4 \
        --gpu_mem_util 0.90 \
        --seed 1234 \
        > "$log" 2>&1 || { echo "  [ERR] see $log"; tail -30 "$log"; return; }
    grep -E "^\[probe\] (all_wrong|wrote)" "$log" | sed "s/^/  /"
}

for entry in "${JOBS[@]}"; do
    IFS='|' read -r exp step model_path <<< "$entry"
    echo "=========================="
    echo "[probe] ${exp} / step ${step}"
    echo "=========================="
    run_one "$exp" "$step" "$model_path"
done

echo "[probe] all done"
