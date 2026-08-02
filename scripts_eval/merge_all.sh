#!/bin/bash
# Merge every FSDP actor checkpoint under the experiment dir into HuggingFace format,
# writing each into global_step_X/actor/huggingface/ (next to existing tokenizer/config files).
set -euo pipefail

EXP_DIR=${PROJECT_ROOT:-.}/checkpoints/verl_exp/RLVR_MATH_VIF_BALANCE/Qwen3-8B-Base_Base2Math_AddDynaSampl
LOG_DIR=${PROJECT_ROOT:-.}/eval_results/logs
mkdir -p "$LOG_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

for step_dir in $(ls -d "$EXP_DIR"/global_step_* | sort -t_ -k3 -n); do
    step=$(basename "$step_dir" | sed 's/global_step_//')
    actor_dir="$step_dir/actor"
    target="$actor_dir/huggingface"
    if [[ ! -d "$actor_dir" ]]; then
        echo "[merge] skip $step: no actor dir"
        continue
    fi
    # skip if already merged (safetensors present)
    if ls "$target"/*.safetensors >/dev/null 2>&1; then
        echo "[merge] step $step: already merged, skip"
        continue
    fi
    echo "[merge] step $step -> $target"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target" \
        2>&1 | tail -n 5 | sed "s/^/  /"
done

echo "[merge] all done"
