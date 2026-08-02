#!/bin/bash
# Targeted WAVE 1 / 1.5 top-K forward for Qwen2.5-Math IFBench.
#
# Produces the missing base-anchored inputs needed by D.2 and D.5:
#   analysis/data/wave1/B_q25m__ifbench.parquet
#   analysis/data/wave1_5/M_q25m_on_B__ifbench.parquet
#   analysis/data/wave1_5/I_q25m_on_B__ifbench.parquet

set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave1_q25m_ifbench_logs
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
N_GPUS=${N_GPUS:-3}
GPU_OFFSET=${GPU_OFFSET:-0}

mkdir -p "$LOG_DIR" analysis/data/wave1 analysis/data/wave1_5
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_IFB=${PROJECT_ROOT:-.}/rollout/val_rollout_ifbench

B_Q25M=${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B
M_Q25M=$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface
I_Q25M=$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface
B_Q25M_IFB=$ROLL_IFB/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl

JOBS=(
    "B_q25m|$B_Q25M|analysis/data/wave1"
    "M_q25m_on_B|$M_Q25M|analysis/data/wave1_5"
    "I_q25m_on_B|$I_Q25M|analysis/data/wave1_5"
)

echo "[wave1-q25m-ifbench] launching ${#JOBS[@]} jobs using up to $N_GPUS GPU(s)"
echo "[wave1-q25m-ifbench] logs: $LOG_DIR"

worker_body='
JOB="$1"
GPU="$2"
LOG_DIR="$3"
SCRIPT="$4"
ROLL="$5"
IFS="|" read -r TAG MODEL OUT_DIR <<< "$JOB"
NAME="${TAG}__ifbench"
JOB_LOG="$LOG_DIR/${NAME}.log"
echo "[gpu$GPU] starting $NAME" > "$JOB_LOG"
CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
    --model_path "$MODEL" \
    --model_tag "$TAG" \
    --rollout_jsonl "$ROLL" \
    --dataset ifbench \
    --samples_per_prompt 8 \
    --save_topk_logits --K 64 \
    --gpu_id 0 \
    --out_dir "$OUT_DIR" \
    >> "$JOB_LOG" 2>&1
EXIT=$?
echo "$EXIT" > "$LOG_DIR/${NAME}.exitcode"
if [ "$EXIT" -eq 0 ]; then
    echo "[gpu$GPU] PASS $NAME" >> "$JOB_LOG"
else
    echo "[gpu$GPU] FAIL $NAME exit=$EXIT" >> "$JOB_LOG"
fi
'

for idx in "${!JOBS[@]}"; do
    gpu_slot=$((idx % N_GPUS))
    GPU=$((GPU_OFFSET + gpu_slot))
    setsid nohup bash -c "$worker_body" "wave1_q25m_ifbench_$idx" \
        "${JOBS[$idx]}" "$GPU" "$LOG_DIR" "$SCRIPT" "$B_Q25M_IFB" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave1-q25m-ifbench] launched. watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
