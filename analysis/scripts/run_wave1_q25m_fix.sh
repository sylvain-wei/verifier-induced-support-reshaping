#!/bin/bash
# WAVE 1 q25m envelope fix re-run.
#
# After fixing _strip_chat_envelope to also handle the
# `system\n<sys>\nuser\n...` prefix Qwen2.5-Math-7B rollouts carry, re-run
# all 6 q25m jobs (B/M/I × {AIME, IFEval}) so prompt_ids are canonical
# across q3 / q25m. Uses 6 of 8 available GPUs.

set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave1_logs_q25m_fix
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave1
N_GPUS=${N_GPUS:-6}

mkdir -p "$LOG_DIR"
rm -f "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro

cat > "$QUEUE" <<EOF
B_q25m|${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B|aime|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl|32
B_q25m|${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B|ifeval|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl|8
M_q25m|$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface|aime|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/480.jsonl|32
M_q25m|$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface|ifeval|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/480.jsonl|8
I_q25m|$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface|aime|$ROLL_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/380.jsonl|32
I_q25m|$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface|ifeval|$ROLL_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/380.jsonl|8
EOF

# Pin to GPUs 2..7 (avoid 0,1 which last carried the longest jobs to
# minimise contention, though all GPUs are free if the orchestrator already
# finished). The pool just rotates over $N_GPUS workers regardless.
GPU_OFFSET=${GPU_OFFSET:-2}

N_JOBS=$(wc -l < "$QUEUE")
echo "[orchestrator] q25m fix queue: $N_JOBS jobs across $N_GPUS GPU workers (offset=$GPU_OFFSET)"

worker_body='
GPU=$1
QUEUE=$2
LOG_DIR=$3
SCRIPT=$4
OUT_DIR=$5

LOCK=$LOG_DIR/queue.lock
: > "$LOG_DIR/gpu${GPU}.log"

while :; do
    JOB=$( ( flock -x 9 ;
        if [ ! -s "$QUEUE" ]; then
            echo ""
        else
            head -n1 "$QUEUE"
            tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
        fi
    ) 9>"$LOCK" )
    if [ -z "$JOB" ]; then
        echo "[gpu$GPU] queue empty, exiting." >> "$LOG_DIR/gpu${GPU}.log"
        break
    fi
    IFS="|" read -r TAG MODEL DS ROLL SPP <<< "$JOB"
    NAME="${TAG}__${DS}"
    echo "[gpu$GPU] starting $NAME (force overwrite)" >> "$LOG_DIR/gpu${GPU}.log"
    JOB_LOG=$LOG_DIR/${NAME}.log

    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --model_path "$MODEL" \
        --model_tag "$TAG" \
        --rollout_jsonl "$ROLL" \
        --dataset "$DS" \
        --samples_per_prompt "$SPP" \
        --save_topk_logits --K 64 \
        --gpu_id 0 \
        --force \
        --out_dir "$OUT_DIR" \
        > "$JOB_LOG" 2>&1
    EXIT=$?
    echo "$EXIT" > "$LOG_DIR/${NAME}.exitcode"
    if [ "$EXIT" -eq 0 ]; then
        echo "[gpu$GPU]   PASS  $NAME" >> "$LOG_DIR/gpu${GPU}.log"
    else
        echo "[gpu$GPU]   FAIL  $NAME  (exit=$EXIT)" >> "$LOG_DIR/gpu${GPU}.log"
    fi
done
echo "[gpu$GPU] worker exiting." >> "$LOG_DIR/gpu${GPU}.log"
'

for k in $(seq 0 $((N_GPUS - 1))); do
    GPU=$((GPU_OFFSET + k))
    setsid nohup bash -c "$worker_body" wave1_q25m_fix_gpu$GPU \
        "$GPU" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[orchestrator] $N_GPUS workers launched on GPUs $GPU_OFFSET..$((GPU_OFFSET+N_GPUS-1))"
echo "[orchestrator] Tail any worker log: tail -f $LOG_DIR/gpu${GPU_OFFSET}.log"
