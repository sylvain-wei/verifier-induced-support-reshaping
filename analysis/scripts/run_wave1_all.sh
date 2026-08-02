#!/bin/bash
# WAVE 1 orchestrator — 14 (model, dataset) logprob jobs across 8 GPUs.
#
# Scheduling model: 8 GPU workers each loop on a shared job queue (jobs.txt)
# protected by flock(1). Each job line has the form:
#   model_tag|model_path|dataset|rollout_jsonl|samples_per_prompt
#
# Survival: launcher uses `setsid nohup ... </dev/null & disown` per worker so
# the workers outlive the codebuddy bash tool's child cleanup (see
# memory/feedback_background_jobs.md).

set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave1_logs
QUEUE=$LOG_DIR/jobs.txt
DONE_DIR=$LOG_DIR/done
WORKER_LOG=$LOG_DIR/worker
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave1
N_GPUS=${N_GPUS:-8}

mkdir -p "$LOG_DIR" "$DONE_DIR" "$OUT_DIR"
rm -f "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro
ROLL_IFB=${PROJECT_ROOT:-.}/rollout/val_rollout_ifbench

# ----- Build job queue (14 jobs) ---------------------------------------------
cat > "$QUEUE" <<EOF
B_q3|${PROJECT_ROOT:-.}/models/Qwen3-8B-Base|aime|$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl|32
B_q3|${PROJECT_ROOT:-.}/models/Qwen3-8B-Base|ifeval|$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl|8
B_q3|${PROJECT_ROOT:-.}/models/Qwen3-8B-Base|ifbench|$ROLL_IFB/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl|8
M_q3|$CKPT_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface|aime|$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/220.jsonl|32
M_q3|$CKPT_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface|ifeval|$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/220.jsonl|8
I_q3|$CKPT_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface|aime|$ROLL_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/100.jsonl|32
I_q3|$CKPT_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface|ifeval|$ROLL_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/100.jsonl|8
I_q3|$CKPT_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface|ifbench|$ROLL_IFB/b2r1_Qwen3-8B-Base_IFTrain_local_H20/100.jsonl|8
B_q25m|${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B|aime|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl|32
B_q25m|${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B|ifeval|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl|8
M_q25m|$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface|aime|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/480.jsonl|32
M_q25m|$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface|ifeval|$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/480.jsonl|8
I_q25m|$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface|aime|$ROLL_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/380.jsonl|32
I_q25m|$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface|ifeval|$ROLL_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/380.jsonl|8
EOF

N_JOBS=$(wc -l < "$QUEUE")
echo "[orchestrator] Queue prepared: $N_JOBS jobs across $N_GPUS GPU workers."
echo "[orchestrator] Logs: $LOG_DIR/  Output: $OUT_DIR/"

# ----- Worker function (runs in subshell per GPU) ----------------------------
# Atomically pop the next job, run it, repeat until queue is empty.
worker_body='
GPU=$1
QUEUE=$2
LOG_DIR=$3
SCRIPT=$4
OUT_DIR=$5

LOCK=$LOG_DIR/queue.lock
: > "$LOG_DIR/gpu${GPU}.log"

while :; do
    # Atomically pop one line.
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
    echo "[gpu$GPU] starting $NAME  (model=$MODEL  spp=$SPP)" \
        >> "$LOG_DIR/gpu${GPU}.log"
    JOB_LOG=$LOG_DIR/${NAME}.log

    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --model_path "$MODEL" \
        --model_tag "$TAG" \
        --rollout_jsonl "$ROLL" \
        --dataset "$DS" \
        --samples_per_prompt "$SPP" \
        --save_topk_logits --K 64 \
        --gpu_id 0 \
        --out_dir "$OUT_DIR" \
        > "$JOB_LOG" 2>&1
    EXIT=$?
    echo "$EXIT" > "$LOG_DIR/${NAME}.exitcode"
    if [ "$EXIT" -eq 0 ]; then
        echo "[gpu$GPU]   PASS  $NAME" >> "$LOG_DIR/gpu${GPU}.log"
    else
        echo "[gpu$GPU]   FAIL  $NAME  (exit=$EXIT, see $JOB_LOG)" \
            >> "$LOG_DIR/gpu${GPU}.log"
    fi
done
echo "[gpu$GPU] worker exiting." >> "$LOG_DIR/gpu${GPU}.log"
'

# ----- Launch 8 GPU workers in detached, hangup-immune mode ------------------
# Pass args via bash -c "<body>" _name positional args
for i in $(seq 0 $((N_GPUS - 1))); do
    setsid nohup bash -c "$worker_body" wave1_worker_gpu$i \
        "$i" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[orchestrator] $N_GPUS workers launched (setsid+nohup+disown)."
echo "[orchestrator] Tail any worker log with: tail -f $LOG_DIR/gpu0.log"
echo "[orchestrator] Watch all jobs with:"
echo "                ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l   # done"
echo "                wc -l $QUEUE                                 # remaining"
