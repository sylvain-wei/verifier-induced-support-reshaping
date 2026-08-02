#!/bin/bash
# WAVE 1.5 — base-anchored cross-model forward.
#
# Computes the *RL model's* top-K logits + per-token logp on B's rollout
# trajectories, so that we can later pair them with WAVE 1's B-on-B-rollout
# parquets to compute same-prefix JS(B, RL) at every position.
#
# 10 jobs total:
#   q3 lineage   : {M_q3, I_q3}   on B_q3 rollouts × {aime, ifeval, ifbench}
#   q25m lineage : {M_q25m, I_q25m} on B_q25m rollouts × {aime, ifeval}
#
# `model_tag` is suffixed `_on_B` so output parquets sit alongside WAVE 1
# without name collision:
#   analysis/data/wave1_5/{TAG}_on_B__{DS}.parquet
#
# Reuses analysis/scripts/a1b_wave1_logprob.py — that script is decoupled
# (model_path = who forwards, rollout_jsonl = whose response prefix we read).
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave1_5_logs
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave1_5
N_GPUS=${N_GPUS:-8}

mkdir -p "$LOG_DIR" "$OUT_DIR"
rm -f "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro
ROLL_IFB=${PROJECT_ROOT:-.}/rollout/val_rollout_ifbench

# B-anchor rollouts
B_Q3_AIME_IF=$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl
B_Q3_IFB=$ROLL_IFB/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl
B_Q25M_AIME_IF=$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl

# RL ckpts
M_Q3=$CKPT_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface
I_Q3=$CKPT_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface
M_Q25M=$CKPT_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface
I_Q25M=$CKPT_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface

cat > "$QUEUE" <<EOF
M_q3_on_B|$M_Q3|aime|$B_Q3_AIME_IF|32
I_q3_on_B|$I_Q3|aime|$B_Q3_AIME_IF|32
M_q3_on_B|$M_Q3|ifeval|$B_Q3_AIME_IF|8
I_q3_on_B|$I_Q3|ifeval|$B_Q3_AIME_IF|8
M_q3_on_B|$M_Q3|ifbench|$B_Q3_IFB|8
I_q3_on_B|$I_Q3|ifbench|$B_Q3_IFB|8
M_q25m_on_B|$M_Q25M|aime|$B_Q25M_AIME_IF|32
I_q25m_on_B|$I_Q25M|aime|$B_Q25M_AIME_IF|32
M_q25m_on_B|$M_Q25M|ifeval|$B_Q25M_AIME_IF|8
I_q25m_on_B|$I_Q25M|ifeval|$B_Q25M_AIME_IF|8
EOF

N_JOBS=$(wc -l < "$QUEUE")
echo "[wave1.5] queue: $N_JOBS jobs across $N_GPUS GPU workers"
echo "[wave1.5] logs: $LOG_DIR  output: $OUT_DIR"

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
    echo "[gpu$GPU] starting $NAME" >> "$LOG_DIR/gpu${GPU}.log"
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
        echo "[gpu$GPU]   FAIL  $NAME  (exit=$EXIT)" >> "$LOG_DIR/gpu${GPU}.log"
    fi
done
echo "[gpu$GPU] worker exiting." >> "$LOG_DIR/gpu${GPU}.log"
'

for i in $(seq 0 $((N_GPUS - 1))); do
    setsid nohup bash -c "$worker_body" wave1_5_gpu$i \
        "$i" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave1.5] $N_GPUS workers launched (setsid+nohup+disown)"
echo "[wave1.5] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
