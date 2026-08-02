#!/bin/bash
# WAVE 1.5 RL-anchored cross-forward — for B.1 helpful/harmful divergence.
#
# Anchor on each RL model's OWN rollout (M_q3 for Math-RLVR, I_q3 for IF-RLVR,
# same for q25m), then forward base through it to get B's distribution at the
# same prefix. Combined with WAVE 1's M-on-M / I-on-I parquets, this gives
# same-prefix JS(B, RL) per token, conditioned on the RL rollout's own reward.
#
# 6 jobs total:
#   B_q3 over M_q3-rollouts × {aime}                — for B.1(a) Math helpful
#   B_q3 over I_q3-rollouts × {aime, ifeval}        — for B.1(b)/(c) IF harmful
#   B_q25m over M_q25m-rollouts × {aime}            — q25m helpful
#   B_q25m over I_q25m-rollouts × {aime, ifeval}    — q25m harmful + IFEval
#
# Tag scheme: {B_lineage}_on_{rl_lineage} so output files are
#     analysis/data/wave1_5_rl_anchored/{TAG}__{ds}.parquet
# (e.g. B_q3_on_M_q3__aime.parquet).
#
# Reuses analysis/scripts/a1b_wave1_logprob.py — already model-decoupled.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave1_5_rla_logs
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave1_5_rl_anchored
N_GPUS=${N_GPUS:-6}
GPU_OFFSET=${GPU_OFFSET:-0}

mkdir -p "$LOG_DIR" "$OUT_DIR"
rm -f "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro

# RL-anchor rollouts (the trajectory we read responses from)
M_Q3_AIME=$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/220.jsonl
I_Q3_AIME_IF=$ROLL_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/100.jsonl
M_Q25M_AIME=$ROLL_BASE/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/480.jsonl
I_Q25M_AIME_IF=$ROLL_BASE/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/380.jsonl

# Forwarding model (always the lineage's base)
B_Q3=${PROJECT_ROOT:-.}/models/Qwen3-8B-Base
B_Q25M=${PROJECT_ROOT:-.}/models/Qwen2.5-Math-7B

cat > "$QUEUE" <<EOF
B_q3_on_M_q3|$B_Q3|aime|$M_Q3_AIME|32
B_q3_on_I_q3|$B_Q3|aime|$I_Q3_AIME_IF|32
B_q3_on_I_q3|$B_Q3|ifeval|$I_Q3_AIME_IF|8
B_q25m_on_M_q25m|$B_Q25M|aime|$M_Q25M_AIME|32
B_q25m_on_I_q25m|$B_Q25M|aime|$I_Q25M_AIME_IF|32
B_q25m_on_I_q25m|$B_Q25M|ifeval|$I_Q25M_AIME_IF|8
EOF

N_JOBS=$(wc -l < "$QUEUE")
echo "[wave1.5_rla] queue: $N_JOBS jobs across $N_GPUS GPU workers (offset=$GPU_OFFSET)"
echo "[wave1.5_rla] logs: $LOG_DIR  output: $OUT_DIR"

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

for k in $(seq 0 $((N_GPUS - 1))); do
    GPU=$((GPU_OFFSET + k))
    setsid nohup bash -c "$worker_body" wave1_5_rla_gpu$GPU \
        "$GPU" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave1.5_rla] $N_GPUS workers launched on GPUs $GPU_OFFSET..$((GPU_OFFSET+N_GPUS-1))"
echo "[wave1.5_rla] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
