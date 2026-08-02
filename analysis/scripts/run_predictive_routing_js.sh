#!/bin/bash
# WAVE 5 — Predictive routing-JS @ step_20 forward.
#
# Computes RL-policy top-K logits + per-token logp on B_q3's rollout
# trajectories, where the RL policy is each of 5 controlled runs taken at
# their *step_20* checkpoint:
#
#   1. I_q3_step20_vanilla   (b2r1_Qwen3-8B-Base_IFTrain_local_H20)
#   2. I_q3_step20_S1soft    (b2r1_h1_S1_RL_dri50_local_H20)
#   3. I_q3_step20_S2hard    (b2r1_h1_S2_RL_dri50_local_H20)
#   4. I_q3_step20_negDAI    (b2r1_h7_negctl_DAI_dri50_local_H20)
#   5. I_q3_step20_negRand   (b2r1_h7_negctl_random_dri50_local_H20)
#
# Each run × {aime, ifeval} = 10 jobs total.
#
# Same prefix (B_q3 rollout) so downstream pos1-JS to base is well-defined
# (matches §5.4 / WAVE 1.5 convention).
#
# Output:  analysis/data/wave5_predictive/{TAG}__{DS}.parquet
# Logs:    /tmp/wave5_predictive_logs/
#
# Reuses analysis/scripts/a1b_wave1_logprob.py — decoupled (model_path =
# who forwards, rollout_jsonl = whose response prefix we read).
#
# 8-GPU concurrent worker pool with flock-based job queue, mirroring
# run_wave1_5_cross.sh.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave5_predictive_logs
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave5_predictive
N_GPUS=${N_GPUS:-8}

mkdir -p "$LOG_DIR" "$OUT_DIR"
rm -f "$QUEUE" "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro

# B-anchor rollout (pure Qwen3-8B-Base, no RL)
B_Q3_ROLL=$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl

# 5 RL runs at step_20 (all share global_step_20/actor/huggingface layout)
VANILLA=$CKPT_BASE/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_20/actor/huggingface
S1=$CKPT_BASE/b2r1_h1_S1_RL_dri50_local_H20/global_step_20/actor/huggingface
S2=$CKPT_BASE/b2r1_h1_S2_RL_dri50_local_H20/global_step_20/actor/huggingface
NEGDAI=$CKPT_BASE/b2r1_h7_negctl_DAI_dri50_local_H20/global_step_20/actor/huggingface
NEGRAND=$CKPT_BASE/b2r1_h7_negctl_random_dri50_local_H20/global_step_20/actor/huggingface

# job format: TAG|MODEL|DATASET|ROLLOUT|SAMPLES_PER_PROMPT
# AIME: 32 samples / prompt (matches wave1 default).
# IFEval: 8 samples / prompt — we only need pos1 + early-token JS, no need
# for full 32 sample density; cuts forward time ~4x.
cat > "$QUEUE" <<EOF
I_q3_step20_vanilla|$VANILLA|aime|$B_Q3_ROLL|32
I_q3_step20_vanilla|$VANILLA|ifeval|$B_Q3_ROLL|8
I_q3_step20_S1soft|$S1|aime|$B_Q3_ROLL|32
I_q3_step20_S1soft|$S1|ifeval|$B_Q3_ROLL|8
I_q3_step20_S2hard|$S2|aime|$B_Q3_ROLL|32
I_q3_step20_S2hard|$S2|ifeval|$B_Q3_ROLL|8
I_q3_step20_negDAI|$NEGDAI|aime|$B_Q3_ROLL|32
I_q3_step20_negDAI|$NEGDAI|ifeval|$B_Q3_ROLL|8
I_q3_step20_negRand|$NEGRAND|aime|$B_Q3_ROLL|32
I_q3_step20_negRand|$NEGRAND|ifeval|$B_Q3_ROLL|8
EOF

N_JOBS=$(wc -l < "$QUEUE")
echo "[wave5_predictive] queue: $N_JOBS jobs across $N_GPUS GPU workers"
echo "[wave5_predictive] logs: $LOG_DIR  output: $OUT_DIR"

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
    setsid nohup bash -c "$worker_body" wave5_pred_gpu$i \
        "$i" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave5_predictive] $N_GPUS workers launched (setsid+nohup+disown)"
echo "[wave5_predictive] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
