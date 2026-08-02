#!/bin/bash
# WAVE 5 — Predictive routing-JS trajectory forward (degraded variant of
# "Δ-steps-before-collapse" framing).
#
# In addition to step_20 already covered by run_predictive_routing_js.sh,
# we now scan the saved checkpoints at step 40, 60, 80, 100 for the three
# fully-trained runs (vanilla / S1 / S2), and step 40 for the two negctl
# runs whose RL is still in progress.
#
# Total: 28 forward jobs.
#   vanilla / S1 / S2  : 4 step × 2 dataset = 8 each → 24 jobs
#   negctl_DAI / Random: 1 step × 2 dataset = 2 each → 4 jobs
#
# Output naming:  {run}_step{NN}_{vanilla|S1soft|S2hard|negDAI|negRand}
# i.e., I_q3_step40_S2hard, I_q3_step60_vanilla, etc. — same suffix scheme
# as run_predictive_routing_js.sh; downstream {h11_*}.py just need
# (run, step) pair.
#
# Same B-anchor (b1r1 step 0 = pure base) as the step_20 batch, so the
# predictive_js precompute can join with B_q3 wave1 parquet on identical
# (prompt_id, sample_id, t).
#
# 8-GPU concurrent worker pool with flock-based job queue.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave5_traj_logs
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=analysis/scripts/a1b_wave1_logprob.py
OUT_DIR=analysis/data/wave5_predictive
N_GPUS=${N_GPUS:-8}

mkdir -p "$LOG_DIR" "$OUT_DIR"
rm -f "$QUEUE" "$QUEUE.lock" "$LOG_DIR"/*.exitcode 2>/dev/null || true

CKPT_BASE=${PROJECT_ROOT:-.}/checkpoints/verl_exp/DAPO_sh_repro
ROLL_BASE=${PROJECT_ROOT:-.}/rollout/val_rollout/DAPO_sh_repro
B_Q3_ROLL=$ROLL_BASE/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl

# (label, ckpt-dir-name)
RUN_DIRS=(
    "vanilla|b2r1_Qwen3-8B-Base_IFTrain_local_H20"
    "S1soft|b2r1_h1_S1_RL_dri50_local_H20"
    "S2hard|b2r1_h1_S2_RL_dri50_local_H20"
    "negDAI|b2r1_h7_negctl_DAI_dri50_local_H20"
    "negRand|b2r1_h7_negctl_random_dri50_local_H20"
)

# fully-trained runs cover step 40,60,80,100 ; negctl only step 40 (so far)
get_steps() {
    case "$1" in
        vanilla|S1soft|S2hard) echo "40 60 80 100" ;;
        negDAI|negRand)        echo "40" ;;
        *)                     echo "" ;;
    esac
}

# job format: TAG|MODEL|DATASET|ROLLOUT|SAMPLES_PER_PROMPT
> "$QUEUE"
for spec in "${RUN_DIRS[@]}"; do
    LBL=${spec%%|*}
    DIR=${spec##*|}
    for STEP in $(get_steps "$LBL"); do
        MODEL=$CKPT_BASE/$DIR/global_step_${STEP}/actor/huggingface
        TAG=I_q3_step${STEP}_${LBL}
        echo "${TAG}|${MODEL}|aime|${B_Q3_ROLL}|32"   >> "$QUEUE"
        echo "${TAG}|${MODEL}|ifeval|${B_Q3_ROLL}|8"  >> "$QUEUE"
    done
done

N_JOBS=$(wc -l < "$QUEUE")
echo "[wave5_traj] queue: $N_JOBS jobs across $N_GPUS GPU workers"
echo "[wave5_traj] logs: $LOG_DIR  output: $OUT_DIR"

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
    setsid nohup bash -c "$worker_body" wave5_traj_gpu$i \
        "$i" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave5_traj] $N_GPUS workers launched (setsid+nohup+disown)"
echo "[wave5_traj] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l  (target $N_JOBS)"
