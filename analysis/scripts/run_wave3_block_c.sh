#!/bin/bash
# WAVE 3 / Block C P0 — orchestrate 9 (condition, shard) jobs across N GPUs.
#
# 9 conditions × 30 AIME prompts × 32 samples each. To fit in 1.5 hr/condition
# on 1 GPU (per plan §6 budget), we shard each condition's 30 prompts across
# ${SHARDS_PER_COND} GPU workers. Default = 4-way prompt-shard, so a single
# condition occupies 4 GPUs simultaneously and produces 4 partial parquets
# that we later merge.
#
# Schedule: queue is (condition, shard_id) pairs. Workers pop greedily.
# So if N_GPUS=8 and SHARDS_PER_COND=4 → 2 conditions running in parallel
# at any time, 9 conditions sequentially in ~9/2 * (30/4) prompts/(2 p/min)
# ≈ 9 hr (worst-case). For tighter control, set SHARDS_PER_COND=8.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/wave3_logs
QUEUE=$LOG_DIR/jobs.txt
SCRIPT=${SCRIPT:-analysis/scripts/f1_vllm_intervention.py}
OUT_DIR=analysis/data/f1_intervention
N_GPUS=${N_GPUS:-8}
GPU_OFFSET=${GPU_OFFSET:-0}
# Optional: GPU_LIST="0 2 3 4 5" overrides N_GPUS / GPU_OFFSET when set.
GPU_LIST=${GPU_LIST:-""}
SHARDS_PER_COND=${SHARDS_PER_COND:-8}
N_SAMPLES=${N_SAMPLES:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
CONDITIONS=${CONDITIONS:-"single_token_C1 single_token_C2 random_top2_B_q3 random_top2_I_q3 forced_DRI_B_q3 forced_DAI_B_q3 forced_DRI_I_q3 forced_DAI_I_q3 free"}
PRIMARIES=${PRIMARIES:-"B_q3 I_q3"}
QUEUE_BY=${QUEUE_BY:-"primary"}  # "primary" -> per (primary,shard); "condition" -> per (condition,shard)

mkdir -p "$LOG_DIR" "$OUT_DIR"
rm -f "$QUEUE.lock" 2>/dev/null

# Build queue.
> "$QUEUE"
if [ "$QUEUE_BY" = "primary" ]; then
    # Per-(primary, shard_id) lines, executed via --primary so the worker
    # loads vLLM once and drains all conditions for that primary.
    for prim in $PRIMARIES; do
        for s in $(seq 0 $((SHARDS_PER_COND - 1))); do
            echo "PRIM|${prim}|${s}|${SHARDS_PER_COND}" >> "$QUEUE"
        done
    done
else
    # Per-(condition, shard_id) lines, executed via --condition.
    for cond in $CONDITIONS; do
        for s in $(seq 0 $((SHARDS_PER_COND - 1))); do
            out_pq="$OUT_DIR/${cond}__shard${s}of${SHARDS_PER_COND}.parquet"
            if [ -f "$out_pq" ]; then
                echo "[wave3] skip already-done $out_pq" >&2
                continue
            fi
            echo "COND|${cond}|${s}|${SHARDS_PER_COND}" >> "$QUEUE"
        done
    done
fi

N_JOBS=$(wc -l < "$QUEUE")
echo "[wave3] queue: $N_JOBS (condition, shard) jobs across $N_GPUS GPU workers"
echo "[wave3] conditions: $CONDITIONS"
echo "[wave3] shards/cond=$SHARDS_PER_COND  n_samples=$N_SAMPLES  max_new=$MAX_NEW_TOKENS"
echo "[wave3] logs: $LOG_DIR  output: $OUT_DIR"

worker_body='
GPU=$1
QUEUE=$2
LOG_DIR=$3
SCRIPT=$4
OUT_DIR=$5
NSAMPLES=$6
MAXNEW=$7

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
    IFS="|" read -r KIND VAL SHARD NSHARDS <<< "$JOB"
    NAME="${KIND}_${VAL}__shard${SHARD}of${NSHARDS}"
    echo "[gpu$GPU] starting $NAME" >> "$LOG_DIR/gpu${GPU}.log"
    JOB_LOG=$LOG_DIR/${NAME}.log

    if [ "$KIND" = "PRIM" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
            --primary "$VAL" \
            --shard_id $SHARD --n_shards $NSHARDS \
            --n_samples $NSAMPLES \
            --max_new_tokens $MAXNEW \
            --tp 1 \
            --out_dir "$OUT_DIR" \
            > "$JOB_LOG" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
            --condition "$VAL" \
            --shard_id $SHARD --n_shards $NSHARDS \
            --n_samples $NSAMPLES \
            --max_new_tokens $MAXNEW \
            --tp 1 \
            --out_dir "$OUT_DIR" \
            > "$JOB_LOG" 2>&1
    fi
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

if [ -n "$GPU_LIST" ]; then
    GPUS_TO_USE="$GPU_LIST"
else
    GPUS_TO_USE=""
    for k in $(seq 0 $((N_GPUS - 1))); do
        GPUS_TO_USE="$GPUS_TO_USE $((GPU_OFFSET + k))"
    done
fi
echo "[wave3] GPUs to use: $GPUS_TO_USE"

for GPU in $GPUS_TO_USE; do
    setsid nohup bash -c "$worker_body" wave3_gpu$GPU \
        "$GPU" "$QUEUE" "$LOG_DIR" "$SCRIPT" "$OUT_DIR" \
        "$N_SAMPLES" "$MAX_NEW_TOKENS" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[wave3] workers launched on GPUs:$GPUS_TO_USE"
echo "[wave3] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
