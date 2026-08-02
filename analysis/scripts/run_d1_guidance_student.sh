#!/bin/bash
# Track B / D.1 — score the OPD vanilla student rollouts under {base,
# teacher_if, student_after_opd}. Sharded across N_GPUS GPU workers.
#
# Pre: extended compute_guidance_scores.py supports --shard_id / --n_shards
# and --model_tag student_after_opd. After workers finish, merge mode
# concats per-shard parquets into the final
# guidance_scores_student_after_opd.parquet.
set -u
cd ${PROJECT_ROOT:-.}

LOG_DIR=/tmp/d1_guidance_logs
N_GPUS=${N_GPUS:-2}
GPU_OFFSET=${GPU_OFFSET:-6}
SCORERS=${SCORERS:-"base,teacher_if,student_after_opd"}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.exitcode 2>/dev/null

worker_body='
GPU=$1
SHARD=$2
NSHARDS=$3
LOG_DIR=$4
SCORERS=$5

JOB_LOG=$LOG_DIR/shard${SHARD}of${NSHARDS}__gpu${GPU}.log
echo "[gpu$GPU shard $SHARD/$NSHARDS] starting, scorers=$SCORERS" >> "$JOB_LOG"
CUDA_VISIBLE_DEVICES=$GPU python analysis/scripts/opd_sanity/compute_guidance_scores.py \
    --model_tag student_after_opd \
    --scorers "$SCORERS" \
    --shard_id $SHARD --n_shards $NSHARDS \
    >> "$JOB_LOG" 2>&1
EXIT=$?
echo "$EXIT" > "$LOG_DIR/shard${SHARD}of${NSHARDS}.exitcode"
echo "[gpu$GPU shard $SHARD/$NSHARDS] EXIT=$EXIT" >> "$JOB_LOG"
'

# Launch one worker per GPU, each handles 1 shard out of N_GPUS shards
for k in $(seq 0 $((N_GPUS - 1))); do
    GPU=$((GPU_OFFSET + k))
    setsid nohup bash -c "$worker_body" d1_guidance_gpu$GPU \
        "$GPU" "$k" "$N_GPUS" "$LOG_DIR" "$SCORERS" \
        </dev/null >/dev/null 2>&1 & disown
done

echo "[d1_guidance] $N_GPUS shards launched on GPUs $GPU_OFFSET..$((GPU_OFFSET+N_GPUS-1))"
echo "[d1_guidance] watch: ls $LOG_DIR/*.exitcode 2>/dev/null | wc -l"
echo "[d1_guidance] when all done, merge with:"
echo "    python analysis/scripts/opd_sanity/compute_guidance_scores.py \\"
echo "        --model_tag student_after_opd --merge"
