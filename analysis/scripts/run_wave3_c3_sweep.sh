#!/bin/bash
# WAVE 3 / C.3 — Position Sweep launcher.
#
# Reuses run_wave3_block_c.sh by overriding CONDITIONS / PRIMARIES env vars.
# `f1_vllm_intervention.py` auto-parses the trailing `_pos{N}` suffix from
# each condition name and dispatches to the 2-stage code path when N > 1.
#
# Total: 4 strategies × 6 new positions = 24 conditions × 8 shards = 192 jobs.
# Estimated wall: 6-8 hr on 8 GPUs (each primary worker drains 12 conditions
# after 1 vLLM load).
#
# Usage:
#   bash analysis/scripts/run_wave3_c3_sweep.sh
#   bash analysis/scripts/run_wave3_c3_sweep.sh single_token_C1_pos5  # one cond
#
# Logs:   /tmp/wave3_logs/
# Output: analysis/data/f1_intervention/{condition}__shard{N}of{M}.parquet
set -u

# Allow overriding the condition list from CLI (for dry-run on a single cond).
if [ "$#" -ge 1 ]; then
    CONDS="$@"
else
    CONDS=""
    for STRAT in single_token_C1 single_token_C2 random_top2_B_q3 random_top2_I_q3; do
        for POS in 2 3 5 20 50 100; do
            CONDS="$CONDS ${STRAT}_pos${POS}"
        done
    done
fi

# Build the PRIMARIES set covering whatever conditions we requested.
# Map each condition to its primary via grep on f1_vllm_intervention.py CONDITIONS.
NEEDED_PRIMARIES=""
for c in $CONDS; do
    case "$c" in
        single_token_C1_pos*|random_top2_B_q3_pos*) p=B_q3 ;;
        single_token_C2_pos*|random_top2_I_q3_pos*) p=I_q3 ;;
        *) echo "[c3_sweep] unknown condition $c, skipping" >&2 ; continue ;;
    esac
    case " $NEEDED_PRIMARIES " in *" $p "*) ;; *) NEEDED_PRIMARIES="$NEEDED_PRIMARIES $p" ;; esac
done

echo "[c3_sweep] conditions: $CONDS"
echo "[c3_sweep] primaries:  $NEEDED_PRIMARIES"

# Use QUEUE_BY=condition so each (condition, shard) is its own queue entry,
# regardless of primary grouping. This way a single `single_token_C1_pos5`
# request can run without dragging in the other 11 conds for B_q3.
export CONDITIONS="$CONDS"
export PRIMARIES="$NEEDED_PRIMARIES"
export QUEUE_BY=condition
export SHARDS_PER_COND=${SHARDS_PER_COND:-8}
export N_GPUS=${N_GPUS:-8}

bash analysis/scripts/run_wave3_block_c.sh
