#!/usr/bin/env bash
# chain_dri_then_3class.sh — wait for DRI backfill (run_dri_backfill_b2r1_step40_80.sh)
# to finish, then launch the E3 3-class judge pipeline.
#
# Idempotent — checks for sentinel outputs and skips if already produced.

set -euo pipefail

ROOT=${PROJECT_ROOT:-.}

# Wait for DRI backfill: poll for both step_40.json and step_80.json to exist.
EVAL_DIR=${ROOT}/eval_results/b2r1_Qwen3-8B-Base_IFTrain_local_H20_math_support_probe
STEP40_JSON=${EVAL_DIR}/step_40.json
STEP80_JSON=${EVAL_DIR}/step_80.json

echo "[chain] waiting for DRI backfill (looking for ${STEP40_JSON} and ${STEP80_JSON})..."
while true; do
    if [[ -f "$STEP40_JSON" && -f "$STEP80_JSON" ]]; then
        echo "[chain] both DRI backfill JSONs present → proceeding"
        break
    fi
    # Also break if the parent bash script of run_dri_backfill is gone AND the
    # outputs are still missing (i.e. it crashed): print the last few lines and
    # bail out so we don't loop forever.
    BPID=$(cat ${ROOT}/analysis/scripts/logs/dri_backfill_pid.txt 2>/dev/null || echo "")
    if [[ -n "$BPID" ]] && ! kill -0 "$BPID" 2>/dev/null; then
        if [[ ! -f "$STEP40_JSON" || ! -f "$STEP80_JSON" ]]; then
            echo "[chain] ERROR: backfill bash $BPID dead but outputs missing" >&2
            tail -40 ${ROOT}/analysis/scripts/logs/dri_backfill_*.log 2>/dev/null
            exit 1
        fi
    fi
    sleep 60
done

# Now launch the E3 3-class judge pipeline.
echo "[chain] launching E3 3-class judge..."
bash ${ROOT}/analysis/scripts/opd_e3/run_e3_3class_judge.sh
echo "[chain] all done"
