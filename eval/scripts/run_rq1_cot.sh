#!/usr/bin/env bash
# scripts/run_rq1_cot.sh
#
# Step-by-step ablation launcher. Structurally identical to run_rq1_full.sh
# but only the three math benchmarks in their CoT variant:
#   math500_cot  (cot_math prompt,          K=16)
#   aime24_cot   (cot_aime prompt,          K=32)
#   gsm8k_cot    (cot_math_numeric prompt,  K=1)
#
# Outputs live under responses/$RUN_ID/ and metrics/$RUN_ID/ with the _cot
# suffix in benchmark names, so they physically do NOT overlap with the
# RQ1 main run at 20260504_rq1/.
#
# Usage:
#   bash scripts/run_rq1_cot.sh
#   nohup bash scripts/run_rq1_cot.sh > /dev/null 2>&1 &
#   tail -F logs/20260507_rq1_cot/02_*.log
#
# Env knobs:
#   RUN_ID       default 20260507_rq1_cot — output namespace
#   GPUS         default "0 1 2"          — one GPU per model, in MODEL order
#   EVAL_ROOT    default $EVAL_ROOT/eval
#
# Resume: re-running is safe (run_inference.py skips finished rows).

set -euo pipefail

# ---- config ----
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export RUN_ID="${RUN_ID:-20260507_rq1_cot}"
GPUS="${GPUS:-0 1 2}"
export PYTHONPATH="${EVAL_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$EVAL_ROOT"

MODELS=(base math_rlvr if_rlvr)

read -ra GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -lt ${#MODELS[@]} ]]; then
  echo "ERROR: need ${#MODELS[@]} GPUs (one per model), got ${#GPU_ARR[@]} from GPUS='$GPUS'" >&2
  exit 1
fi

TASKS=(
  "math500_cot|sampling_k16"
  "aime24_cot|sampling_k32"
  "gsm8k_cot|greedy"
)

LOG_DIR="$EVAL_ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# ---- preflight banner ----
echo "============================================================"
echo "  RQ1 COT ABLATION RUN"
echo "------------------------------------------------------------"
echo "  run_id:    $RUN_ID"
echo "  eval_root: $EVAL_ROOT"
echo "  log_dir:   $LOG_DIR"
echo "  models:    ${MODELS[*]}"
echo "  gpus:      $GPUS  (one model per GPU)"
echo "  tasks:     ${TASKS[*]}"
echo "  started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo

# ---- [1/4] env check ----
echo "==> [1/4] env check"
python scripts/check_env.py 2>&1 | tee "$LOG_DIR/00_env_check.log"
echo

# ---- [1.5/4] GPU memory check ----
echo "==> GPU memory check on GPUs: $GPUS"
export GPUS
python - <<'PY' 2>&1 | tee "$LOG_DIR/00_gpu_check.log"
import os, sys
try:
    import torch
except Exception as e:
    print("torch missing:", e); sys.exit(1)
gpus = [int(x) for x in os.environ.get("GPUS", "0 1 2").split()]
any_busy = False
for g in gpus:
    free, total = torch.cuda.mem_get_info(g)
    free_gb, total_gb = free / 1e9, total / 1e9
    tag = "OK" if free_gb > 80 else "BUSY"
    if free_gb <= 80:
        any_busy = True
    print(f"  GPU {g}: free={free_gb:.1f}G / total={total_gb:.1f}G  [{tag}]")
if any_busy:
    print("WARNING: at least one assigned GPU has <80G free. vLLM may OOM.")
PY
echo

# ---- [2/4] prepare data ----
# Prepare ONLY the cot variants; non-cot processed files already exist from
# the main RQ1 run and are not touched.
echo "==> [2/4] prepare data (idempotent): math500_cot aime24_cot gsm8k_cot"
python scripts/prepare_data.py --only math500_cot aime24_cot gsm8k_cot 2>&1 | tee "$LOG_DIR/01_prepare_data.log"
echo

# ---- [3/4] launch model shells in parallel ----
echo "==> [3/4] launching parallel inference+metrics on ${#MODELS[@]} GPUs"
echo "         Monitor:  tail -F $LOG_DIR/02_*.log"
echo

START_TS=$(date +%s)

launch_model() {
  local model=$1 gpu=$2 i=$3
  local log="$LOG_DIR/02_${model}.log"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    sleep $((i * 5))  # stagger cold-starts
    exec 1>>"$log" 2>&1
    echo "############################################################"
    echo "# MODEL=$model  GPU=$gpu  start=$(date '+%H:%M:%S')"
    echo "############################################################"
    for spec in "${TASKS[@]}"; do
      local bench="${spec%%|*}"
      local mode="${spec##*|}"
      echo
      echo "---- [$(date '+%H:%M:%S')] $model / $bench / $mode ----"
      python scripts/run_inference.py \
        --run-id "$RUN_ID" --model-id "$model" \
        --benchmark "$bench" --mode "$mode"
      python scripts/compute_metrics.py \
        --run-id "$RUN_ID" --model-id "$model" \
        --benchmark "$bench" --mode "$mode"
    done
    echo
    echo "############################################################"
    echo "# MODEL=$model DONE  end=$(date '+%H:%M:%S')"
    echo "############################################################"
  ) &
}

PIDS=()
for i in "${!MODELS[@]}"; do
  launch_model "${MODELS[$i]}" "${GPU_ARR[$i]}" "$i"
  pid=$!
  PIDS+=("$pid")
  echo "  launched ${MODELS[$i]} on GPU ${GPU_ARR[$i]} (PID $pid, log $LOG_DIR/02_${MODELS[$i]}.log)"
done
echo

trap 'echo; echo "[INT] killing child PIDs ${PIDS[*]}"; kill "${PIDS[@]}" 2>/dev/null || true; exit 130' INT TERM

FAIL=0
FAILED_MODELS=()
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  model="${MODELS[$idx]}"
  if wait "$pid"; then
    echo "  [$(date '+%H:%M:%S')] $model finished OK"
  else
    echo "  [$(date '+%H:%M:%S')] $model FAILED (see $LOG_DIR/02_${model}.log)"
    FAILED_MODELS+=("$model")
    FAIL=$((FAIL + 1))
  fi
done

trap - INT TERM

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "  parallel phase: $((ELAPSED/60))m $((ELAPSED%60))s  (failed: $FAIL/${#MODELS[@]})"
echo

# ---- [4/4] aggregate + plot data ----
echo "==> [4/4] aggregate + plot data"
python scripts/aggregate_metrics.py --run-id "$RUN_ID" 2>&1 | tee "$LOG_DIR/03_aggregate.log"
# make_plot_data hardcodes the RQ1 5-benchmark layout; cot ablation's
# aggregate csv is enough on its own, so we skip the 2D pattern plot here.

echo
echo "============================================================"
echo "  RQ1 COT ABLATION RUN  — summary"
echo "------------------------------------------------------------"
echo "  run_id:         $RUN_ID"
echo "  wall time:      $((ELAPSED/60))m $((ELAPSED%60))s"
echo "  failed models:  ${FAILED_MODELS[*]:-(none)}"
echo "  main summary:   $EVAL_ROOT/metrics/$RUN_ID/aggregate/rq1_main_summary.csv"
echo "  per-model log:  $LOG_DIR/02_{base,math_rlvr,if_rlvr}.log"
echo "============================================================"

# Compact preview of the main summary
if [[ -f "$EVAL_ROOT/metrics/$RUN_ID/aggregate/rq1_main_summary.csv" ]]; then
  echo
  echo "=== rq1_main_summary.csv (cot headline metrics) ==="
  python - <<'PY'
import csv, os
run_id = os.environ["RUN_ID"]
root = os.environ["EVAL_ROOT"]
path = f"{root}/metrics/{run_id}/aggregate/rq1_main_summary.csv"
with open(path) as f:
    rows = list(csv.reader(f))
hdr = rows[0]; idx = {c: hdr.index(c) for c in hdr}

headline_specs = [
    ("m500cot_pass@1",    "math500_cot__sampling_k16__pass@1"),
    ("m500cot_best@k",    "math500_cot__sampling_k16__best@k"),
    ("m500cot_len",       "math500_cot__sampling_k16__response_length_tokens_mean"),
    ("m500cot_stepd",     "math500_cot__sampling_k16__step_density_mean"),
    ("aime24cot_pass@1",  "aime24_cot__sampling_k32__pass@1"),
    ("aime24cot_best@k",  "aime24_cot__sampling_k32__best@k"),
    ("aime24cot_bmp",     "aime24_cot__sampling_k32__best_minus_pass"),
    ("aime24cot_len",     "aime24_cot__sampling_k32__response_length_tokens_mean"),
    ("gsm8kcot_pass@1",   "gsm8k_cot__greedy__pass@1"),
    ("gsm8kcot_len",      "gsm8k_cot__greedy__response_length_tokens_mean"),
    ("gsm8kcot_stepd",    "gsm8k_cot__greedy__step_density_mean"),
]

header = ["model_id"] + [label for label, _ in headline_specs]
print("  " + " | ".join(f"{c:>16s}" for c in header))
print("  " + "-" * (17 * len(header) + 3))
def fmt(v):
    try: return f"{float(v):.3f}"
    except: return str(v)[:14]
for r in rows[1:]:
    row = [r[idx["model_id"]]]
    for _, col in headline_specs:
        row.append(fmt(r[idx[col]]) if col in idx else "—")
    print("  " + " | ".join(f"{c:>16s}" for c in row))
PY
fi

echo
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
