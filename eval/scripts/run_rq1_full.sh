#!/usr/bin/env bash
# scripts/run_rq1_full.sh
#
# Production parallel launcher for the RQ1 required matrix.
#
# Layout: one model per GPU. Each model serially runs its 5 benchmark cells
# (math500, aime24, gsm8k, ifeval, ifbench). When all three model shells
# finish, run aggregate + make_plot_data once to produce the main summary.
#
# Expected wall time on 3× H20 parallel: ~65-70 minutes.
# Expected wall time on 1× H20 serial:   ~3.3 hours.
#
# Required matrix (15 cells, 3 models × 5 benchmarks):
#   math500  sampling_k16  N=500   K=16   T=0.7   (pattern + correctness)
#   aime24   sampling_k32  N=30    K=32   T=0.7   (hard math + search benefit)
#   gsm8k    greedy        N=1319  K=1    T=0.0   (short-chain diagnostic)
#   ifeval   greedy        N=541   K=1    T=0.0   (main IF benchmark)
#   ifbench  greedy        N=300   K=1    T=0.0   (OOD IF benchmark)
#
# Usage (foreground, blocking):
#   bash scripts/run_rq1_full.sh
#
# Usage (background, disconnectable):
#   nohup bash scripts/run_rq1_full.sh > /dev/null 2>&1 &
#   # then monitor:
#   tail -F logs/20260504_rq1/02_*.log
#
# Env knobs:
#   RUN_ID       default 20260504_rq1     — output namespace
#   GPUS         default "0 1 2"          — one GPU per model, in MODEL order
#   EVAL_ROOT    default $EVAL_ROOT/eval
#
# Resume: re-running the same command is safe. run_inference.py skips any
# (example_id, sample_id) already written to the target jsonl. Pass
# --overwrite on individual calls to force re-run.

set -euo pipefail

# ---- config ----
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT:-.}/eval}"
export RUN_ID="${RUN_ID:-20260504_rq1}"
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
  "math500|sampling_k16"
  "aime24|sampling_k32"
  "gsm8k|greedy"
  "ifeval|greedy"
  "ifbench|greedy"
)

LOG_DIR="$EVAL_ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# ---- preflight banner ----
echo "============================================================"
echo "  RQ1 FULL RUN"
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
# Export GPUS so the python heredoc below can read it from os.environ.
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
echo "==> [2/4] prepare data (idempotent)"
python scripts/prepare_data.py --only math500 aime24 gsm8k ifeval ifbench 2>&1 | tee "$LOG_DIR/01_prepare_data.log"
echo

# ---- [3/4] launch 3 model shells in parallel ----
echo "==> [3/4] launching parallel inference+metrics on ${#MODELS[@]} GPUs"
echo "         Monitor:  tail -F $LOG_DIR/02_*.log"
echo

START_TS=$(date +%s)

# Per-model subshell runner. Called inline (NOT via $(...)) so the
# background job it spawns is a direct child of this shell, so `wait` works.
# Usage: launch_model MODEL GPU_ID INDEX  → spawns backgrounded job, sets $!.
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

# Graceful SIGINT: kill children and exit
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
echo "==> [4/4] aggregate + plot data (from whatever responses exist)"
python scripts/aggregate_metrics.py --run-id "$RUN_ID" 2>&1 | tee "$LOG_DIR/03_aggregate.log"
python scripts/make_plot_data.py    --run-id "$RUN_ID" 2>&1 | tee "$LOG_DIR/04_make_plot.log"

# ---- final summary ----
echo
echo "============================================================"
echo "  RQ1 FULL RUN  — summary"
echo "------------------------------------------------------------"
echo "  run_id:         $RUN_ID"
echo "  wall time:      $((ELAPSED/60))m $((ELAPSED%60))s"
echo "  failed models:  ${FAILED_MODELS[*]:-(none)}"
echo "  main summary:   $EVAL_ROOT/metrics/$RUN_ID/aggregate/rq1_main_summary.csv"
echo "  pattern index:  $EVAL_ROOT/metrics/$RUN_ID/plot_data/pattern_indexes.csv"
echo "  per-model log:  $LOG_DIR/02_{base,math_rlvr,if_rlvr}.log"
echo "============================================================"

# Print a compact preview of the main summary
if [[ -f "$EVAL_ROOT/metrics/$RUN_ID/aggregate/rq1_main_summary.csv" ]]; then
  echo
  echo "=== rq1_main_summary.csv (headline metrics) ==="
  python - <<'PY'
import csv, os
run_id = os.environ["RUN_ID"]
root = os.environ["EVAL_ROOT"]
path = f"{root}/metrics/{run_id}/aggregate/rq1_main_summary.csv"
with open(path) as f:
    rows = list(csv.reader(f))
hdr = rows[0]; idx = {c: hdr.index(c) for c in hdr}

headline_specs = [
    ("math500_pass@1",     "math500__sampling_k16__pass@1"),
    ("math500_best@k",     "math500__sampling_k16__best@k"),
    ("math500_len",        "math500__sampling_k16__response_length_tokens_mean"),
    ("math500_step_dens",  "math500__sampling_k16__step_density_mean"),
    ("aime24_pass@1",      "aime24__sampling_k32__pass@1"),
    ("aime24_best@k",      "aime24__sampling_k32__best@k"),
    ("aime24_best-pass",   "aime24__sampling_k32__best_minus_pass"),
    ("aime24_len",         "aime24__sampling_k32__response_length_tokens_mean"),
    ("gsm8k_pass@1",       "gsm8k__greedy__pass@1"),
    ("gsm8k_len",          "gsm8k__greedy__response_length_tokens_mean"),
    ("gsm8k_step_dens",    "gsm8k__greedy__step_density_mean"),
    ("ifeval_strict",      "ifeval__greedy__strict_prompt_pass"),
    ("ifeval_compl_eff",   "ifeval__greedy__compliance_efficiency"),
    ("ifeval_len",         "ifeval__greedy__response_length_tokens"),
    ("ifbench_strict",     "ifbench__greedy__strict_prompt_pass"),
    ("ifbench_compl_eff",  "ifbench__greedy__compliance_efficiency"),
    ("ifbench_len",        "ifbench__greedy__response_length_tokens"),
]

header = ["model_id"] + [label for label, _ in headline_specs]
print("  " + " | ".join(f"{c:>14s}" for c in header))
print("  " + "-" * (15 * len(header) + 3))
def fmt(v):
    try: return f"{float(v):.3f}"
    except: return str(v)[:12]
for r in rows[1:]:
    row = [r[idx["model_id"]]]
    for _, col in headline_specs:
        row.append(fmt(r[idx[col]]) if col in idx else "—")
    print("  " + " | ".join(f"{c:>14s}" for c in row))
PY
fi

echo
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
