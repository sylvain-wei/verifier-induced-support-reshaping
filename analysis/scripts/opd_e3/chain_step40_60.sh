#!/usr/bin/env bash
# Wait until the step-40 evaluation finishes, then run the step-60 evaluation.
set -euo pipefail
ROOT=${PROJECT_ROOT:-.}

while pgrep -f "RUN_FILTER=step40" >/dev/null 2>&1 || pgrep -f "VLLM::Worker_TP" >/dev/null 2>&1 || pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; do sleep 60; done
# extra: after the eval_if_offline.sh exits, wait for GPU to drain
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1500 ]; do sleep 30; done
echo "[chain] step40 eval finished, GPUs idle. starting step60 eval"

RUN_FILTER="step60" bash "${ROOT}/analysis/scripts/opd_e3/eval_if_offline.sh" 2>&1 | tee "${ROOT}/analysis/scripts/opd_e3/logs/eval_if_offline_step60_$(date +%Y%m%d_%H%M%S).log"

# wait for step60 GPU to drain too
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1500 ]; do sleep 30; done
echo "[chain] step60 eval finished"
echo "[chain] done."
