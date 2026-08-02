#!/usr/bin/env python3
"""auto_run_offline_eval.py — wait for step_80 to finish, then chain offline IF eval.

Workflow
--------
1. Polls every 120 s for "step_80 training has finished":
     - opd_lab.train / ray::OPDTaskRunner processes are gone, AND
     - all 8 GPUs report < IDLE_MEM_THRESHOLD_MB used.
2. Once idle is detected, sleeps 60 s for graceful settle, then
   `setsid nohup bash analysis/scripts/opd_e3/eval_if_offline.sh ...`
   to evaluate IFEval-test + IFBench across all 4 E3 runs × all 4 ckpts
   (step 25/50/75/100). step_20's IFEval/IFBench were already gathered
   in-line during training; the offline pass only fills in step_40/60/80
   (and re-runs step_20 only if those CSV rows aren't already present —
   eval_if_offline.py is idempotent).
3. Logs everything to analysis/scripts/opd_e3/logs/auto_offline_eval.log.

Why a separate watcher
----------------------
queue_e3_runs.py already exited after launching step_80 (its job was just
the 3-launcher chain). This is a one-shot post-training trigger and
doesn't share state with the queue runner.

Usage (detached, recommended):
    setsid nohup python3 analysis/scripts/opd_e3/auto_run_offline_eval.py \\
        < /dev/null > analysis/scripts/opd_e3/logs/auto_offline_eval.out 2>&1 & disown

Stop (e.g. if you want to launch eval manually):
    pkill -f auto_run_offline_eval.py
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
import traceback

ROOT = os.environ.get("PROJECT_ROOT", ".")
LOG_DIR = f"{ROOT}/analysis/scripts/opd_e3/logs"
SELF_LOG = f"{LOG_DIR}/auto_offline_eval.log"
EVAL_SH = f"{ROOT}/analysis/scripts/opd_e3/eval_if_offline.sh"

POLL_INTERVAL = 120          # s between idleness checks
IDLE_MEM_THRESHOLD_MB = 1500
SETTLE_SLEEP_S = 60
LAUNCH_VERIFY_DELAY = 600


def _utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _log(msg: str) -> None:
    line = f"[{_utcnow()}] {msg}\n"
    try:
        with open(SELF_LOG, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


def _gpu_mem_used_mb() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return [int(x.strip()) for x in out]
    except Exception as e:
        _log(f"nvidia-smi query failed: {e}")
        return [99999] * 8


def _opd_train_running() -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "cmd"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return True
    for line in out.splitlines():
        if "python3 -m opd_lab.train" in line:
            return True
        if "ray::OPDTaskRunner" in line:
            return True
        if "run_opd_e3_step" in line:
            return True
    return False


def _gpus_idle() -> bool:
    if _opd_train_running():
        return False
    return all(m < IDLE_MEM_THRESHOLD_MB for m in _gpu_mem_used_mb())


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    _log("auto_run_offline_eval started; waiting for step_80 to finish")

    # Wait for idle
    wait_iters = 0
    while not _gpus_idle():
        time.sleep(POLL_INTERVAL)
        wait_iters += 1
        if wait_iters % 15 == 0:
            mem = _gpu_mem_used_mb()
            _log(
                f"still waiting ({wait_iters * POLL_INTERVAL}s elapsed) — "
                f"GPU mem: {mem} max={max(mem)} train_running={_opd_train_running()}"
            )

    _log(f"GPUs idle detected; sleeping {SETTLE_SLEEP_S}s for graceful settle")
    time.sleep(SETTLE_SLEEP_S)

    # Launch offline eval, detached
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_log = f"{LOG_DIR}/eval_if_offline_chained.{ts}.log"
    cmd = (
        f"setsid nohup bash {EVAL_SH} "
        f"< /dev/null > {out_log} 2>&1 & disown"
    )
    _log(f"launching offline IF eval; output_log={out_log}")
    rc = subprocess.run(["bash", "-c", cmd], cwd=ROOT).returncode
    if rc != 0:
        _log(f"launch shell exited non-zero: rc={rc}")
        return

    # Verify it actually started by waiting for GPU activity
    t0 = time.time()
    while time.time() - t0 < LAUNCH_VERIFY_DELAY:
        time.sleep(20)
        mem = _gpu_mem_used_mb()
        if max(mem) > 4000:
            _log(f"offline eval verified — max GPU memory={max(mem)} MB")
            return
    _log(f"offline eval launch verification TIMED OUT after {LAUNCH_VERIFY_DELAY}s; "
         f"check {out_log} manually")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("auto_run_offline_eval stopped by SIGINT")
        sys.exit(0)
    except Exception:
        _log("auto_run_offline_eval crashed:\n" + traceback.format_exc())
        sys.exit(1)
