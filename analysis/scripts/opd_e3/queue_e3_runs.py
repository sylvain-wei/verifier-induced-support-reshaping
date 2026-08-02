#!/usr/bin/env python3
"""queue_e3_runs.py — serially launch the remaining E3 OPD runs.

Workflow
--------
1. Reads the queue (step_40, step_60, step_80) from QUEUE below.
2. Polls every POLL_INTERVAL seconds for "GPUs idle" condition:
     no nvidia-smi compute apps owned by the current user with
     `python3 -m opd_lab.train` in their cmdline AND
     8 GPUs all reporting <IDLE_MEM_THRESHOLD_MB used.
3. When idle is detected, launches the next launcher script with the same
   detached pattern as run_opd_e3_step20 (setsid + nohup + disown).
4. After launching, waits for vLLM to actually allocate KV cache (>1 GB on
   any GPU) before declaring the launch successful, then resumes polling
   until that run completes (back to idle), then launches the next.

Why this rather than `bash &&`
------------------------------
Each E3 run is ~16 h and codebuddy's background-shell turn-end SIGHUP makes
multi-step shell chains brittle. A persistent Python watchdog detached from
the codebuddy session boundary is more reliable. It also gracefully picks up
where it left off if you kill / re-spawn it (see STATE_PATH).

Usage
-----
Detached (recommended):
    setsid nohup python3 analysis/scripts/opd_e3/queue_e3_runs.py \
        < /dev/null > analysis/scripts/opd_e3/logs/queue.out 2>&1 & disown

Stop:
    pkill -f queue_e3_runs.py

Resume after a kill: just relaunch — the queue file is read fresh each time
and STATE_PATH tracks which jobs already started.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = os.environ.get("PROJECT_ROOT", ".")
LAUNCHER_DIR = f"{ROOT}/analysis/scripts/opd_e3"
LOG_DIR = f"{LAUNCHER_DIR}/logs"
QUEUE_LOG = f"{LOG_DIR}/queue.log"
STATE_PATH = f"{LOG_DIR}/queue_state.txt"

# Order matters — step_40 next after step_20, then step_60, then step_80.
QUEUE: list[str] = [
    "run_opd_e3_step40.sh",
    "run_opd_e3_step60.sh",
    "run_opd_e3_step80.sh",
]

POLL_INTERVAL = 120          # s between idleness checks
IDLE_MEM_THRESHOLD_MB = 1500 # all GPUs must be below this to count as idle
LAUNCH_VERIFY_DELAY = 600    # s to wait after launch before declaring success
                             # (model load + Ray init can take ~5 min)
MIN_KVCACHE_MB = 4000        # post-launch, expect ≥ this on at least one GPU


def _utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _log(msg: str) -> None:
    line = f"[{_utcnow()}] {msg}\n"
    try:
        with open(QUEUE_LOG, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


def _load_state() -> set[str]:
    if not os.path.isfile(STATE_PATH):
        return set()
    with open(STATE_PATH) as fh:
        return {l.strip() for l in fh if l.strip()}


def _mark_launched(launcher: str) -> None:
    with open(STATE_PATH, "a") as fh:
        fh.write(f"launched::{launcher}::{_utcnow()}\n")


def _is_marked_launched(launcher: str, state: set[str]) -> bool:
    return any(s.startswith(f"launched::{launcher}::") for s in state)


def _gpu_mem_used_mb() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return [int(x.strip()) for x in out]
    except Exception as e:
        _log(f"nvidia-smi query failed: {e}")
        return [99999] * 8  # treat as "not idle" on error


def _opd_train_running() -> bool:
    """True iff a python3 -m opd_lab.train child is alive on this host."""
    try:
        out = subprocess.check_output(["ps", "-eo", "cmd"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return True  # be conservative
    for line in out.splitlines():
        if "python3 -m opd_lab.train" in line:
            return True
        if "ray::OPDTaskRunner" in line:
            return True
    return False


def _gpus_idle() -> bool:
    if _opd_train_running():
        return False
    mem = _gpu_mem_used_mb()
    return all(m < IDLE_MEM_THRESHOLD_MB for m in mem)


def _launch(launcher: str) -> bool:
    """Launch the named launcher detached. Return True on apparent success."""
    path = os.path.join(LAUNCHER_DIR, launcher)
    if not os.path.isfile(path):
        _log(f"launcher missing: {path}")
        return False

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    boot_log = f"{LOG_DIR}/{launcher.replace('.sh','')}.queue_boot.{ts}.log"
    cmd = (
        f"setsid nohup bash {path} "
        f"< /dev/null > {boot_log} 2>&1 & disown"
    )
    _log(f"launching: {launcher}\n  boot_log={boot_log}\n  cmd={cmd}")

    # Use shell so & disown semantics work as intended.
    rc = subprocess.run(["bash", "-c", cmd], cwd=ROOT).returncode
    if rc != 0:
        _log(f"launch shell exited non-zero: rc={rc}")
        return False

    # Verify: wait until either GPU activity rises OR LAUNCH_VERIFY_DELAY elapses.
    t0 = time.time()
    while time.time() - t0 < LAUNCH_VERIFY_DELAY:
        time.sleep(20)
        mem = _gpu_mem_used_mb()
        if max(mem) > MIN_KVCACHE_MB and _opd_train_running():
            _log(f"launch verified — max GPU memory={max(mem)} MB, opd_lab.train alive")
            _mark_launched(launcher)
            return True
    _log(f"launch verification TIMED OUT after {LAUNCH_VERIFY_DELAY}s")
    _log(f"  most recent boot log: {boot_log}")
    _log(f"  current GPU mem: {_gpu_mem_used_mb()}")
    _log(f"  opd_lab.train alive? {_opd_train_running()}")
    return False


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    state = _load_state()
    pending = [q for q in QUEUE if not _is_marked_launched(q, state)]
    _log(f"queue runner started; pending={pending}")

    if not pending:
        _log("nothing pending; exiting cleanly")
        return

    for launcher in pending:
        # Wait for GPUs to be idle (i.e. previous run finished).
        _log(f"waiting for GPUs to become idle before launching {launcher} …")
        wait_iters = 0
        while not _gpus_idle():
            time.sleep(POLL_INTERVAL)
            wait_iters += 1
            if wait_iters % 15 == 0:
                # ~30 min progress note
                mem = _gpu_mem_used_mb()
                _log(
                    f"still waiting ({wait_iters * POLL_INTERVAL}s elapsed) — "
                    f"GPU mem: {mem} max={max(mem)} train_running={_opd_train_running()}"
                )

        _log(f"GPUs idle detected; sleeping 60s for graceful settle, then launching")
        time.sleep(60)

        ok = _launch(launcher)
        if not ok:
            _log(f"FAILED to launch {launcher}; aborting queue. Manual intervention needed.")
            return

        # Now wait for it to occupy GPUs before going back to "idle wait" loop
        # (we don't want to immediately think GPUs are idle and double-launch).
        # _launch already verified, so just continue.

    _log("queue exhausted; all 3 runs launched. Exit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("queue runner stopped by SIGINT")
        sys.exit(0)
    except Exception:
        _log("queue runner crashed:\n" + traceback.format_exc())
        sys.exit(1)
