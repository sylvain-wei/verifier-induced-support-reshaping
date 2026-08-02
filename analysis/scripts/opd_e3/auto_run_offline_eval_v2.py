#!/usr/bin/env python3
"""auto_run_offline_eval_v2.py — wait for merger to finish, then chain offline IF eval.

V2 difference from v1
---------------------
v1 only watched for "step_80 training done"; it didn't account for the FSDP
merger step. v2 polls until:
  - all 16 E3 ckpts have safetensors files in their huggingface/ subdirs, AND
  - GPUs are idle (max < IDLE_MEM_THRESHOLD_MB), AND
  - no merge_e3_ckpts.sh / legacy_model_merger.py / opd_lab.train procs alive
Then it launches eval_if_offline.sh detached.

Usage (detached, recommended):
    setsid nohup python3 analysis/scripts/opd_e3/auto_run_offline_eval_v2.py \\
        < /dev/null > analysis/scripts/opd_e3/logs/auto_offline_eval_v2.out 2>&1 & disown
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
import traceback

ROOT = os.environ.get("PROJECT_ROOT", ".")
OPD_OUTPUT_ROOT = os.environ.get("OPD_OUTPUT_ROOT", f"{ROOT}/opd-lab/outputs")
LOG_DIR = f"{ROOT}/analysis/scripts/opd_e3/logs"
SELF_LOG = f"{LOG_DIR}/auto_offline_eval_v2.log"
EVAL_SH = f"{ROOT}/analysis/scripts/opd_e3/eval_if_offline.sh"

# E3 ckpts whose huggingface/ dir we expect to contain safetensors.
E3_CKPT_HF_DIRS = [
    f"{OPD_OUTPUT_ROOT}/baseline_opd_topk_reverse_kl_k16_tch_huggingface_{stamp}/global_step_{step}/actor/huggingface"
    for stamp in ("20260522_141515", "20260523_030754", "20260523_055520", "20260523_090845")
    for step in (25, 50, 75, 100)
]

POLL_INTERVAL = 60
IDLE_MEM_THRESHOLD_MB = 1500
SETTLE_SLEEP_S = 30
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


def _busy_procs_alive() -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "cmd"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return True
    for kw in ("merge_e3_ckpts.sh", "legacy_model_merger.py",
               "python3 -m opd_lab.train", "ray::OPDTaskRunner",
               "eval_if_offline"):
        for line in out.splitlines():
            if kw in line:
                return True
    return False


def _all_ckpts_merged() -> tuple[bool, int]:
    n_ready = 0
    for hf_dir in E3_CKPT_HF_DIRS:
        if not os.path.isdir(hf_dir):
            continue
        # at least one safetensors file
        any_sf = any(f.endswith(".safetensors") for f in os.listdir(hf_dir))
        if any_sf:
            n_ready += 1
    return n_ready == len(E3_CKPT_HF_DIRS), n_ready


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    _log(f"auto_run_offline_eval_v2 started; need all {len(E3_CKPT_HF_DIRS)} ckpts merged + GPU idle")

    wait_iters = 0
    while True:
        all_ready, n_ready = _all_ckpts_merged()
        gpus_idle = max(_gpu_mem_used_mb()) < IDLE_MEM_THRESHOLD_MB
        busy = _busy_procs_alive()
        if all_ready and gpus_idle and not busy:
            break
        time.sleep(POLL_INTERVAL)
        wait_iters += 1
        if wait_iters % 10 == 0:
            mem = _gpu_mem_used_mb()
            _log(
                f"still waiting ({wait_iters * POLL_INTERVAL}s) — "
                f"merged={n_ready}/{len(E3_CKPT_HF_DIRS)} "
                f"gpu_max={max(mem)} busy_procs={busy}"
            )

    _log(f"all conditions met; sleeping {SETTLE_SLEEP_S}s for settle")
    time.sleep(SETTLE_SLEEP_S)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_log = f"{LOG_DIR}/eval_if_offline_chained_v2.{ts}.log"
    cmd = (
        f"setsid nohup bash {EVAL_SH} "
        f"< /dev/null > {out_log} 2>&1 & disown"
    )
    _log(f"launching eval_if_offline.sh; out_log={out_log}")
    rc = subprocess.run(["bash", "-c", cmd], cwd=ROOT).returncode
    if rc != 0:
        _log(f"launch shell exited non-zero: rc={rc}")
        return

    t0 = time.time()
    while time.time() - t0 < LAUNCH_VERIFY_DELAY:
        time.sleep(20)
        if max(_gpu_mem_used_mb()) > 4000:
            _log(f"eval verified — GPU mem rose")
            break
    else:
        _log("verification timed out; check eval log manually")
        return

    # Wait for offline eval to finish before plotting.
    # Heuristic: poll until eval_if_offline procs all gone AND GPUs idle.
    _log("waiting for offline eval to complete before triggering plots…")
    iters = 0
    while True:
        time.sleep(POLL_INTERVAL)
        iters += 1
        try:
            ps_out = subprocess.check_output(
                ["ps", "-eo", "cmd"], stderr=subprocess.DEVNULL,
            ).decode()
        except Exception:
            ps_out = ""
        eval_running = ("eval_if_offline" in ps_out) or ("VLLM::Worker_TP" in ps_out) or ("VLLM::EngineCore" in ps_out)
        if (not eval_running) and max(_gpu_mem_used_mb()) < IDLE_MEM_THRESHOLD_MB:
            _log("offline eval finished; GPUs idle")
            break
        if iters % 10 == 0:
            _log(f"  still evaluating ({iters * POLL_INTERVAL}s)")

    _log("offline evaluation complete; daemon exiting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("stopped by SIGINT")
        sys.exit(0)
    except Exception:
        _log("crashed:\n" + traceback.format_exc())
        sys.exit(1)
