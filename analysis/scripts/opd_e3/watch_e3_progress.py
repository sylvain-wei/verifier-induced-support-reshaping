#!/usr/bin/env python3
"""watch_e3_progress.py — long-running watcher for the E3 OPD training run.

What it does
------------
Every POLL_INTERVAL seconds:
  1. Re-discovers the most recent E3 training log under
     analysis/scripts/opd_e3/logs/opd_e3_step{20,40,60}_*.log.
  2. Scans the entire log (idempotent) for:
       * `step:N -` validation dumps (N = 0, 25, 50, 75, 100)  → metric extraction
       * "AssertionError" / "RayTaskError" / "Traceback (most recent" / "Error executing job" → crash detection
  3. For every newly-seen val dump it appends a row to:
       - tables/e3_val_metrics.csv  (machine-readable; one row per (run, step, dataset))
       - findings_opd_E3_progress.md (human-readable; one section per val event)
     Both files are append-only and idempotent (skips events already recorded).
  4. For every crash signature it appends to findings_opd_E3_progress.md
     and writes a short alert file at logs/CRASH_DETECTED_<ts>.txt
     so the next codebuddy session can grep for it.

Usage
-----
Foreground (debug):
    python3 analysis/scripts/opd_e3/watch_e3_progress.py

Detached (recommended; survives codebuddy turn boundaries):
    setsid nohup python3 analysis/scripts/opd_e3/watch_e3_progress.py \
        < /dev/null > analysis/scripts/opd_e3/logs/watcher.out 2>&1 & disown

Stop:
    pkill -f watch_e3_progress.py
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import os
import re
import sys
import time
import traceback

ROOT = os.environ.get("PROJECT_ROOT", ".")
LOG_GLOB = f"{ROOT}/analysis/scripts/opd_e3/logs/opd_e3_step*_2*.log"
TRAIN_BOOT_GLOB = f"{ROOT}/analysis/scripts/opd_e3/logs/opd_e3_step*.boot.*.log"
CSV_PATH = f"{ROOT}/analysis/tables/e3_val_metrics.csv"
MD_PATH = f"{ROOT}/analysis/findings_opd_E3_progress.md"
ALERT_DIR = f"{ROOT}/analysis/scripts/opd_e3/logs"
STATE_PATH = f"{ROOT}/analysis/scripts/opd_e3/logs/watcher_state.txt"
SELF_LOG = f"{ROOT}/analysis/scripts/opd_e3/logs/watcher.log"

POLL_INTERVAL = 60          # seconds between scans
CSV_HEADER = [
    "run", "global_step", "dataset",
    "acc_mean@16", "acc_best@16", "acc_maj@16", "acc_worst@16",
    "score_mean@16", "logged_at_utc",
]
DATASETS = ("aime24", "aime25", "math_dapo", "ifeval_test", "ifbench_test")
WATCHED_STEPS = (0, 25, 50, 75, 100)

# Regex helpers -------------------------------------------------------------

# Match each metric inside a `step:N - ...` dump, e.g.:
#   val-core/aime24/acc/mean@16:0.0312
_MET_RE = re.compile(
    r"val-core/([a-zA-Z0-9_]+)/(acc|score)/(mean@16|best@16/mean|maj@16/mean|worst@16/mean):([-\d.]+)"
)
# Match the "step:N -" prefix; capture N.
_STEP_RE = re.compile(r"step:(\d+)\s+-\s+")
# Crash patterns we actively look for.
_CRASH_PATTERNS = (
    "AssertionError",
    "RayTaskError",
    "Error executing job with overrides",
    "CUDA out of memory",
    "RuntimeError: ",
    "TypeError: ",
)
# Run name from log path: opd_e3_step20_20260522_141435.log → "step20"
_RUN_FROM_LOG = re.compile(r"opd_e3_(step\d+)_\d+_\d+\.log$")


def _utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _self_log(msg: str) -> None:
    line = f"[{_utcnow()}] {msg}\n"
    try:
        with open(SELF_LOG, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


def _ensure_outputs() -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    os.makedirs(ALERT_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)
    if not os.path.isfile(MD_PATH):
        with open(MD_PATH, "w") as fh:
            fh.write(
                "# E3 OPD progress — live watcher\n\n"
                "Auto-appended by `analysis/scripts/opd_e3/watch_e3_progress.py`. "
                "One section per validation event (step 0/25/50/75/100) per run.\n\n"
                "Crash markers prefixed with `🚨 CRASH`.\n\n"
                "---\n\n"
            )


def _load_seen() -> set[str]:
    """Return the set of (run, step) keys already recorded in CSV_PATH.

    Reading from CSV is the source of truth so we never double-write a metric
    row. `STATE_PATH` is just a low-precision hint for crash de-duplication.
    """
    seen: set[str] = set()
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                seen.add(f"{row['run']}::{row['global_step']}")
    return seen


def _load_seen_crashes() -> set[str]:
    if not os.path.isfile(STATE_PATH):
        return set()
    with open(STATE_PATH) as fh:
        return {l.strip() for l in fh if l.strip().startswith("crash::")}


def _save_seen_crash(key: str) -> None:
    with open(STATE_PATH, "a") as fh:
        fh.write(f"crash::{key}\n")


def _newest_train_log() -> str | None:
    files = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def _run_label(log_path: str) -> str:
    m = _RUN_FROM_LOG.search(log_path)
    return m.group(1) if m else os.path.basename(log_path)


def _extract_step_dump(line: str, run: str) -> tuple[int, dict[str, dict[str, float]]] | None:
    """If `line` is a `step:N -` dump, return (N, {dataset: {key: value}})."""
    sm = _STEP_RE.search(line)
    if not sm:
        return None
    step = int(sm.group(1))
    if step not in WATCHED_STEPS:
        return None
    out: dict[str, dict[str, float]] = {}
    for ds, kind, stat, val in _MET_RE.findall(line):
        try:
            v = float(val)
        except ValueError:
            continue
        # Normalise stat key: "mean@16" -> "mean@16",
        # "best@16/mean" -> "best@16", etc.
        stat_key = stat.split("/")[0] if "/mean" in stat else stat
        out.setdefault(ds, {})[f"{kind}_{stat_key}"] = v
    if not out:
        return None
    return step, out


def _record_val(run: str, step: int, per_ds: dict[str, dict[str, float]]) -> None:
    """Append CSV + Markdown for one (run, step) val event."""
    now = _utcnow()

    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        for ds in DATASETS:
            r = per_ds.get(ds, {})
            # Skip datasets that aren't in this val dump — happens when
            # training-time val_files only covers a subset (e.g. E3 reruns
            # use math-only val; IFEval-test/IFBench are evaluated offline
            # via eval_if_offline.{py,sh}). Without this, the CSV would get
            # spurious 0.0 placeholder rows that look like the model scored 0.
            if not r:
                continue
            w.writerow([
                run, step, ds,
                r.get("acc_mean@16", ""),
                r.get("acc_best@16", ""),
                r.get("acc_maj@16", ""),
                r.get("acc_worst@16", ""),
                r.get("score_mean@16", ""),
                now,
            ])

    # Markdown table
    md = [
        f"## {run} — step {step} ({now})\n",
        "| dataset | acc mean@16 | acc best@16 | acc maj@16 | score mean@16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        r = per_ds.get(ds, {})
        def fmt(k: str) -> str:
            return f"{r[k]:.4f}" if k in r else "—"
        md.append(
            f"| {ds} | {fmt('acc_mean@16')} | {fmt('acc_best@16')} "
            f"| {fmt('acc_maj@16')} | {fmt('score_mean@16')} |"
        )
    md.append("")
    with open(MD_PATH, "a") as fh:
        fh.write("\n".join(md) + "\n")

    _self_log(f"recorded val event run={run} step={step} datasets={list(per_ds)}")


def _record_crash(run: str, line_no: int, line: str, log_path: str) -> None:
    key = f"{run}::{line_no}::{line[:80]}"
    snippet_path = f"{ALERT_DIR}/CRASH_DETECTED_{run}_line{line_no}.txt"
    if not os.path.isfile(snippet_path):
        with open(snippet_path, "w") as fh:
            fh.write(f"crash detected at {_utcnow()}\n")
            fh.write(f"log: {log_path}\n")
            fh.write(f"line {line_no}: {line}\n")
            try:
                with open(log_path) as src:
                    all_lines = src.read().splitlines()
                lo = max(0, line_no - 30)
                hi = min(len(all_lines), line_no + 5)
                fh.write(f"\n--- context [{lo}:{hi}] ---\n")
                fh.write("\n".join(all_lines[lo:hi]))
            except Exception:
                pass
    md = [
        f"## 🚨 CRASH — {run} ({_utcnow()})\n",
        f"- log: `{log_path}`",
        f"- line {line_no}: `{line[:200]}`",
        f"- context dumped to: `{snippet_path}`",
        "",
    ]
    with open(MD_PATH, "a") as fh:
        fh.write("\n".join(md) + "\n")
    _save_seen_crash(key)
    _self_log(f"CRASH detected run={run} line={line_no}: {line[:120]}")


def scan_once(seen_val: set[str], seen_crash: set[str]) -> None:
    log = _newest_train_log()
    if log is None:
        return
    run = _run_label(log)
    try:
        with open(log) as fh:
            content = fh.read()
    except Exception as e:
        _self_log(f"failed to read {log}: {e}")
        return

    lines = content.splitlines()
    for i, line in enumerate(lines):
        # 1) val dumps
        if "step:" in line and " - " in line:
            res = _extract_step_dump(line, run)
            if res is not None:
                step, per_ds = res
                key = f"{run}::{step}"
                if key not in seen_val:
                    _record_val(run, step, per_ds)
                    seen_val.add(key)
        # 2) crashes (cheap substring scan)
        if any(p in line for p in _CRASH_PATTERNS):
            crash_key = f"{run}::{i}::{line[:80]}"
            if crash_key not in seen_crash:
                _record_crash(run, i, line, log)
                seen_crash.add(crash_key)


def main() -> None:
    _ensure_outputs()
    seen_val = _load_seen()
    seen_crash = _load_seen_crashes()
    _self_log(f"watcher started; seen_val={len(seen_val)} seen_crash={len(seen_crash)}")
    _self_log(f"watching log glob: {LOG_GLOB}")
    _self_log(f"poll interval: {POLL_INTERVAL}s")
    while True:
        try:
            scan_once(seen_val, seen_crash)
        except Exception:
            _self_log("scan_once raised:\n" + traceback.format_exc())
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _self_log("watcher stopped by SIGINT")
        sys.exit(0)
