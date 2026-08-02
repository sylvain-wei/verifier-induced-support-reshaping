#!/usr/bin/env python
"""WAVE 0.3 — build a rollout index over val_rollout / val_rollout_ifbench /
train_rollout / math_support_probe.

Output (default, lightweight):
  - analysis/data/rollout_index_summary.parquet
      one row per (root, model_tag, step, benchmark)
      columns: root, model_tag, step, benchmark, n_rows, n_unique_prompts,
               n_samples_per_prompt, file_path, file_mtime, file_size_mb

This is the "index" that downstream scripts join against to know which
(model, dataset, step) trajectories exist and where to load them from.

Optionally also writes a full per-row index (one row per (model, step, bench,
prompt, sample)) with `--full`; this can get large (~10s of millions of rows)
so it's gated.

Also reports missing (model, dataset) coverage vs the lineage we need for
WAVE 1 (Block A):
    B_q3, M_q3, I_q3, B_q25m, M_q25m, I_q25m × {AIME, IFEval-100, IFBench-100}.
Note: B_q3 / B_q25m have no training rollout per se — they correspond to the
step-0 ckpt rollouts of any *r1 run, which we surface separately.

Usage:
  python build_rollout_index.py                   # summary only
  python build_rollout_index.py --full            # also full per-row index
  python build_rollout_index.py --coverage_only   # skip rebuild, just print
                                                  # which (model, ds) pairs are
                                                  # missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import List, Tuple

import pandas as pd


ROOT = os.environ.get("PROJECT_ROOT", ".")
AIME_MARKER = "Solve the following math problem step by step"

# (root_label, on-disk path). The first two are always scanned (WAVE 1 inputs);
# the last two are opt-in via --include train_rollout / --include math_support_probe.
ROLLOUT_ROOTS = [
    ("val_rollout",         f"{ROOT}/rollout/val_rollout/DAPO_sh_repro"),
    ("val_rollout_ifbench", f"{ROOT}/rollout/val_rollout_ifbench"),
]
OPTIONAL_ROOTS = [
    ("train_rollout",       f"{ROOT}/rollout/train_rollout/DAPO_sh_repro"),
    ("math_support_probe",  f"{ROOT}/rollout/math_support_probe"),
]

SUMMARY_OUT  = f"{ROOT}/analysis/data/rollout_index_summary.parquet"
FULL_OUT     = f"{ROOT}/analysis/data/rollout_index_full.parquet"

# ---------------------------------------------------------------------------
# What WAVE 1 needs (Block A). model_tag → expected datasets.
# ---------------------------------------------------------------------------
WAVE1_COVERAGE_NEEDS = {
    # Qwen3-8B lineage
    "B_q3   (Qwen3-8B-Base)":                {"aime", "ifeval", "ifbench"},
    "M_q3   (b1r1_Qwen3-8B-Base_math7.5k step 220)":  {"aime", "ifeval"},
    "I_q3   (b2r1_Qwen3-8B-Base_IFTrain step 100)":   {"aime", "ifeval", "ifbench"},
    # Qwen2.5-Math lineage
    "B_q25m (Qwen2.5-Math-7B)":              {"aime", "ifeval"},
    "M_q25m (b1r1_Qwen2.5-math-7B step 470)":         {"aime", "ifeval"},
    "I_q25m (b2r1_Qwen2.5-math-7B step 390)":         {"aime", "ifeval"},
}

# Map run_name → (lineage_role, base_model). Used to fill the WAVE-1 coverage
# matrix. `step_0` of any *r1 run = the base model rollout.
RUN_TO_ROLE = {
    "b1r1_Qwen3-8B-Base_math7.5k_local_H20":    ("M_q3",   "Qwen3-8B-Base"),
    "b1r1_Qwen2.5-math-7B_math7.5k_local_H20":  ("M_q25m", "Qwen2.5-Math-7B"),
    "b2r1_Qwen3-8B-Base_IFTrain_local_H20":     ("I_q3",   "Qwen3-8B-Base"),
    "b2r1_Qwen3-8B-Base_IFTrain_clip_high_0.2_local_H20":
                                                ("I_q3_clip", "Qwen3-8B-Base"),
    "b2r1_Qwen2.5-math-7B_IFTrain_local_H20":   ("I_q25m", "Qwen2.5-Math-7B"),
    "b3r1_Qwen2.5-math-7B_math2if_local_H20":   ("b3r1_q25m", "Qwen2.5-Math-7B"),
    "b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20":
                                                ("b4r1_q3",   "Qwen3-8B-Base"),
    "b4r1_Qwen2.5-Math-7B_math7.5k_if2math_local_H20":
                                                ("b4r1_q25m", "Qwen2.5-Math-7B"),
}


# ---------------------------------------------------------------------------
# Scan helpers.
# ---------------------------------------------------------------------------
_STEP_FILE_RE = re.compile(r"^(\d+)\.jsonl$")


def _list_runs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def _list_step_files(run_dir: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(run_dir):
        return out
    for fn in os.listdir(run_dir):
        m = _STEP_FILE_RE.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(run_dir, fn)))
    return sorted(out)


def _classify_bench(input_text: str, root_label: str) -> str:
    if root_label == "val_rollout_ifbench":
        return "ifbench"
    if root_label == "train_rollout":
        return "train"
    if root_label == "math_support_probe":
        return "math_probe"
    # val_rollout: AIME marker vs everything else
    if AIME_MARKER in input_text:
        return "aime"
    return "ifeval"


def _hash_prompt(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def scan_file(path: str, root_label: str, run_name: str, step: int,
              keep_full: bool = False):
    """Walk one jsonl. Return (summary_rows, full_rows).

    If `keep_full` is False (default), we DO NOT json-parse each row — we just
    classify the bench from the prefix bytes (`"input"` always comes first in
    these jsonl files; we look for the AIME marker and the first ~1000 bytes
    only). This makes a 6-min scan a 30-second one.

    If `keep_full` is True, we fall back to full json.loads + per-row dict so
    that `score`, `acc`, `output_len` get cached.
    """
    n_rows_per_bench: Counter = Counter()
    unique_prompts: defaultdict = defaultdict(set)
    samples_per_prompt: defaultdict = defaultdict(Counter)  # bench -> Counter(pid)
    full_rows = []

    fsize_mb = os.path.getsize(path) / 1024 / 1024
    fmtime   = os.path.getmtime(path)

    if not keep_full:
        # Fast path: count rows + classify bench by AIME marker presence in
        # first ~2KB of each line. We don't need a unique prompt count for
        # the fast path either (set membership over millions of strings is
        # expensive); instead we report `n_unique_prompts = -1` meaning
        # "uncomputed" — the file's row count is enough for downstream
        # planning.
        aime_bytes = AIME_MARKER.encode("utf-8")
        with open(path, "rb") as fb:
            for raw in fb:
                head = raw[:2048]
                if root_label == "val_rollout_ifbench":
                    bench = "ifbench"
                elif root_label == "train_rollout":
                    bench = "train"
                elif root_label == "math_support_probe":
                    bench = "math_probe"
                elif aime_bytes in head:
                    bench = "aime"
                else:
                    bench = "ifeval"
                n_rows_per_bench[bench] += 1
        # build summary without unique-prompt info
        summary_rows = []
        for bench, n in n_rows_per_bench.items():
            summary_rows.append({
                "root": root_label,
                "run_name": run_name,
                "model_role": RUN_TO_ROLE.get(run_name, (run_name, ""))[0],
                "base_model": RUN_TO_ROLE.get(run_name, ("", ""))[1],
                "step": step,
                "benchmark": bench,
                "n_rows": n,
                "n_unique_prompts": -1,
                "n_samples_per_prompt": -1,
                "file_path": path,
                "file_mtime": fmtime,
                "file_size_mb": round(fsize_mb, 2),
            })
        return summary_rows, []

    # ----- slow path: full json.loads + per-row dict -----
    with open(path) as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            input_text = o.get("input", "")
            bench = _classify_bench(input_text, root_label)
            pid   = _hash_prompt(input_text)
            n_rows_per_bench[bench] += 1
            unique_prompts[bench].add(pid)
            samples_per_prompt[bench][pid] += 1
            full_rows.append({
                "root": root_label,
                "run_name": run_name,
                "model_role": RUN_TO_ROLE.get(run_name, (run_name, ""))[0],
                "base_model": RUN_TO_ROLE.get(run_name, ("", ""))[1],
                "step": step,
                "benchmark": bench,
                "prompt_id": pid,
                "sample_id": samples_per_prompt[bench][pid] - 1,
                "score": float(o.get("score", 0.0)),
                "acc":   o.get("acc"),
                "resp_len_chars": len(o.get("output", "")),
            })

    summary_rows = []
    for bench, n in n_rows_per_bench.items():
        nu = len(unique_prompts[bench])
        spp = sum(samples_per_prompt[bench].values()) / max(1, nu)
        summary_rows.append({
            "root": root_label,
            "run_name": run_name,
            "model_role": RUN_TO_ROLE.get(run_name, (run_name, ""))[0],
            "base_model": RUN_TO_ROLE.get(run_name, ("", ""))[1],
            "step": step,
            "benchmark": bench,
            "n_rows": n,
            "n_unique_prompts": nu,
            "n_samples_per_prompt": round(spp, 2),
            "file_path": path,
            "file_mtime": fmtime,
            "file_size_mb": round(fsize_mb, 2),
        })
    return summary_rows, full_rows


def build_index(write_full: bool, include_optional: List[str]):
    all_summary = []
    all_full    = []
    roots_to_scan = list(ROLLOUT_ROOTS)
    for name, path in OPTIONAL_ROOTS:
        if name in include_optional:
            roots_to_scan.append((name, path))
    print(f"[index] roots: {[r[0] for r in roots_to_scan]}  "
          f"(use --include train_rollout / math_support_probe for others)")

    t0 = time.time()
    for root_label, root_path in roots_to_scan:
        runs = _list_runs(root_path)
        if not runs:
            print(f"[index] (skip) {root_path} — empty or missing")
            continue
        print(f"[index] scanning {root_label}: {len(runs)} runs")
        for run in runs:
            files = _list_step_files(os.path.join(root_path, run))
            t_run = time.time()
            n_rows = 0
            for step, path in files:
                summary_rows, full_rows = scan_file(path, root_label, run, step,
                                                    keep_full=write_full)
                all_summary.extend(summary_rows)
                if write_full:
                    all_full.extend(full_rows)
                n_rows += sum(r["n_rows"] for r in summary_rows)
            elapsed = time.time() - t_run
            print(f"  - {run}: {len(files)} files, {n_rows} rows, "
                  f"{elapsed:.1f}s")

    summary_df = pd.DataFrame(all_summary)
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    summary_df.to_parquet(SUMMARY_OUT, index=False)
    print(f"[index] wrote {SUMMARY_OUT}  ({len(summary_df)} rows, "
          f"{time.time()-t0:.1f}s total)")

    if write_full:
        full_df = pd.DataFrame(all_full)
        full_df.to_parquet(FULL_OUT, index=False)
        print(f"[index] wrote {FULL_OUT}  ({len(full_df)} rows)")

    return summary_df


# ---------------------------------------------------------------------------
# WAVE 1 coverage report.
# ---------------------------------------------------------------------------
def report_coverage(summary_df: pd.DataFrame):
    """Cross-check WAVE 1 coverage needs against what's on disk."""
    print("\n=== WAVE 1 coverage check (need AIME / IFEval / IFBench rollouts "
          "per model role) ===")

    # For each lineage role, find the step we want and check which benches
    # are covered. For B_q3 / B_q25m, we accept "step 0 of any *r1 run" or
    # an explicit "_base" / "_b0" run if present.
    wanted_steps = {
        "M_q3":   220,
        "I_q3":   100,
        "M_q25m": 470,
        "I_q25m": 390,
    }
    # For base models, surface step-0 rollouts of *r1 runs.
    base_step0_sources = {
        "B_q3":   {"M_q3", "I_q3", "I_q3_clip", "b4r1_q3"},
        "B_q25m": {"M_q25m", "I_q25m", "b3r1_q25m", "b4r1_q25m"},
    }

    rows = []
    for role, need_benches in WAVE1_COVERAGE_NEEDS.items():
        role_key = role.split()[0]
        if role_key in wanted_steps:
            sub = summary_df[(summary_df["model_role"] == role_key) &
                             (summary_df["step"] == wanted_steps[role_key])]
            have = set(sub["benchmark"].unique())
        elif role_key in base_step0_sources:
            srcs = base_step0_sources[role_key]
            sub = summary_df[(summary_df["model_role"].isin(srcs)) &
                             (summary_df["step"] == 0)]
            have = set(sub["benchmark"].unique())
        else:
            have = set()

        missing = need_benches - have
        extra   = have - need_benches
        rows.append({
            "role": role,
            "need":     sorted(need_benches),
            "have":     sorted(have & need_benches),
            "missing":  sorted(missing),
            "extra":    sorted(extra),
        })

    for r in rows:
        flag = "[OK]" if not r["missing"] else "[MISSING]"
        print(f"  {flag:10s} {r['role']:55s}"
              f" need={r['need']}  have={r['have']}"
              + (f"  MISSING={r['missing']}" if r['missing'] else ""))

    # Also print a sanity matrix of (lineage role × step) row counts on AIME.
    print("\n=== AIME row counts per (model_role, step) — sanity ===")
    aime = summary_df[summary_df["benchmark"] == "aime"]
    if len(aime):
        piv = aime.pivot_table(index="model_role", columns="step",
                               values="n_rows", aggfunc="sum",
                               fill_value=0)
        print(piv.to_string())


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also write per-row index (rollout_index_full.parquet)")
    ap.add_argument("--include", action="append", default=[],
                    choices=["train_rollout", "math_support_probe"],
                    help="opt-in optional rollout root (slow, can repeat). "
                         "Default scans val_rollout + val_rollout_ifbench only.")
    ap.add_argument("--coverage_only", action="store_true",
                    help="skip rebuild; load existing summary and report coverage")
    args = ap.parse_args()

    if args.coverage_only:
        if not os.path.exists(SUMMARY_OUT):
            print(f"[index] ERROR: no existing summary at {SUMMARY_OUT}")
            return
        summary_df = pd.read_parquet(SUMMARY_OUT)
        print(f"[index] loaded {SUMMARY_OUT}  ({len(summary_df)} rows)")
    else:
        summary_df = build_index(write_full=args.full,
                                 include_optional=args.include)

    report_coverage(summary_df)


if __name__ == "__main__":
    main()
