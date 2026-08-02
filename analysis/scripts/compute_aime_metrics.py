#!/usr/bin/env python
"""Compute AIME metrics across all (run, step).

For each (run, step):
  - per-prompt pass_count (out of 32 rollouts)
  - per-prompt pass@1, best@32, maj@32
  - aggregate to per-step pass@1 / best@32 / maj@32 / mean_len
  - per-prompt rows for downstream support-shift analysis (Finding 1 + 2)

maj@32 is computed from `pred` field (extracted answer string). The prompt's
maj@32 = 1 iff the most-frequent extracted answer matches the ground-truth
answer (i.e. some rollout in the majority bucket has acc=True). We approximate
"matches" by: among rollouts whose `pred` equals the modal `pred`, if any has
acc=True, return 1; else 0. This is robust to formatting noise.

Outputs:
  analysis/tables/aime_metrics_per_step.csv     (run, step, pass@1, best@32, maj@32, mean_len, n_prompts)
  analysis/tables/aime_metrics_per_prompt.csv   (run, step, prompt_id, pass_count, pass@1, best@32, maj@32, modal_pred, ...)
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_rollouts import RUNS, load_run, list_run_steps  # noqa

OUT_TABLES = f"{ROOT}/analysis/tables"


def _maj_per_prompt(g: pd.DataFrame) -> int:
    preds = list(g["extracted_answer"].fillna("[INVALID]").astype(str))
    if not preds:
        return 0
    cnt = Counter(preds)
    # exclude [INVALID] from majority if a non-invalid majority exists
    valid_cnt = Counter({p: c for p, c in cnt.items() if p != "[INVALID]"})
    if valid_cnt:
        modal = max(valid_cnt, key=valid_cnt.get)
    else:
        modal = max(cnt, key=cnt.get)
    # any rollout with this pred and acc=True means the majority answer is correct
    sub = g[g["extracted_answer"].fillna("[INVALID]").astype(str) == modal]
    return int(bool(sub["acc"].any()))


def aggregate_per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, step, pid), g in df.groupby(["run_name", "checkpoint_step", "prompt_id"]):
        n = len(g)
        pass_count = int(g["acc"].sum())
        # for AIME, score is ±1.0 — but acc is the canonical bool
        rows.append({
            "run_name": run, "checkpoint_step": step, "prompt_id": pid,
            "n": n,
            "pass_count": pass_count,
            "pass@1": pass_count / max(1, n),
            "best@32": int(pass_count > 0),
            "maj@32": _maj_per_prompt(g),
            "mean_len_chars": float(g["response_length_chars"].mean()),
        })
    return pd.DataFrame(rows)


def aggregate_per_step(per_prompt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, step), g in per_prompt.groupby(["run_name", "checkpoint_step"]):
        rows.append({
            "run_name": run, "checkpoint_step": step,
            "n_prompts": len(g),
            "pass@1":  float(g["pass@1"].mean()),
            "best@32": float(g["best@32"].mean()),
            "maj@32":  float(g["maj@32"].mean()),
            "mean_len_chars": float(g["mean_len_chars"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["run_name", "checkpoint_step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--out_step", default=f"{OUT_TABLES}/aime_metrics_per_step.csv")
    ap.add_argument("--out_prompt", default=f"{OUT_TABLES}/aime_metrics_per_prompt.csv")
    args = ap.parse_args()
    os.makedirs(OUT_TABLES, exist_ok=True)

    runs = (args.runs.split(",") if args.runs else list(RUNS.keys()))
    steps_filter = ([int(s) for s in args.steps.split(",")] if args.steps else None)

    all_rows = []
    for run in runs:
        steps = steps_filter or list_run_steps(run, "val_rollout")
        if not steps:
            print(f"[skip] {run}: no val_rollout dir / files")
            continue
        df = load_run(run, kind="val_rollout", steps=steps)
        if df.empty:
            continue
        df = df[df["benchmark"] == "aime"]
        print(f"[{run}] {len(df)} AIME rows over {df['checkpoint_step'].nunique()} steps")
        all_rows.append(df)
    if not all_rows:
        print("[error] no AIME rollouts loaded")
        return
    big = pd.concat(all_rows, ignore_index=True)

    per_prompt = aggregate_per_prompt(big)
    per_prompt.to_csv(args.out_prompt, index=False)
    print(f"[csv] {args.out_prompt} ({len(per_prompt)} rows)")

    step_df = aggregate_per_step(per_prompt)
    step_df.to_csv(args.out_step, index=False)
    print(f"[csv] {args.out_step} ({len(step_df)} rows)")
    # quick look
    pd.set_option("display.max_rows", None)
    print()
    print(step_df.to_string(index=False))


if __name__ == "__main__":
    main()
