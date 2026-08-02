#!/usr/bin/env python
"""WAVE 3 — merge per-shard f1_intervention parquets into a single parquet
and produce a numeric summary table.

For each condition, concat all shards into f1_intervention_aime__{condition}.parquet
Then merge ALL conditions into f1_intervention_aime.parquet.

Per-condition summary:
    - n_total, n_correct, pass@1
    - best@k for k in {1,2,4,8,16,32}
    - mode distribution {DRI, DAI, CSI, Other}
    - DAI rate, DRI rate (just renamed mode counts)
    - mean response chars / tokens

Outputs:
    analysis/data/f1_intervention_aime.parquet
    analysis/tables/c_summary.csv  (one row per condition)
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from math import lgamma

import numpy as np
import pandas as pd

REPO = os.environ.get("PROJECT_ROOT", ".")
F1_DIR = f"{REPO}/analysis/data/f1_intervention"
OUT_PARQUET = f"{REPO}/analysis/data/f1_intervention_aime.parquet"
OUT_TABLE = f"{REPO}/analysis/tables/c_summary.csv"


def best_at_k(accs: list, k: int) -> float:
    n = len(accs)
    c = sum(1 for a in accs if a)
    if k > n:
        return float(c > 0)
    if c == 0: return 0.0
    if c == n: return 1.0
    def lg_comb(a, b):
        if b < 0 or b > a: return float("-inf")
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    return float(1.0 - np.exp(lg_comb(n - c, k) - lg_comb(n, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1_dir", default=F1_DIR)
    ap.add_argument("--out_parquet", default=OUT_PARQUET)
    ap.add_argument("--out_table", default=OUT_TABLE)
    args = ap.parse_args()

    # Discover conditions and their shard parquets.
    files = sorted(glob(os.path.join(args.f1_dir, "*__shard*.parquet")))
    if not files:
        print(f"[merge] no shard files under {args.f1_dir}")
        return
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        df["src_file"] = os.path.basename(f)
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(args.out_parquet, index=False)
    print(f"[merge] wrote {args.out_parquet}  ({len(full)} rows, "
          f"{full['condition'].nunique()} conditions)")

    # Per (condition, prompt_id) compute best@k from acc booleans
    # then aggregate over prompts.
    summary_rows = []
    for cond, gc in full.groupby("condition"):
        # accs grouped by prompt
        per_prompt = []
        for pid, gp in gc.groupby("prompt_id"):
            accs = gp["acc"].astype(bool).tolist()
            row = {"prompt_id": pid, "n_samples": len(accs),
                   "pass@1": float(np.mean(accs))}
            for k in (1, 2, 4, 8, 16, 32):
                if k <= len(accs):
                    row[f"best@{k}"] = best_at_k(accs, k)
            per_prompt.append(row)
        pp = pd.DataFrame(per_prompt)
        # mode distribution
        mode_counts = gc["mode"].value_counts(normalize=True).to_dict()
        sample_acc = float(gc["acc"].mean())
        # primary tag for this condition
        primary = gc["primary_tag"].iloc[0]
        intervention = gc["intervention_tag"].iloc[0] if pd.notna(gc["intervention_tag"].iloc[0]) else None

        row = {
            "condition": cond,
            "primary_tag": primary,
            "intervention_tag": intervention,
            "force_prefix": gc["force_prefix"].iloc[0],
            "n_rows": len(gc),
            "n_prompts": len(pp),
            "sample_pass@1": sample_acc,
            "best@1": float(pp.get("best@1", pd.Series(dtype=float)).mean()),
            "best@8": float(pp.get("best@8", pd.Series(dtype=float)).mean()),
            "best@32": float(pp.get("best@32", pd.Series(dtype=float)).mean()),
            "DRI_rate": mode_counts.get("DRI", 0.0),
            "DAI_rate": mode_counts.get("DAI", 0.0),
            "CSI_rate": mode_counts.get("CSI", 0.0),
            "Other_rate": mode_counts.get("Other", 0.0),
            "mean_resp_chars": float(gc["response_chars"].mean()),
            "mean_n_resp_tokens": float(gc["n_resp_tokens"].mean()),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["primary_tag", "condition"]).reset_index(drop=True)
    summary.to_csv(args.out_table, index=False)
    print(f"[merge] wrote {args.out_table}")
    print()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
