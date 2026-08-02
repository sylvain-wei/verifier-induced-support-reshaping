#!/usr/bin/env python
"""h11 — Indicator A: aggregate routing-position JS for the predictive
trajectory analysis (§5.6).

Reads every analysis/data/wave5_predictive_js/B_q3__{rl_tag}__{ds}.parquet
where rl_tag follows `I_q3_step{N}_{label}` (label ∈ vanilla/S1soft/S2hard/
negDAI/negRand). For each (label, step, dataset) we compute:

    js_pos1_mean       — mean(js | t == 1)
    js_pos1_median     — median(js | t == 1)
    js_interior_mean   — mean(js | 5 <= t and rel_pos < 0.9)
    pos1_ratio         — js_pos1_mean / js_interior_mean
    js_pos2_mean       — mean(js | t == 2)
    js_pos5_mean       — mean(js | t == 5)

Output: analysis/tables/h11_predictive_routing_js.csv
        (long format: one row per (run, step, dataset))

Backward-compat: if any downstream still wants the old wide form keyed by
`run = I_q3_step20_*`, the same row is also emitted with `legacy_run` set,
so existing h11_predictive_correlation.py keeps working.
"""
from __future__ import annotations
import os

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
JS_DIR = ROOT / "analysis/data/wave5_predictive_js"
OUT_CSV = ROOT / "analysis/tables/h11_predictive_routing_js.csv"

# Maps short label → display "run" name we use everywhere else.
LABEL2RUN = {
    "vanilla": "vanilla_I_q3",
    "S1soft":  "S1_soft",
    "S2hard":  "S2_hard",
    "negDAI":  "negctl_DAI",
    "negRand": "negctl_random",
}

TAG_RE = re.compile(r"^I_q3_step(\d+)_([A-Za-z0-9]+)$")


def aggregate_one(df: pd.DataFrame) -> dict:
    pos1 = df[df["t"] == 1]
    pos2 = df[df["t"] == 2]
    pos5 = df[df["t"] == 5]
    interior = df[(df["t"] >= 5) & (df["rel_pos"] < 0.9)]
    out = {
        "js_pos1_mean": float(pos1["js"].mean()) if len(pos1) else np.nan,
        "js_pos1_median": float(pos1["js"].median()) if len(pos1) else np.nan,
        "js_pos1_std": float(pos1["js"].std()) if len(pos1) else np.nan,
        "js_pos2_mean": float(pos2["js"].mean()) if len(pos2) else np.nan,
        "js_pos5_mean": float(pos5["js"].mean()) if len(pos5) else np.nan,
        "js_interior_mean": (float(interior["js"].mean())
                             if len(interior) else np.nan),
        "n_pos1": int(len(pos1)),
        "n_interior": int(len(interior)),
        "n_total_tokens": int(len(df)),
    }
    if out["js_interior_mean"] and out["js_interior_mean"] > 0:
        out["pos1_ratio"] = out["js_pos1_mean"] / out["js_interior_mean"]
    else:
        out["pos1_ratio"] = np.nan
    return out


def main():
    rows = []
    paths = sorted(JS_DIR.glob("B_q3__I_q3_step*__*.parquet"))
    if not paths:
        print(f"[h11] no parquet under {JS_DIR}")
        return
    for p in paths:
        # filename pattern: B_q3__I_q3_step{N}_{label}__{ds}.parquet
        stem = p.stem  # B_q3__I_q3_step40_S1soft__aime
        try:
            _, rl_tag, ds = stem.split("__")
        except ValueError:
            print(f"[h11] cannot parse {stem}")
            continue
        m = TAG_RE.match(rl_tag)
        if not m:
            print(f"[h11] cannot parse rl_tag {rl_tag}")
            continue
        step = int(m.group(1))
        label = m.group(2)
        run = LABEL2RUN.get(label, label)

        df = pd.read_parquet(p)
        agg = aggregate_one(df)
        agg["run"] = run
        agg["step"] = step
        agg["dataset"] = ds
        agg["legacy_run"] = rl_tag  # for backward-compat with h11_correlation
        rows.append(agg)
        print(f"[h11] {rl_tag} × {ds}: {len(df):,} tokens, "
              f"pos1={agg['js_pos1_mean']:.4f}  ratio={agg['pos1_ratio']:.1f}")

    out_df = pd.DataFrame(rows).sort_values(["run", "dataset", "step"])
    cols = ["run", "step", "dataset", "legacy_run",
            "js_pos1_mean", "js_pos1_median", "js_pos1_std",
            "js_pos2_mean", "js_pos5_mean",
            "js_interior_mean", "pos1_ratio",
            "n_pos1", "n_interior", "n_total_tokens"]
    out_df = out_df[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[h11] wrote {OUT_CSV} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
