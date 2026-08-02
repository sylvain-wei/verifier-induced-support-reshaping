#!/usr/bin/env python
"""h11 — Trajectory-derived predictors of cross-task collapse.

Reads the long-form `analysis/tables/h11_predictive_routing_js.csv`
(one row per run × step × dataset, written by h11_predictive_routing_js.py)
and derives **early-trajectory** predictors per (run, dataset):

    early_slope_20_40   — (ratio @ step40 − ratio @ step20) / 20
                          (units: ratio per RL step). Captures "how fast
                          routing concentration grows in the first quarter
                          of training". Higher = more aggressive routing edit.

    early_auc_20_40     — trapezoidal AUC of pos1/interior ratio over
                          step ∈ [20, 40], divided by the interval (= mean).
                          Less sensitive to noise at single steps.

    log_early_slope     — sign-preserving log of early_slope (since ratio
                          can vary by an order of magnitude across runs).

    pos1_step20         — alias for ratio @ step 20 (single-step baseline,
                          for comparison with the new trajectory-based metric).

    pos1_step40         — alias for ratio @ step 40.

    js_pos1_step20      — raw js_pos1 mean @ step 20 (alias).
    js_pos1_step40      — raw js_pos1 mean @ step 40 (alias).
    js_pos1_slope_20_40 — (js_pos1 @ 40 − js_pos1 @ 20) / 20 (raw JS slope).

Output: analysis/tables/h11_trajectory_features.csv
        (one row per (run, dataset))
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
IN_CSV = ROOT / "analysis/tables/h11_predictive_routing_js.csv"
OUT_CSV = ROOT / "analysis/tables/h11_trajectory_features.csv"


def signed_log10(x: float) -> float:
    """log10(|x|) with sign preserved; 0 → NaN."""
    if x is None or np.isnan(x) or x == 0:
        return np.nan
    return np.sign(x) * np.log10(abs(x))


def trajectory_features(df: pd.DataFrame) -> dict:
    """Given subset of df for one (run, dataset), compute features."""
    out = {}
    by_step = df.set_index("step").sort_index()

    # Single-step references
    for s in [20, 40, 60, 80, 100]:
        if s in by_step.index:
            out[f"pos1_step{s}"] = float(by_step.loc[s, "pos1_ratio"])
            out[f"js_pos1_step{s}"] = float(by_step.loc[s, "js_pos1_mean"])
            out[f"js_interior_step{s}"] = float(by_step.loc[s, "js_interior_mean"])
        else:
            out[f"pos1_step{s}"] = np.nan
            out[f"js_pos1_step{s}"] = np.nan
            out[f"js_interior_step{s}"] = np.nan

    # Early slope step 20 -> 40
    if 20 in by_step.index and 40 in by_step.index:
        r20 = float(by_step.loc[20, "pos1_ratio"])
        r40 = float(by_step.loc[40, "pos1_ratio"])
        out["early_slope_20_40"] = (r40 - r20) / 20.0
        out["log_early_slope"] = signed_log10(out["early_slope_20_40"])
        out["early_auc_20_40"] = (r20 + r40) / 2.0  # trapezoidal mean
        # Raw JS slope (no normalization)
        j20 = float(by_step.loc[20, "js_pos1_mean"])
        j40 = float(by_step.loc[40, "js_pos1_mean"])
        out["js_pos1_slope_20_40"] = (j40 - j20) / 20.0
        out["js_pos1_auc_20_40"]   = (j20 + j40) / 2.0
    else:
        for k in ["early_slope_20_40", "log_early_slope", "early_auc_20_40",
                  "js_pos1_slope_20_40", "js_pos1_auc_20_40"]:
            out[k] = np.nan

    # Steps available (string for transparency)
    steps_avail = sorted(by_step.index.tolist())
    out["steps_available"] = ",".join(str(s) for s in steps_avail)
    out["n_steps_available"] = len(steps_avail)

    return out


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(IN_CSV)

    df = pd.read_csv(IN_CSV)
    rows = []
    for (run, dataset), sub in df.groupby(["run", "dataset"]):
        feat = trajectory_features(sub)
        feat["run"] = run
        feat["dataset"] = dataset
        rows.append(feat)

    out_df = pd.DataFrame(rows)
    cols = ["run", "dataset",
            "pos1_step20", "pos1_step40", "pos1_step60", "pos1_step80", "pos1_step100",
            "js_pos1_step20", "js_pos1_step40", "js_pos1_step60", "js_pos1_step80", "js_pos1_step100",
            "early_slope_20_40", "log_early_slope", "early_auc_20_40",
            "js_pos1_slope_20_40", "js_pos1_auc_20_40",
            "steps_available", "n_steps_available"]
    out_df = out_df[cols].sort_values(["dataset", "run"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[h11_traj] wrote {OUT_CSV} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
