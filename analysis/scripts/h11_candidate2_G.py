#!/usr/bin/env python
"""h11 — Candidate 2 step-free predictor.

For each (run, dataset), compute:

    G_run = mean over available ckpt steps t of
            log10( JS_pos1_mean(t) / JS_interior_mean(t) )

This is a trajectory-averaged log-ratio of routing-vs-reasoning divergence
to base, with NO specific step number in the definition. Each run uses
whatever ckpts it has (vanilla/S1/S2 → {20,40,60,80,100}; negctl → {20,40}).

Companion / sanity-check variants:
    G_unweighted_arith  — current G (arithmetic mean over available steps)
    G_log_pos1_minus_log_interior — same as G but factored
    G_geomean_ratio     — geomean(ratio)= 10^G, presented in raw scale for readability
    max_log_ratio       — max over t of log10(ratio)   (candidate 3 for reference)

Then correlate vs the same step-100 (or latest) target as before:
    target_aime_best32
    target_aime_dri_rate
    collapse_aime_best32
    collapse_dri_rate

Output:
    analysis/tables/h11_candidate2_G.csv          (per-run G values)
    analysis/tables/h11_candidate2_correlations.csv  (Spearman/Kendall vs targets)
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
LONG_CSV = ROOT / "analysis/tables/h11_predictive_routing_js.csv"
MERGED_CSV = ROOT / "analysis/tables/h11_merged_indicator_target.csv"
OUT_G = ROOT / "analysis/tables/h11_candidate2_G.csv"
OUT_CORR = ROOT / "analysis/tables/h11_candidate2_correlations.csv"

RUN_ORDER = ["vanilla_I_q3", "S1_soft", "S2_hard", "negctl_DAI", "negctl_random"]
DATASETS = ["aime", "ifeval"]


def compute_G(sub: pd.DataFrame) -> dict:
    """sub is the (run, ds) slice of the long-form table, sorted by step."""
    sub = sub.sort_values("step")
    js_pos1 = sub["js_pos1_mean"].values.astype(float)
    js_int = sub["js_interior_mean"].values.astype(float)
    # mask out non-positive (shouldn't happen, but defensive)
    mask = (js_pos1 > 0) & (js_int > 0)
    if mask.sum() == 0:
        return {"G": np.nan, "geomean_ratio": np.nan,
                "max_log_ratio": np.nan, "n_steps": 0,
                "steps_used": "",
                "log_pos1_mean": np.nan, "log_interior_mean": np.nan}
    log_pos1 = np.log10(js_pos1[mask])
    log_int = np.log10(js_int[mask])
    log_ratio = log_pos1 - log_int

    return {
        "G": float(np.mean(log_ratio)),
        "geomean_ratio": float(10 ** np.mean(log_ratio)),
        "max_log_ratio": float(np.max(log_ratio)),
        "max_ratio": float(10 ** np.max(log_ratio)),
        "log_pos1_mean": float(np.mean(log_pos1)),
        "log_interior_mean": float(np.mean(log_int)),
        "n_steps": int(mask.sum()),
        "steps_used": ",".join(str(int(s)) for s in
                                sub.loc[mask, "step"].values),
    }


def safe_corr(xs: np.ndarray, ys: np.ndarray) -> dict:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3 or np.std(xs[mask]) == 0 or np.std(ys[mask]) == 0:
        return {"n": int(mask.sum()), "spearman_r": np.nan,
                "spearman_p": np.nan, "kendall_tau": np.nan,
                "kendall_p": np.nan}
    sr = stats.spearmanr(xs[mask], ys[mask])
    kt = stats.kendalltau(xs[mask], ys[mask])
    return {
        "n": int(mask.sum()),
        "spearman_r": float(sr.correlation),
        "spearman_p": float(sr.pvalue),
        "kendall_tau": float(kt.correlation),
        "kendall_p": float(kt.pvalue),
    }


def main():
    if not LONG_CSV.exists():
        raise FileNotFoundError(LONG_CSV)
    long_df = pd.read_csv(LONG_CSV)

    rows = []
    for run in RUN_ORDER:
        for ds in DATASETS:
            sub = long_df[(long_df["run"] == run) & (long_df["dataset"] == ds)]
            if len(sub) == 0:
                continue
            agg = compute_G(sub)
            agg["run"] = run
            agg["dataset"] = ds
            rows.append(agg)

    G_df = pd.DataFrame(rows).sort_values(["dataset", "run"])
    cols = ["run", "dataset", "G", "geomean_ratio", "max_log_ratio", "max_ratio",
            "log_pos1_mean", "log_interior_mean", "n_steps", "steps_used"]
    G_df = G_df[cols]
    OUT_G.parent.mkdir(parents=True, exist_ok=True)
    G_df.to_csv(OUT_G, index=False)
    print(f"[h11_C2] wrote {OUT_G} ({len(G_df)} rows)")
    print(G_df.to_string(index=False))

    # Merge G with targets
    if not MERGED_CSV.exists():
        raise FileNotFoundError(MERGED_CSV)
    merged = pd.read_csv(MERGED_CSV)
    targets = ["target_aime_best32", "target_aime_pass1",
               "target_aime_dri_rate", "target_aime_dai_rate",
               "collapse_aime_best32", "collapse_dri_rate",
               "target_ifeval_pass1", "target_ifeval_best32"]

    sum_rows = []
    for ds in DATASETS:
        G_ds = G_df[G_df["dataset"] == ds].set_index("run")
        for indicator in ["G", "geomean_ratio", "max_log_ratio", "max_ratio",
                          "log_pos1_mean", "log_interior_mean"]:
            xs_full = []
            ys_full_by_target = {t: [] for t in targets}
            for run in RUN_ORDER:
                if run not in G_ds.index:
                    continue
                m = merged[merged["run"] == run]
                if len(m) != 1:
                    continue
                xs_full.append(float(G_ds.loc[run, indicator]))
                for t in targets:
                    ys_full_by_target[t].append(float(m[t].iloc[0])
                                                if t in m.columns else np.nan)

            xs = np.array(xs_full, dtype=float)
            for t in targets:
                ys = np.array(ys_full_by_target[t], dtype=float)
                corr = safe_corr(xs, ys)
                sum_rows.append({
                    "dataset": ds,
                    "indicator": indicator,
                    "target": t,
                    **corr,
                })

    corr_df = pd.DataFrame(sum_rows)
    corr_df.to_csv(OUT_CORR, index=False)
    print(f"\n[h11_C2] wrote {OUT_CORR} ({len(corr_df)} rows)")

    # Headline
    print("\n=== Headline: G (and variants) vs targets ===")
    headline = corr_df[
        corr_df["indicator"].isin(["G", "geomean_ratio", "max_log_ratio",
                                   "log_pos1_mean", "log_interior_mean"])
        & corr_df["target"].isin(["target_aime_best32", "collapse_aime_best32",
                                  "collapse_dri_rate"])
        & (corr_df["dataset"] == "aime")
    ]
    print(headline[["indicator", "target", "n", "spearman_r", "spearman_p",
                    "kendall_tau", "kendall_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
