#!/usr/bin/env python
"""h11 — Predictive validation: correlate Indicator A/B @ step_20 with
step-100 cross-task collapse.

Inputs:
  - analysis/tables/h11_predictive_routing_js.csv   (Indicator A)
  - analysis/tables/h11_behavioral_proxy.csv         (Indicator B)
  - analysis/tables/wave5_final_eval.csv             (S1/S2 step-100 truth)
  - For runs missing from wave5_final_eval (vanilla I_q3, negctl_DAI,
    negctl_random), we recompute step-100 metrics on-the-fly from
    rollout/val_rollout/.../{step_max}.jsonl (using the same logic as
    h5_wave5_final_eval). For negctl runs whose RL hasn't reached step 100
    yet, we use the latest available step as a proxy and flag it.

Computes:
  - spearmanr(indicator, target) over n=5 runs
  - kendalltau(indicator, target)
  - Indicator A: js_pos1_mean (aime, ifeval), pos1_ratio
  - Indicator B: dri_slope_0_30, dai_first_above_50,
                  jaccard_first5_step20, mean_chars_step20, dri_rate_step20
  - Targets:    aime_best32_step100, dai_rate_step100

Output: analysis/tables/h11_predictive_summary.csv
"""
from __future__ import annotations
import os

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(ROOT / "analysis/scripts"))
from classify_opening_modes import classify_mode  # noqa: E402

VAL_DIR = ROOT / "rollout/val_rollout/DAPO_sh_repro"
A_CSV = ROOT / "analysis/tables/h11_predictive_routing_js.csv"
B_CSV = ROOT / "analysis/tables/h11_behavioral_proxy.csv"
TRAJ_CSV = ROOT / "analysis/tables/h11_trajectory_features.csv"
WAVE5_CSV = ROOT / "analysis/tables/wave5_final_eval.csv"
OUT_SUM = ROOT / "analysis/tables/h11_predictive_summary.csv"
OUT_MERGED = ROOT / "analysis/tables/h11_merged_indicator_target.csv"

# Run-name normalization: behavioral / trajectory / rollout dirs use
# different conventions. Map them all to a canonical "label".
RUNS = [
    # (canonical_label, behavioral_run, rollout_dir)
    ("vanilla_I_q3",  "vanilla_I_q3",  "b2r1_Qwen3-8B-Base_IFTrain_local_H20"),
    ("S1_soft",       "S1_soft",       "b2r1_h1_S1_RL_dri50_local_H20"),
    ("S2_hard",       "S2_hard",       "b2r1_h1_S2_RL_dri50_local_H20"),
    ("negctl_DAI",    "negctl_DAI",    "b2r1_h7_negctl_DAI_dri50_local_H20"),
    ("negctl_random", "negctl_random", "b2r1_h7_negctl_random_dri50_local_H20"),
]


# ---------------------------------------------------------------------------
# Compute step-100 (or latest) target metrics from rollout JSONL
# ---------------------------------------------------------------------------
def compute_target_from_rollout(rollout_dir: str, step: int | None = None) -> dict:
    """Returns metrics on the step.jsonl. If step is None, picks latest
    available ≤ 100.
    """
    rdir = VAL_DIR / rollout_dir
    if not rdir.exists():
        return None
    available = sorted(int(p.stem) for p in rdir.glob("*.jsonl")
                       if p.stem.isdigit())
    if not available:
        return None
    if step is None:
        target = max(s for s in available if s <= 100)
    else:
        if step not in available:
            return None
        target = step
    jpath = rdir / f"{target}.jsonl"

    aime_by_prompt = defaultdict(list)
    ifeval_by_prompt = defaultdict(list)
    aime_modes = Counter()
    aime_chars = []

    with open(jpath) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            inp = o.get("input", "")
            resp = o.get("output", "")
            acc = bool(o.get("acc", False))
            score = float(o.get("score", 0) or 0)
            sep = "\nassistant\n"
            pkey = inp[: inp.index(sep)] if sep in inp else inp[:200]

            if "Solve the following math problem step by step" in inp:
                aime_by_prompt[pkey].append((acc, resp, score))
                m = classify_mode(resp)
                aime_modes[m] += 1
                aime_chars.append(len(resp))
            else:
                ifeval_by_prompt[pkey].append((acc, resp, score))

    if not aime_by_prompt:
        return None

    aime_pass1 = float(np.mean([
        np.mean([a for a, _, _ in lst]) for lst in aime_by_prompt.values()
    ]))
    aime_best = float(np.mean([
        1.0 if any(a for a, _, _ in lst) else 0.0
        for lst in aime_by_prompt.values()
    ]))
    n_aime = sum(aime_modes.values())
    dri_rate = aime_modes.get("DRI", 0) / max(n_aime, 1)
    dai_rate = aime_modes.get("DAI", 0) / max(n_aime, 1)
    other_rate = aime_modes.get("Other", 0) / max(n_aime, 1)
    mean_chars = float(np.mean(aime_chars)) if aime_chars else np.nan

    if ifeval_by_prompt:
        if_pass1 = float(np.mean([
            np.mean([s for _, _, s in lst]) for lst in ifeval_by_prompt.values()
        ]))
        if_best = float(np.mean([
            max(s for _, _, s in lst) for lst in ifeval_by_prompt.values()
        ]))
    else:
        if_pass1 = np.nan
        if_best = np.nan

    return {
        "step_used": target,
        "target_is_step100": (target == 100),
        "aime_best32": aime_best,
        "aime_pass1": aime_pass1,
        "aime_dri_rate": dri_rate,
        "aime_dai_rate": dai_rate,
        "aime_other_rate": other_rate,
        "aime_mean_chars": mean_chars,
        "ifeval_pass1": if_pass1,
        "ifeval_best32": if_best,
    }


# ---------------------------------------------------------------------------
# Build the merged (label × indicator × target) table
# ---------------------------------------------------------------------------
def build_merged() -> pd.DataFrame:
    df_a = pd.read_csv(A_CSV) if A_CSV.exists() else pd.DataFrame()
    df_b = pd.read_csv(B_CSV) if B_CSV.exists() else pd.DataFrame()
    df_t = pd.read_csv(TRAJ_CSV) if TRAJ_CSV.exists() else pd.DataFrame()

    rows = []
    for (label, beh_run, rollout_dir) in RUNS:
        rec = {"run": label}

        # ---- Indicator A @ step 20 (single-step baseline)
        for ds in ["aime", "ifeval"]:
            sub = df_a[(df_a["run"] == label)
                       & (df_a["dataset"] == ds)
                       & (df_a["step"] == 20)]
            if len(sub) == 1:
                rec[f"js_pos1_{ds}"] = sub["js_pos1_mean"].iloc[0]
                rec[f"js_interior_{ds}"] = sub["js_interior_mean"].iloc[0]
                rec[f"pos1_ratio_{ds}"] = sub["pos1_ratio"].iloc[0]
            else:
                rec[f"js_pos1_{ds}"] = np.nan
                rec[f"js_interior_{ds}"] = np.nan
                rec[f"pos1_ratio_{ds}"] = np.nan

        # ---- Indicator A trajectory features (early slope etc.)
        for ds in ["aime", "ifeval"]:
            sub = df_t[(df_t["run"] == label) & (df_t["dataset"] == ds)]
            if len(sub) == 1:
                for col in ["early_slope_20_40", "log_early_slope",
                            "early_auc_20_40",
                            "js_pos1_slope_20_40", "js_pos1_auc_20_40",
                            "pos1_step40", "pos1_step100"]:
                    rec[f"{col}_{ds}"] = sub[col].iloc[0]
            else:
                for col in ["early_slope_20_40", "log_early_slope",
                            "early_auc_20_40",
                            "js_pos1_slope_20_40", "js_pos1_auc_20_40",
                            "pos1_step40", "pos1_step100"]:
                    rec[f"{col}_{ds}"] = np.nan

        # ---- Indicator B
        sub = df_b[df_b["run"] == beh_run]
        if len(sub) == 1:
            for col in ["dri_slope_0_30", "dai_first_above_50",
                        "jaccard_first5_step20", "mean_chars_step20",
                        "dri_rate_step20"]:
                rec[col] = sub[col].iloc[0]
        else:
            for col in ["dri_slope_0_30", "dai_first_above_50",
                        "jaccard_first5_step20", "mean_chars_step20",
                        "dri_rate_step20"]:
                rec[col] = np.nan

        # ---- Target (step-100 or latest) + step-0 baseline + collapse magnitude
        tgt = compute_target_from_rollout(rollout_dir)
        if tgt is not None:
            rec.update({f"target_{k}": v for k, v in tgt.items()})
        else:
            print(f"[h11_corr] no target rollout for {label}")

        # step-0 baseline metrics for the same run
        baseline = compute_target_from_rollout(rollout_dir, step=0)
        if baseline is not None:
            rec.update({f"baseline_{k}": v for k, v in baseline.items()})
            # collapse magnitude (positive = collapsed)
            if (rec.get("baseline_aime_best32") is not None and
                rec.get("target_aime_best32") is not None):
                rec["collapse_aime_best32"] = (
                    rec["baseline_aime_best32"] - rec["target_aime_best32"]
                )
            if (rec.get("baseline_aime_dri_rate") is not None and
                rec.get("target_aime_dri_rate") is not None):
                rec["collapse_dri_rate"] = (
                    rec["baseline_aime_dri_rate"] - rec["target_aime_dri_rate"]
                )

        rows.append(rec)

    return pd.DataFrame(rows)


def safe_corr(xs: np.ndarray, ys: np.ndarray):
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "spearman_r": np.nan, "spearman_p": np.nan,
                "kendall_tau": np.nan, "kendall_p": np.nan}
    xs2 = xs[mask]
    ys2 = ys[mask]
    if np.std(xs2) == 0 or np.std(ys2) == 0:
        return {"n": int(mask.sum()), "spearman_r": np.nan, "spearman_p": np.nan,
                "kendall_tau": np.nan, "kendall_p": np.nan}
    sr = stats.spearmanr(xs2, ys2)
    kt = stats.kendalltau(xs2, ys2)
    return {
        "n": int(mask.sum()),
        "spearman_r": float(sr.correlation),
        "spearman_p": float(sr.pvalue),
        "kendall_tau": float(kt.correlation),
        "kendall_p": float(kt.pvalue),
    }


def main():
    print("[h11_corr] building merged indicator × target table...")
    merged = build_merged()
    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"[h11_corr] wrote merged table → {OUT_MERGED}")
    print(merged.to_string(index=False))

    # All correlations: indicator vs target
    indicators = [
        # ---- Single-step (legacy) baseline
        ("js_pos1_aime",          "Single-step: pos-1 JS @ step 20 (AIME)"),
        ("js_pos1_ifeval",        "Single-step: pos-1 JS @ step 20 (IFEval)"),
        ("pos1_ratio_aime",       "Single-step: pos-1/interior ratio @ step 20 (AIME)"),
        ("pos1_ratio_ifeval",     "Single-step: pos-1/interior ratio @ step 20 (IFEval)"),
        ("js_interior_aime",      "Negative control: interior JS @ step 20 (AIME)"),
        ("js_interior_ifeval",    "Negative control: interior JS @ step 20 (IFEval)"),
        # ---- New trajectory-based predictors (Indicator A')
        ("early_slope_20_40_aime",    "Trajectory: early slope of pos1/interior ratio (AIME)"),
        ("early_auc_20_40_aime",      "Trajectory: early AUC of pos1/interior ratio (AIME)"),
        ("js_pos1_slope_20_40_aime",  "Trajectory: early slope of raw pos-1 JS (AIME)"),
        ("js_pos1_auc_20_40_aime",    "Trajectory: early AUC of raw pos-1 JS (AIME)"),
        ("log_early_slope_aime",      "Trajectory: log10 early slope of ratio (AIME)"),
        ("early_slope_20_40_ifeval",  "Trajectory: early slope of pos1/interior ratio (IFEval)"),
        ("early_auc_20_40_ifeval",    "Trajectory: early AUC of pos1/interior ratio (IFEval)"),
        ("pos1_step40_aime",          "Single-step: pos-1/interior ratio @ step 40 (AIME)"),
        # ---- Indicator B (rollout-only, no GPU)
        ("dri_slope_0_30",        "Indicator B: DRI slope step 0-30"),
        ("dai_first_above_50",    "Indicator B: step at which DAI > 0.5"),
        ("jaccard_first5_step20", "Indicator B: jaccard first-5 vs base @ step 20"),
        ("mean_chars_step20",     "Indicator B: mean chars @ step 20"),
        ("dri_rate_step20",       "Indicator B: DRI rate @ step 20"),
    ]
    targets = [
        ("target_aime_best32",   "AIME best@32 (latest)"),
        ("target_aime_pass1",    "AIME pass@1 (latest)"),
        ("target_aime_dai_rate", "AIME DAI rate (latest)"),
        ("collapse_aime_best32", "AIME b@32 collapse (baseline − latest)"),
        ("collapse_dri_rate",    "AIME DRI rate collapse (baseline − latest)"),
    ]

    sum_rows = []
    for ic, ic_label in indicators:
        for tc, tc_label in targets:
            if ic not in merged.columns or tc not in merged.columns:
                continue
            corr = safe_corr(merged[ic].values, merged[tc].values)
            sum_rows.append({
                "indicator": ic, "indicator_label": ic_label,
                "target": tc, "target_label": tc_label,
                **corr,
            })

    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(OUT_SUM, index=False)
    print(f"\n[h11_corr] wrote summary → {OUT_SUM}")
    # Highlight headline rows
    headline = sum_df[
        sum_df["indicator"].isin(
            ["pos1_ratio_aime",
             "early_slope_20_40_aime", "early_auc_20_40_aime",
             "js_pos1_slope_20_40_aime", "js_pos1_auc_20_40_aime",
             "log_early_slope_aime", "pos1_step40_aime",
             "js_interior_aime",
             "jaccard_first5_step20", "dai_first_above_50"])
        & (sum_df["target"].isin(["target_aime_best32",
                                  "collapse_aime_best32",
                                  "collapse_dri_rate"]))
    ]
    print("\n=== Headline correlations ===")
    print(headline[["indicator", "target", "n", "spearman_r", "spearman_p",
                    "kendall_tau", "kendall_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
