#!/usr/bin/env python
"""h11 — Indicator B: behavioral early-warning proxy.

For each of the 5 controlled IF-RLVR runs (vanilla I_q3, S1_soft, S2_hard,
DAI negctl, Random negctl), compute behavior-side proxies that should
*predict* step-100 cross-task collapse, using ONLY the rollout JSONLs that
exist at step ∈ {0, 5, 10, 15, 20, 25, 30}. No GPU forward needed.

Behavioral proxies:

    1. dri_rate_step_X       — DRI mode rate at AIME at step X (X=5,10,20,30).
    2. dri_slope_0_30        — linear-regression slope of DRI rate over
                               step 0–30 (negative = collapsing).
    3. dai_first_above_50    — earliest step at which DAI rate > 0.5 (NaN
                               if never reached in 0–100).
    4. jaccard_first5_step20 — mean Jaccard similarity between this run's
                               step-20 response first 5 tokens and B_q3's
                               step-0 response first 5 tokens, averaged
                               over AIME prompts. Lower = more diverged.
    5. mean_chars_step20     — mean response length at step 20 (chars).
                               Drops sharply at DAI collapse.

Per-run dependencies:

  Run                  | rollout root
  ---------------------+--------------------------------------------------
  vanilla_I_q3         | b2r1_Qwen3-8B-Base_IFTrain_local_H20/{step}.jsonl
  S1_soft              | b2r1_h1_S1_RL_dri50_local_H20/{step}.jsonl
  S2_hard              | b2r1_h1_S2_RL_dri50_local_H20/{step}.jsonl
  negctl_DAI           | b2r1_h7_negctl_DAI_dri50_local_H20/{step}.jsonl
  negctl_random        | b2r1_h7_negctl_random_dri50_local_H20/{step}.jsonl

B_q3 anchor:           b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl

Output: analysis/tables/h11_behavioral_proxy.csv
"""
from __future__ import annotations
import os

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(ROOT / "analysis/scripts"))
from classify_opening_modes import classify_mode  # noqa

VAL_DIR = ROOT / "rollout/val_rollout/DAPO_sh_repro"
OUT_CSV = ROOT / "analysis/tables/h11_behavioral_proxy.csv"

RUNS = [
    ("vanilla_I_q3",   "b2r1_Qwen3-8B-Base_IFTrain_local_H20"),
    ("S1_soft",        "b2r1_h1_S1_RL_dri50_local_H20"),
    ("S2_hard",        "b2r1_h1_S2_RL_dri50_local_H20"),
    ("negctl_DAI",     "b2r1_h7_negctl_DAI_dri50_local_H20"),
    ("negctl_random",  "b2r1_h7_negctl_random_dri50_local_H20"),
]

B_ANCHOR = VAL_DIR / "b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl"

EARLY_STEPS = [0, 5, 10, 15, 20, 25, 30]
ALL_STEPS = list(range(0, 105, 5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TOKEN_SPLIT = re.compile(r"\S+")


def first_n_tokens(text: str, n: int = 5) -> tuple:
    """Whitespace-split first N words; returns tuple (immutable, hashable)."""
    return tuple(_TOKEN_SPLIT.findall(text.lstrip())[:n])


def is_aime(input_text: str) -> bool:
    return "Solve the following math problem step by step" in input_text


def prompt_key(input_text: str) -> str:
    sep = "\nassistant\n"
    if sep in input_text:
        return input_text[: input_text.index(sep)]
    return input_text[:200]


def load_aime_rollouts(jsonl_path: Path) -> list[dict]:
    """Yield per-row dicts for AIME rows: {pkey, response, acc}."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            inp = o.get("input", "")
            if not is_aime(inp):
                continue
            out.append({
                "pkey": prompt_key(inp),
                "response": o.get("output", ""),
                "acc": bool(o.get("acc", False)),
            })
    return out


def aime_mode_rates(rows: list[dict]) -> dict:
    """Returns {DRI_rate, DAI_rate, Other_rate, CSI_rate, mean_chars}."""
    if not rows:
        return {"DRI_rate": np.nan, "DAI_rate": np.nan, "Other_rate": np.nan,
                "CSI_rate": np.nan, "mean_chars": np.nan, "n_resp": 0}
    modes = Counter()
    chars = []
    for r in rows:
        m = classify_mode(r["response"])
        modes[m] += 1
        chars.append(len(r["response"]))
    n = sum(modes.values())
    return {
        "DRI_rate": modes.get("DRI", 0) / n,
        "DAI_rate": modes.get("DAI", 0) / n,
        "Other_rate": modes.get("Other", 0) / n,
        "CSI_rate": modes.get("CSI", 0) / n,
        "mean_chars": float(np.mean(chars)),
        "n_resp": n,
    }


def jaccard_first5_to_anchor(rows: list[dict],
                             anchor_first5: dict[str, list[set]]) -> float:
    """Mean Jaccard@5 between this run's response and anchor's response on
    the same prompt. Anchor is keyed by pkey -> list of token-sets (one per
    sample). We average each sample's Jaccard against the *first* anchor
    sample for that prompt (for simplicity), then mean over rows.
    """
    if not rows or not anchor_first5:
        return np.nan
    sims = []
    for r in rows:
        pkey = r["pkey"]
        if pkey not in anchor_first5:
            continue
        run_set = set(first_n_tokens(r["response"], 5))
        # use first anchor sample for that prompt as the reference
        anchor_set = anchor_first5[pkey][0]
        if not run_set and not anchor_set:
            sims.append(1.0)
            continue
        union = run_set | anchor_set
        if not union:
            sims.append(np.nan)
            continue
        sims.append(len(run_set & anchor_set) / len(union))
    return float(np.nanmean(sims)) if sims else np.nan


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Simple least-squares slope. Returns NaN if degenerate."""
    if len(xs) < 2 or any(np.isnan(ys)):
        return np.nan
    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)
    if np.var(xs_a) == 0:
        return np.nan
    return float(np.cov(xs_a, ys_a, bias=True)[0, 1] / np.var(xs_a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[h11] loading B-anchor: {B_ANCHOR}")
    anchor_rows = load_aime_rollouts(B_ANCHOR)
    print(f"[h11]   {len(anchor_rows)} AIME rows in anchor")

    # group anchor first5 token sets by pkey
    anchor_first5 = defaultdict(list)
    for r in anchor_rows:
        anchor_first5[r["pkey"]].append(set(first_n_tokens(r["response"], 5)))
    print(f"[h11]   {len(anchor_first5)} unique AIME prompts in anchor")

    out_rows = []

    for run_label, run_dir in RUNS:
        run_root = VAL_DIR / run_dir
        if not run_root.exists():
            print(f"[h11] SKIP {run_label}: dir missing")
            continue

        # Step-by-step mode rates over ALL_STEPS that exist on disk
        per_step = {}
        for step in ALL_STEPS:
            jpath = run_root / f"{step}.jsonl"
            if not jpath.exists():
                continue
            rows = load_aime_rollouts(jpath)
            per_step[step] = aime_mode_rates(rows)
            per_step[step]["jaccard_first5_to_anchor"] = (
                jaccard_first5_to_anchor(rows, anchor_first5)
            )

        if not per_step:
            print(f"[h11] SKIP {run_label}: no rollouts on disk")
            continue

        # Indicator B aggregates
        max_step_avail = max(per_step.keys())
        print(f"[h11] {run_label}: {len(per_step)} steps available "
              f"(0..{max_step_avail})")

        # dri_slope_0_30
        early_pairs = [(s, per_step[s]["DRI_rate"])
                       for s in EARLY_STEPS if s in per_step]
        if early_pairs:
            xs, ys = zip(*early_pairs)
            dri_slope_0_30 = linear_slope(list(xs), list(ys))
        else:
            dri_slope_0_30 = np.nan

        # dai_first_above_50
        dai_first_50 = np.nan
        for s in sorted(per_step.keys()):
            if per_step[s]["DAI_rate"] > 0.5:
                dai_first_50 = s
                break

        # jaccard / chars / DRI rate at step 20
        s20 = per_step.get(20, {})
        jaccard_first5_step20 = s20.get("jaccard_first5_to_anchor", np.nan)
        mean_chars_step20 = s20.get("mean_chars", np.nan)
        dri_rate_step20 = s20.get("DRI_rate", np.nan)

        # extra: DRI rate at every early step
        snap = {}
        for s in [5, 10, 15, 20, 25, 30]:
            if s in per_step:
                snap[f"dri_rate_step{s}"] = per_step[s]["DRI_rate"]
                snap[f"dai_rate_step{s}"] = per_step[s]["DAI_rate"]
            else:
                snap[f"dri_rate_step{s}"] = np.nan
                snap[f"dai_rate_step{s}"] = np.nan

        out_rows.append({
            "run": run_label,
            "max_step_available": max_step_avail,
            "dri_slope_0_30": dri_slope_0_30,
            "dai_first_above_50": dai_first_50,
            "jaccard_first5_step20": jaccard_first5_step20,
            "mean_chars_step20": mean_chars_step20,
            "dri_rate_step20": dri_rate_step20,
            **snap,
        })

    df = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[h11] wrote {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
