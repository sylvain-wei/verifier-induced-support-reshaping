#!/usr/bin/env python
"""ReasonIF-paper-style comprehensive evaluation.

Per ReasonIF (Kwon et al., 2025) the central distinction is:
  - Reasoning IFS   : constraint applied to the reasoning trace (text BEFORE
                     the first <answer> tag)
  - Response IFS    : constraint applied to the final response

We already computed both per-rollout in `v2_scored_*_reasonif.jsonl`. This
script aggregates them along all axes the ReasonIF paper uses:

  1. Overall: Reasoning IFS, Response IFS, gap (response - reasoning), acc.
  2. By 6 reasoning-instruction families: punctuation / length / language /
     startend / detectable_format / change_case.
  3. By 5 sources: gsm8k / arc / amc / aime / gpqa.
  4. Length-adjusted: per-model length deciles → IFS curves.
  5. Failure-mode taxonomy: bypass / answer-side compliance / partial trace
     compliance / contradicted / clean-fail.
  6. RIF-style decomposition: which families would benefit most from
     reasoning-IF-aware finetuning (largest reasoning-vs-response gap).
  7. Coupling between accuracy and reasoning IF (independence, +/- correlation).
  8. pass@1 vs best-of-16 vs majority@16 for both Reasoning IFS and Response IFS.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import read_records  # noqa: E402

MODEL_ORDER = ["base", "math_rlvr", "if_rlvr"]


def load() -> pd.DataFrame:
    rows: List[Dict] = []
    for m in MODEL_ORDER:
        p = ROOT / "analysis" / "tables" / f"v2_scored_{m}_reasonif.jsonl"
        for r in read_records(p):
            r["model"] = m
            rows.append(r)
    df = pd.DataFrame(rows)
    df["acc"] = pd.to_numeric(df.get("acc"), errors="coerce").fillna(0).astype(int)
    df["reasoning_follow_strict"] = pd.to_numeric(df.get("reasoning_follow_strict"), errors="coerce")
    df["response_follow_strict"] = pd.to_numeric(df.get("response_follow_strict"), errors="coerce")
    df["reasoning_pass"] = (df["reasoning_follow_strict"] == 1.0).astype(int)
    df["response_pass"] = (df["response_follow_strict"] == 1.0).astype(int)
    df["bypass"] = df["bypass_reasoning"].astype(bool)
    df["genuine_pass"] = ((df["reasoning_pass"] == 1) & (~df["bypass"])).astype(int)
    df["joint"] = ((df["genuine_pass"] == 1) & (df["acc"] == 1)).astype(int)
    df["family"] = df["constraint_families"].apply(
        lambda v: v[0] if isinstance(v, list) and v else (str(v) if v else "")
    )
    df["len_words"] = pd.to_numeric(df.get("response_length_words"), errors="coerce")
    df["reasoning_words"] = pd.to_numeric(df.get("reasoning_word_count"), errors="coerce")
    return df


def _aggrow(g: pd.DataFrame) -> Dict[str, float]:
    return {
        "n": len(g),
        "Reasoning_IFS": g["reasoning_pass"].mean(),
        "Response_IFS": g["response_pass"].mean(),
        "IFS_gap": g["response_pass"].mean() - g["reasoning_pass"].mean(),
        "Genuine_IFS": g["genuine_pass"].mean(),
        "bypass_rate": g["bypass"].mean(),
        "acc": g["acc"].mean(),
        "joint": g["joint"].mean(),
        "reasoning_words": float(g["reasoning_words"].mean()),
        "response_words": float(g["len_words"].mean()),
    }


def _ag(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))
        rec.update(_aggrow(g))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if "model" in out:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    return out.sort_values(group_cols)


# ------------------------------------------------------------------
# 1-3. overall / by family / by source
# ------------------------------------------------------------------

def overall(df, out_dir):
    _ag(df, ["model"]).to_csv(out_dir / "reasonifpaper_overall.csv", index=False)


def by_family(df, out_dir):
    _ag(df, ["family", "model"]).to_csv(out_dir / "reasonifpaper_by_family.csv", index=False)


def by_source(df, out_dir):
    _ag(df, ["source", "model"]).to_csv(out_dir / "reasonifpaper_by_source.csv", index=False)


def by_family_source(df, out_dir):
    _ag(df, ["family", "source", "model"]).to_csv(out_dir / "reasonifpaper_by_family_source.csv", index=False)


# ------------------------------------------------------------------
# 4. length-adjusted: deciles of reasoning_words
# ------------------------------------------------------------------

def length_deciles(df: pd.DataFrame, out_dir: Path):
    rows = []
    for m, g in df.groupby("model"):
        if g["reasoning_words"].dropna().empty:
            continue
        try:
            bins = pd.qcut(g["reasoning_words"].fillna(0), 10, labels=False, duplicates="drop")
        except Exception:
            continue
        gg = g.assign(_bin=bins)
        for b, sub in gg.groupby("_bin"):
            rows.append({
                "model": m, "decile": int(b),
                "n": len(sub),
                "min_words": float(sub["reasoning_words"].min()),
                "max_words": float(sub["reasoning_words"].max()),
                "mean_words": float(sub["reasoning_words"].mean()),
                "Reasoning_IFS": sub["reasoning_pass"].mean(),
                "Response_IFS": sub["response_pass"].mean(),
                "Genuine_IFS": sub["genuine_pass"].mean(),
                "bypass_rate": sub["bypass"].mean(),
                "acc": sub["acc"].mean(),
            })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values(["model", "decile"]).to_csv(out_dir / "reasonifpaper_length_deciles.csv", index=False)


# ------------------------------------------------------------------
# 5. failure-mode taxonomy
# ------------------------------------------------------------------

def failure_modes(df: pd.DataFrame, out_dir: Path):
    """Classify each rollout into one of 5 failure modes:
       FM_PASS        — reasoning IF satisfied AND ¬bypass AND acc=1
       FM_GENUINE     — reasoning IF satisfied AND ¬bypass (regardless of acc)
       FM_BYPASS      — bypass=True (trace skipped, IF trivially holds)
       FM_ANSWER_ONLY — response_pass=1 AND reasoning_pass=0 (rule held in
                        final answer but violated in trace)
       FM_PARTIAL     — both reasoning_pass=0 and response_pass=0 (clean miss)
    """
    def _mode(r):
        if r["reasoning_pass"] == 1 and not r["bypass"]:
            return "FM_GENUINE"
        if r["bypass"] and r["reasoning_pass"] == 1:
            return "FM_BYPASS"
        if r["reasoning_pass"] == 0 and r["response_pass"] == 1:
            return "FM_ANSWER_ONLY"
        return "FM_PARTIAL"

    df = df.copy()
    df["mode"] = df.apply(_mode, axis=1)
    rows = []
    for (m, mode), g in df.groupby(["model", "mode"]):
        rows.append({
            "model": m, "mode": mode,
            "n": len(g),
            "share": len(g) / len(df[df["model"] == m]),
            "acc": g["acc"].mean(),
            "reasoning_words": g["reasoning_words"].mean(),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values(["mode", "model"]).to_csv(out_dir / "reasonifpaper_failure_modes.csv", index=False)


# ------------------------------------------------------------------
# 6. RIF-style decomposition: which families have largest reasoning vs
# response gap (= largest opportunity for RIF finetuning)
# ------------------------------------------------------------------

def rif_opportunity(df: pd.DataFrame, out_dir: Path):
    rows = []
    for (fam, m), g in df.groupby(["family", "model"]):
        rows.append({
            "family": fam, "model": m,
            "Reasoning_IFS": g["reasoning_pass"].mean(),
            "Response_IFS": g["response_pass"].mean(),
            "gap_response_minus_reasoning": g["response_pass"].mean() - g["reasoning_pass"].mean(),
            "Genuine_IFS": g["genuine_pass"].mean(),
            "bypass_rate": g["bypass"].mean(),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values(["family", "model"]).to_csv(out_dir / "reasonifpaper_rif_opportunity.csv", index=False)


# ------------------------------------------------------------------
# 7. coupling between accuracy and reasoning_IF
# ------------------------------------------------------------------

def coupling(df: pd.DataFrame, out_dir: Path):
    rows = []
    for m, g in df.groupby("model"):
        a = g["acc"].astype(int).values
        f_r = g["reasoning_pass"].astype(int).values
        f_R = g["response_pass"].astype(int).values
        rows.append({
            "model": m, "n": len(g),
            "P(reasoning_pass)": f_r.mean(),
            "P(response_pass)": f_R.mean(),
            "P(reasoning_pass | acc=1)": f_r[a == 1].mean() if (a == 1).any() else float("nan"),
            "P(reasoning_pass | acc=0)": f_r[a == 0].mean() if (a == 0).any() else float("nan"),
            "P(acc | reasoning_pass=1)": a[f_r == 1].mean() if (f_r == 1).any() else float("nan"),
            "P(acc | reasoning_pass=0)": a[f_r == 0].mean() if (f_r == 0).any() else float("nan"),
            "phi_acc_reasoning": _phi(a, f_r),
            "phi_acc_response": _phi(a, f_R),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values("model").to_csv(out_dir / "reasonifpaper_coupling.csv", index=False)


def _phi(a, b):
    a = a.astype(int); b = b.astype(int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    num = n11 * n00 - n10 * n01
    den = float(np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------
# 8. pass@1 / best@16 / maj@16 for Reasoning IFS, Response IFS, Genuine IFS
# ------------------------------------------------------------------

def passk(df: pd.DataFrame, out_dir: Path):
    rows = []
    for (m, eid), g in df.groupby(["model", "example_id"]):
        if g.empty:
            continue
        rows.append({
            "model": m, "example_id": eid, "k": len(g),
            "RIFS_pass1": g["reasoning_pass"].mean(),
            "RIFS_bestK": int(g["reasoning_pass"].max()),
            "RIFS_majK": int(g["reasoning_pass"].sum() > len(g) / 2),
            "ResIFS_pass1": g["response_pass"].mean(),
            "ResIFS_bestK": int(g["response_pass"].max()),
            "ResIFS_majK": int(g["response_pass"].sum() > len(g) / 2),
            "Genuine_pass1": g["genuine_pass"].mean(),
            "Genuine_bestK": int(g["genuine_pass"].max()),
            "Genuine_majK": int(g["genuine_pass"].sum() > len(g) / 2),
            "acc_pass1": g["acc"].mean(),
            "acc_bestK": int(g["acc"].max()),
            "joint_pass1": g["joint"].mean(),
            "joint_bestK": int(g["joint"].max()),
        })
    pdf = pd.DataFrame(rows)
    summary = pdf.groupby("model").agg(
        n_prompts=("example_id", "nunique"),
        RIFS_pass1=("RIFS_pass1", "mean"),
        RIFS_bestK=("RIFS_bestK", "mean"),
        RIFS_majK=("RIFS_majK", "mean"),
        ResIFS_pass1=("ResIFS_pass1", "mean"),
        ResIFS_bestK=("ResIFS_bestK", "mean"),
        ResIFS_majK=("ResIFS_majK", "mean"),
        Genuine_pass1=("Genuine_pass1", "mean"),
        Genuine_bestK=("Genuine_bestK", "mean"),
        Genuine_majK=("Genuine_majK", "mean"),
        acc_pass1=("acc_pass1", "mean"),
        acc_bestK=("acc_bestK", "mean"),
        joint_pass1=("joint_pass1", "mean"),
        joint_bestK=("joint_bestK", "mean"),
    ).reset_index()
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary.sort_values("model").to_csv(out_dir / "reasonifpaper_passk.csv", index=False)


# ------------------------------------------------------------------
# 9. deltas vs base
# ------------------------------------------------------------------

def deltas(df: pd.DataFrame, out_dir: Path):
    base = _ag(df, ["model"]).set_index("model").loc["base"]
    rows = []
    for m in MODEL_ORDER:
        if m == "base":
            continue
        sub = _ag(df, ["model"]).set_index("model").loc[m]
        rows.append({
            "model": m,
            "Δacc": sub["acc"] - base["acc"],
            "ΔReasoning_IFS": sub["Reasoning_IFS"] - base["Reasoning_IFS"],
            "ΔResponse_IFS": sub["Response_IFS"] - base["Response_IFS"],
            "ΔIFS_gap": sub["IFS_gap"] - base["IFS_gap"],
            "ΔGenuine_IFS": sub["Genuine_IFS"] - base["Genuine_IFS"],
            "Δbypass": sub["bypass_rate"] - base["bypass_rate"],
            "Δjoint": sub["joint"] - base["joint"],
            "Δreasoning_words": sub["reasoning_words"] - base["reasoning_words"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "reasonifpaper_deltas.csv", index=False)


def main():
    df = load()
    out_dir = ROOT / "analysis" / "tables"
    overall(df, out_dir)
    by_family(df, out_dir)
    by_source(df, out_dir)
    by_family_source(df, out_dir)
    length_deciles(df, out_dir)
    failure_modes(df, out_dir)
    rif_opportunity(df, out_dir)
    coupling(df, out_dir)
    passk(df, out_dir)
    deltas(df, out_dir)
    print("wrote ReasonIF paper-style tables to", out_dir)


if __name__ == "__main__":
    main()
