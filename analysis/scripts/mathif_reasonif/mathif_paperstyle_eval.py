#!/usr/bin/env python
"""MathIF-paper-style comprehensive evaluation.

For each (model, condition) we compute:
  - HAcc      = mean(acc=1 AND strict_follow=1)
  - SAcc      = mean(acc=1)
  - IFAcc     = mean(strict_follow=1)
  - gap       = SAcc - HAcc       (price paid for following constraints)
  - ratio     = HAcc / max(SAcc, eps)
  - len_words = mean response length in words

Conditions covered:
  1. overall
  2. by source (gsm8k / math500 / minerva / olympiad / aime)
  3. by difficulty (single / double / triple)
  4. by constraint family (mapped to MathIF 4 macro-classes:
        length / lexical / format / affix)
  5. by individual constraint name
  6. SAcc-conditioned IF: P(follow | acc=1) vs P(follow | acc=0)
     and P(acc | follow=1) vs P(acc | follow=0)
  7. response length deciles vs HAcc (long response → IF drops?)
  8. pass@1 vs best-of-16 vs maj@16 for SAcc, IFAcc, HAcc
  9. multi-constraint compositional drop (single/double/triple decomposition)
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

# MathIF paper groups its constraints into 4 macro categories. Map our
# IFEval-style families to those macros.
FAMILY_TO_MACRO = {
    "length_constraint_checkers": "length",
    "keywords": "lexical",
    "language": "lexical",
    "punctuation": "lexical",
    "change_case": "format",
    "detectable_format": "format",
    "combination": "format",
    "startend": "affix",
}


def load_mathif() -> pd.DataFrame:
    rows: List[Dict] = []
    for m in MODEL_ORDER:
        p = ROOT / "analysis" / "tables" / f"v2_scored_{m}_mathif.jsonl"
        for r in read_records(p):
            r["model"] = m
            rows.append(r)
    df = pd.DataFrame(rows)
    df["acc"] = pd.to_numeric(df.get("acc"), errors="coerce").fillna(0).astype(int)
    df["response_follow_strict"] = pd.to_numeric(df.get("response_follow_strict"), errors="coerce")
    df["response_pass"] = (df["response_follow_strict"] == 1.0).astype(int)
    df["joint"] = ((df["acc"] == 1) & (df["response_pass"] == 1)).astype(int)
    df["len_words"] = pd.to_numeric(df.get("response_length_words"), errors="coerce")
    df["families"] = df["constraint_families"].apply(
        lambda v: list(v) if isinstance(v, list) else ([str(v)] if v else [])
    )
    df["macro_set"] = df["families"].apply(
        lambda fs: sorted({FAMILY_TO_MACRO.get(f, "other") for f in fs})
    )
    df["constraint_names_str"] = df["constraint_names"].apply(
        lambda v: ",".join(v) if isinstance(v, list) else str(v) if v else ""
    )
    return df


# ----------------------- aggregation primitives -----------------------

def _aggrow(g: pd.DataFrame) -> Dict[str, float]:
    n = len(g)
    sacc = g["acc"].mean()
    hacc = g["joint"].mean()
    ifacc = g["response_pass"].mean()
    return {
        "n": n,
        "SAcc": sacc,
        "IFAcc": ifacc,
        "HAcc": hacc,
        "gap_SAcc_HAcc": sacc - hacc,
        "ratio_HAcc_SAcc": hacc / sacc if sacc > 0 else float("nan"),
        "len_words": float(g["len_words"].mean()),
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


# ----------------------- 1-2-3 overall / source / difficulty -----------------------

def overall(df, out_dir):
    _ag(df, ["model"]).to_csv(out_dir / "mathifpaper_overall.csv", index=False)


def by_source(df, out_dir):
    _ag(df, ["source", "model"]).to_csv(out_dir / "mathifpaper_by_source.csv", index=False)


def by_difficulty(df, out_dir):
    _ag(df, ["difficulty", "model"]).to_csv(out_dir / "mathifpaper_by_difficulty.csv", index=False)


def by_difficulty_source(df, out_dir):
    _ag(df, ["difficulty", "source", "model"]).to_csv(out_dir / "mathifpaper_by_difficulty_source.csv", index=False)


# ----------------------- 4. macro / family -----------------------

def by_macro_constraint(df: pd.DataFrame, out_dir: Path):
    rows = []
    for _, r in df.iterrows():
        per = r.get("response_per_constraint") or []
        for pc in per:
            if not pc.get("supported"):
                continue
            macro = FAMILY_TO_MACRO.get(pc.get("family"), "other")
            rows.append({
                "model": r["model"],
                "macro": macro,
                "family": pc.get("family"),
                "instruction_name": pc.get("instruction_name"),
                "passed": int(pc.get("passed") is True),
                "acc": r["acc"],
                "len_words": r["len_words"],
            })
    pdf = pd.DataFrame(rows)
    macro = pdf.groupby(["macro", "model"]).agg(
        n=("passed", "size"),
        IFAcc=("passed", "mean"),
        SAcc=("acc", "mean"),
        len_words=("len_words", "mean"),
    ).reset_index()
    macro["model"] = pd.Categorical(macro["model"], MODEL_ORDER, ordered=True)
    macro.sort_values(["macro", "model"]).to_csv(out_dir / "mathifpaper_by_macro.csv", index=False)

    fam = pdf.groupby(["family", "model"]).agg(
        n=("passed", "size"),
        IFAcc=("passed", "mean"),
        SAcc=("acc", "mean"),
        len_words=("len_words", "mean"),
    ).reset_index()
    fam["model"] = pd.Categorical(fam["model"], MODEL_ORDER, ordered=True)
    fam.sort_values(["family", "model"]).to_csv(out_dir / "mathifpaper_by_family.csv", index=False)

    inst = pdf.groupby(["instruction_name", "model"]).agg(
        n=("passed", "size"),
        IFAcc=("passed", "mean"),
        SAcc=("acc", "mean"),
    ).reset_index()
    inst["model"] = pd.Categorical(inst["model"], MODEL_ORDER, ordered=True)
    inst.sort_values(["instruction_name", "model"]).to_csv(out_dir / "mathifpaper_by_instruction.csv", index=False)


# ----------------------- 5. correctness-following coupling -----------------------

def coupling(df: pd.DataFrame, out_dir: Path):
    rows = []
    for m, g in df.groupby("model"):
        n = len(g)
        a1 = g[g["acc"] == 1]
        a0 = g[g["acc"] == 0]
        f1 = g[g["response_pass"] == 1]
        f0 = g[g["response_pass"] == 0]
        rows.append({
            "model": m,
            "n": n,
            "P(acc=1)": g["acc"].mean(),
            "P(follow=1)": g["response_pass"].mean(),
            "P(both)": g["joint"].mean(),
            "P(follow|acc=1)": f1.shape[0] / max(n, 1) if a1.shape[0] == 0 else a1["response_pass"].mean(),
            "P(follow|acc=0)": float("nan") if a0.shape[0] == 0 else a0["response_pass"].mean(),
            "P(acc|follow=1)": float("nan") if f1.shape[0] == 0 else f1["acc"].mean(),
            "P(acc|follow=0)": float("nan") if f0.shape[0] == 0 else f0["acc"].mean(),
            # Independence test: Phi (2x2 correlation)
            "phi": _phi(g["acc"].values, g["response_pass"].values),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values("model").to_csv(out_dir / "mathifpaper_coupling.csv", index=False)


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(int); b = b.astype(int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    num = n11 * n00 - n10 * n01
    den = float(np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return num / den if den > 0 else float("nan")


# ----------------------- 6. response length × IF coupling -----------------------

def length_bins(df: pd.DataFrame, out_dir: Path):
    rows = []
    for m, g in df.groupby("model"):
        if g["len_words"].dropna().empty:
            continue
        # Use deciles per-model so comparison is within-distribution.
        try:
            bins = pd.qcut(g["len_words"], 10, labels=False, duplicates="drop")
        except Exception:
            continue
        gg = g.assign(_bin=bins)
        for b, sub in gg.groupby("_bin"):
            rows.append({
                "model": m,
                "len_decile": int(b) if not np.isnan(b) else -1,
                "n": len(sub),
                "min_len_words": float(sub["len_words"].min()),
                "max_len_words": float(sub["len_words"].max()),
                "mean_len_words": float(sub["len_words"].mean()),
                "SAcc": sub["acc"].mean(),
                "IFAcc": sub["response_pass"].mean(),
                "HAcc": sub["joint"].mean(),
            })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values(["model", "len_decile"]).to_csv(out_dir / "mathifpaper_length_deciles.csv", index=False)


# ----------------------- 7. pass@1 vs best-of-16 vs maj@16 -----------------------

def passk_metrics(df: pd.DataFrame, out_dir: Path):
    rows = []
    for (m, eid), g in df.groupby(["model", "example_id"]):
        if g.empty:
            continue
        rows.append({
            "model": m,
            "example_id": eid,
            "k": len(g),
            "SAcc_pass@1": g["acc"].mean(),
            "SAcc_best@k": int(g["acc"].max()),
            "IFAcc_pass@1": g["response_pass"].mean(),
            "IFAcc_best@k": int(g["response_pass"].max()),
            "HAcc_pass@1": g["joint"].mean(),
            "HAcc_best@k": int(g["joint"].max()),
            "majority_acc": int(g["acc"].sum() > len(g) / 2),
            "majority_follow": int(g["response_pass"].sum() > len(g) / 2),
            "majority_joint": int(g["joint"].sum() > len(g) / 2),
            "difficulty": g["difficulty"].iloc[0] if "difficulty" in g else "",
            "source": g["source"].iloc[0] if "source" in g else "",
        })
    pdf = pd.DataFrame(rows)
    summary = pdf.groupby("model").agg(
        n_prompts=("example_id", "nunique"),
        SAcc_pass1=("SAcc_pass@1", "mean"),
        SAcc_bestK=("SAcc_best@k", "mean"),
        SAcc_majK=("majority_acc", "mean"),
        IFAcc_pass1=("IFAcc_pass@1", "mean"),
        IFAcc_bestK=("IFAcc_best@k", "mean"),
        IFAcc_majK=("majority_follow", "mean"),
        HAcc_pass1=("HAcc_pass@1", "mean"),
        HAcc_bestK=("HAcc_best@k", "mean"),
        HAcc_majK=("majority_joint", "mean"),
    ).reset_index()
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary.sort_values("model").to_csv(out_dir / "mathifpaper_passk.csv", index=False)
    pdf.to_csv(out_dir / "mathifpaper_perprompt_passk.csv", index=False)


# ----------------------- 8. composition decay (single/double/triple) -----------------------

def composition_decay(df: pd.DataFrame, out_dir: Path):
    rows = []
    for (m, d), g in df.groupby(["model", "difficulty"]):
        rows.append({
            "model": m,
            "difficulty": d,
            "n": len(g),
            "SAcc": g["acc"].mean(),
            "IFAcc_strict": g["response_pass"].mean(),
            "HAcc_strict": g["joint"].mean(),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out.sort_values(["difficulty", "model"]).to_csv(out_dir / "mathifpaper_composition_decay.csv", index=False)

    # Implied per-constraint pass rate under independence assumption:
    #   IF_strict(d=k) = p^k  =>  p = IF_strict ** (1/k)
    rows2 = []
    for m, g in out.groupby("model"):
        for d, sub in g.groupby("difficulty"):
            k = {"single": 1, "double": 2, "triple": 3}.get(d, np.nan)
            if k == k and len(sub) and sub["IFAcc_strict"].iloc[0] > 0:
                p = sub["IFAcc_strict"].iloc[0] ** (1.0 / k)
            else:
                p = float("nan")
            rows2.append({"model": m, "difficulty": d, "k": k,
                          "IFAcc_strict": float(sub["IFAcc_strict"].iloc[0]),
                          "implied_per_constraint_p": p})
    pd.DataFrame(rows2).to_csv(out_dir / "mathifpaper_composition_implied_p.csv", index=False)


# ----------------------- 9. RL effect direction (delta vs base) -----------------------

def deltas(df: pd.DataFrame, out_dir: Path):
    base = _ag(df, ["model"]).set_index("model").loc["base"]
    rows = []
    for m in MODEL_ORDER:
        if m == "base":
            continue
        sub = _ag(df, ["model"]).set_index("model").loc[m]
        rows.append({
            "model": m,
            "ΔSAcc": sub["SAcc"] - base["SAcc"],
            "ΔIFAcc": sub["IFAcc"] - base["IFAcc"],
            "ΔHAcc": sub["HAcc"] - base["HAcc"],
            "ΔGap": (sub["SAcc"] - sub["HAcc"]) - (base["SAcc"] - base["HAcc"]),
            "Δlen_words": sub["len_words"] - base["len_words"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "mathifpaper_deltas_vs_base.csv", index=False)


# ----------------------- driver -----------------------

def main():
    df = load_mathif()
    out_dir = ROOT / "analysis" / "tables"
    overall(df, out_dir)
    by_source(df, out_dir)
    by_difficulty(df, out_dir)
    by_difficulty_source(df, out_dir)
    by_macro_constraint(df, out_dir)
    coupling(df, out_dir)
    length_bins(df, out_dir)
    passk_metrics(df, out_dir)
    composition_decay(df, out_dir)
    deltas(df, out_dir)
    print("wrote MathIF paper-style tables to", out_dir)


if __name__ == "__main__":
    main()
