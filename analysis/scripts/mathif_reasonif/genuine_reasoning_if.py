#!/usr/bin/env python
"""Compute the three layered ReasonIF metrics:

  1) reasoning_pass_raw       = reasoning_follow_strict == 1
  2) reasoning_pass_genuine   = raw=1 AND bypass=0
  3) joint_reason_correct     = genuine=1 AND acc=1

Plus per-family breakdown and the response_vs_reasoning IF gap that exposes
the "bypass / answer-side" surface shortcut.
"""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import read_records  # noqa: E402

MODEL_ORDER = ["base", "math_rlvr", "if_rlvr"]


def load_reasonif() -> pd.DataFrame:
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
    df["reasoning_pass_raw"] = (df["reasoning_follow_strict"] == 1.0).astype(int)
    df["response_pass"] = (df["response_follow_strict"] == 1.0).astype(int)
    df["bypass"] = df["bypass_reasoning"].astype(bool)
    df["genuine_pass"] = ((df["reasoning_pass_raw"] == 1) & (~df["bypass"])).astype(int)
    df["joint_reason_correct"] = ((df["genuine_pass"] == 1) & (df["acc"] == 1)).astype(int)
    df["family"] = df["constraint_families"].apply(
        lambda v: v[0] if isinstance(v, list) and v else (str(v) if v else "")
    )
    return df


def overall(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, g in df.groupby("model"):
        out.append({
            "model": m,
            "n": len(g),
            "acc": g["acc"].mean(),
            "reasoning_follow_raw": g["reasoning_pass_raw"].mean(),
            "bypass_rate": g["bypass"].mean(),
            "reasoning_follow_genuine": g["genuine_pass"].mean(),
            "joint_reason_correct": g["joint_reason_correct"].mean(),
            "response_follow_strict": g["response_pass"].mean(),
            "answerside_shortcut_gap": g["response_pass"].mean() - g["reasoning_pass_raw"].mean(),
            "mean_reasoning_words": g["reasoning_word_count"].mean(),
        })
    df_out = pd.DataFrame(out)
    df_out["model"] = pd.Categorical(df_out["model"], MODEL_ORDER, ordered=True)
    return df_out.sort_values("model")


def per_family(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fam, m), g in df.groupby(["family", "model"]):
        out.append({
            "family": fam,
            "model": m,
            "n": len(g),
            "reasoning_follow_raw": g["reasoning_pass_raw"].mean(),
            "bypass_rate": g["bypass"].mean(),
            "reasoning_follow_genuine": g["genuine_pass"].mean(),
            "joint_reason_correct": g["joint_reason_correct"].mean(),
            "answerside_shortcut_gap": g["response_pass"].mean() - g["reasoning_pass_raw"].mean(),
        })
    df_out = pd.DataFrame(out)
    df_out["model"] = pd.Categorical(df_out["model"], MODEL_ORDER, ordered=True)
    return df_out.sort_values(["family", "model"])


def per_source(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, m), g in df.groupby(["source", "model"]):
        out.append({
            "source": s,
            "model": m,
            "n": len(g),
            "acc": g["acc"].mean(),
            "reasoning_follow_raw": g["reasoning_pass_raw"].mean(),
            "bypass_rate": g["bypass"].mean(),
            "reasoning_follow_genuine": g["genuine_pass"].mean(),
            "joint_reason_correct": g["joint_reason_correct"].mean(),
        })
    df_out = pd.DataFrame(out)
    df_out["model"] = pd.Categorical(df_out["model"], MODEL_ORDER, ordered=True)
    return df_out.sort_values(["source", "model"])


def main():
    out_dir = ROOT / "analysis" / "tables"
    df = load_reasonif()
    a = overall(df)
    b = per_family(df)
    c = per_source(df)
    a.to_csv(out_dir / "reasonif_genuine_overview.csv", index=False)
    b.to_csv(out_dir / "reasonif_genuine_per_family.csv", index=False)
    c.to_csv(out_dir / "reasonif_genuine_per_source.csv", index=False)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    print("== overall ==\n", a.to_string(index=False))
    print("\n== per family ==\n", b.to_string(index=False))
    print("\n== per source ==\n", c.to_string(index=False))


if __name__ == "__main__":
    main()
