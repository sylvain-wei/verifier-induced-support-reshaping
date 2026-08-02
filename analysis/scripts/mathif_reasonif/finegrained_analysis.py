#!/usr/bin/env python
"""Fine-grained MathIF/ReasonIF analysis answering three questions:

Q1 (ReasonIF): does the model actually follow instructions *while reasoning*,
   or does it bypass reasoning to satisfy the verifier? Per-family breakdown.
Q2 (MathIF):   does IF-RLVR transfer to math-domain IF, or is it a generic
   surface-shortcut? Per source/difficulty/family breakdown.
Q3 (Both):     mechanism of Math-RLVR vs IF-RLVR behavior change.
"""
from __future__ import annotations
import os

import argparse
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


def load_all() -> pd.DataFrame:
    rows: List[Dict] = []
    for ds in ("mathif", "reasonif"):
        for m in MODEL_ORDER:
            p = ROOT / "analysis" / "tables" / f"v2_scored_{m}_{ds}.jsonl"
            for r in read_records(p):
                r["model"] = m
                r["dataset"] = ds
                rows.append(r)
    df = pd.DataFrame(rows)
    for col in ("acc", "reasoning_follow_strict", "reasoning_follow_soft",
                "response_follow_strict", "response_follow_soft",
                "joint_acc_follow", "reasoning_word_count",
                "response_length_words", "n_supported_constraints"):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bypass_reasoning"] = df.get("bypass_reasoning", False).astype(bool)
    df["has_answer_tag"] = df.get("has_answer_tag", False).astype(bool)
    df["family"] = df["constraint_families"].apply(
        lambda v: ",".join(v) if isinstance(v, list) else str(v))
    return df


# ----------------------------- helpers -----------------------------

def _rate(s: pd.Series, value) -> float:
    return float((s == value).mean()) if len(s) else float("nan")


def _agg(g: pd.DataFrame) -> Dict[str, float]:
    return {
        "n": len(g),
        "acc": float(pd.to_numeric(g["acc"], errors="coerce").mean()),
        "reasoning_follow_strict": float(g["reasoning_follow_strict"].mean()),
        "response_follow_strict": float(g["response_follow_strict"].mean()),
        "joint": float(pd.to_numeric(g["joint_acc_follow"], errors="coerce").mean()),
        "reasoning_word_count": float(g["reasoning_word_count"].mean()),
        "response_length_words": float(g["response_length_words"].mean()),
        "bypass_rate": float(g["bypass_reasoning"].mean()),
        "has_answer_tag_rate": float(g["has_answer_tag"].mean()),
    }


# ----------------------------- Q1 ReasonIF ----------------------------

def q1_reasonif(df: pd.DataFrame, out_dir: Path) -> None:
    rdf = df[df["dataset"] == "reasonif"].copy()
    # explode per-constraint rows for trace and response side-by-side
    rows = []
    for _, r in rdf.iterrows():
        rps = r.get("reasoning_per_constraint") or []
        ops = r.get("response_per_constraint") or []
        for i, pc in enumerate(rps):
            if not pc.get("supported"):
                continue
            other = ops[i] if i < len(ops) else {}
            rows.append({
                "model": r["model"],
                "source": r.get("source"),
                "family": pc.get("family"),
                "instruction_name": pc.get("instruction_name"),
                "reasoning_pass": int(pc.get("passed") is True),
                "response_pass": int(other.get("passed") is True),
                "bypass": bool(r.get("bypass_reasoning")),
                "has_answer_tag": bool(r.get("has_answer_tag")),
                "acc": r.get("acc"),
                "reasoning_word_count": r.get("reasoning_word_count"),
            })
    er = pd.DataFrame(rows)

    # by model x family
    per_fam = er.groupby(["family", "model"]).agg(
        n=("reasoning_pass", "size"),
        reasoning_pass_rate=("reasoning_pass", "mean"),
        response_pass_rate=("response_pass", "mean"),
        bypass_rate=("bypass", "mean"),
        acc=("acc", "mean"),
        reasoning_words=("reasoning_word_count", "mean"),
    ).reset_index()
    per_fam["model"] = pd.Categorical(per_fam["model"], MODEL_ORDER, ordered=True)
    per_fam = per_fam.sort_values(["family", "model"])
    per_fam.to_csv(out_dir / "q1_reasonif_per_family.csv", index=False)

    # bypass-conditioned IF rate
    cond = er.groupby(["model", "bypass"]).agg(
        n=("reasoning_pass", "size"),
        reasoning_pass_rate=("reasoning_pass", "mean"),
        response_pass_rate=("response_pass", "mean"),
        acc=("acc", "mean"),
    ).reset_index()
    cond.to_csv(out_dir / "q1_reasonif_bypass_conditioned.csv", index=False)

    # source x model
    by_src = er.groupby(["source", "model"]).agg(
        n=("reasoning_pass", "size"),
        reasoning_pass_rate=("reasoning_pass", "mean"),
        response_pass_rate=("response_pass", "mean"),
        bypass_rate=("bypass", "mean"),
        acc=("acc", "mean"),
    ).reset_index()
    by_src.to_csv(out_dir / "q1_reasonif_per_source.csv", index=False)


# ----------------------------- Q2 MathIF ----------------------------

def q2_mathif(df: pd.DataFrame, out_dir: Path) -> None:
    mdf = df[df["dataset"] == "mathif"].copy()

    # main table per model
    per_model = mdf.groupby("model").apply(_agg).apply(pd.Series).reset_index()
    per_model["model"] = pd.Categorical(per_model["model"], MODEL_ORDER, ordered=True)
    per_model = per_model.sort_values("model")
    per_model.to_csv(out_dir / "q2_mathif_overview.csv", index=False)

    # by difficulty
    by_diff = mdf.groupby(["difficulty", "model"]).apply(_agg).apply(pd.Series).reset_index()
    by_diff["model"] = pd.Categorical(by_diff["model"], MODEL_ORDER, ordered=True)
    by_diff = by_diff.sort_values(["difficulty", "model"])
    by_diff.to_csv(out_dir / "q2_mathif_by_difficulty.csv", index=False)

    # by source
    by_src = mdf.groupby(["source", "model"]).apply(_agg).apply(pd.Series).reset_index()
    by_src["model"] = pd.Categorical(by_src["model"], MODEL_ORDER, ordered=True)
    by_src = by_src.sort_values(["source", "model"])
    by_src.to_csv(out_dir / "q2_mathif_by_source.csv", index=False)

    # explode per-constraint
    rows = []
    for _, r in mdf.iterrows():
        per = r.get("response_per_constraint") or []
        for pc in per:
            if not pc.get("supported"):
                continue
            rows.append({
                "model": r["model"],
                "source": r.get("source"),
                "difficulty": r.get("difficulty"),
                "family": pc.get("family"),
                "instruction_name": pc.get("instruction_name"),
                "response_pass": int(pc.get("passed") is True),
                "acc": r.get("acc"),
            })
    pdf = pd.DataFrame(rows)
    fam = pdf.groupby(["family", "model"]).agg(
        n=("response_pass", "size"),
        response_pass_rate=("response_pass", "mean"),
        acc=("acc", "mean"),
    ).reset_index()
    fam["model"] = pd.Categorical(fam["model"], MODEL_ORDER, ordered=True)
    fam = fam.sort_values(["family", "model"])
    fam.to_csv(out_dir / "q2_mathif_per_family.csv", index=False)

    # joint acc∧follow per (source, model)
    joint = mdf.groupby(["source", "model"]).agg(
        n=("acc", "size"),
        acc=("acc", "mean"),
        strict_follow=("response_follow_strict", "mean"),
        joint=("joint_acc_follow", "mean"),
    ).reset_index()
    joint["model"] = pd.Categorical(joint["model"], MODEL_ORDER, ordered=True)
    joint = joint.sort_values(["source", "model"])
    joint.to_csv(out_dir / "q2_mathif_joint_per_source.csv", index=False)


# ----------------------------- Q3 mechanism ----------------------------

def q3_mechanism(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        rows.append({
            "dataset": ds,
            "model": model,
            "n": len(g),
            "acc": float(pd.to_numeric(g["acc"], errors="coerce").mean()),
            "reasoning_follow_strict": float(g["reasoning_follow_strict"].mean()),
            "response_follow_strict": float(g["response_follow_strict"].mean()),
            "joint": float(pd.to_numeric(g["joint_acc_follow"], errors="coerce").mean()),
            "mean_reasoning_words": float(g["reasoning_word_count"].mean()),
            "mean_response_words": float(g["response_length_words"].mean()),
            "bypass_rate": float(g["bypass_reasoning"].mean()),
            "has_answer_tag_rate": float(g["has_answer_tag"].mean()),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
            "dai_rate": _rate(g["opening_mode"], "DAI"),
            "csi_rate": _rate(g["opening_mode"], "CSI"),
            "other_rate": _rate(g["opening_mode"], "Other"),
        })
    odf = pd.DataFrame(rows)
    odf["model"] = pd.Categorical(odf["model"], MODEL_ORDER, ordered=True)
    odf = odf.sort_values(["dataset", "model"])
    odf.to_csv(out_dir / "q3_overview.csv", index=False)

    # mode-conditioned acc & follow on each dataset
    rows = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        for mode in ("DRI", "DAI", "CSI", "Other"):
            sub = g[g["opening_mode"] == mode]
            if len(sub) == 0:
                continue
            rows.append({
                "dataset": ds,
                "model": model,
                "mode": mode,
                "n": len(sub),
                "fraction": len(sub) / len(g),
                "acc": float(pd.to_numeric(sub["acc"], errors="coerce").mean()),
                "reasoning_follow_strict": float(sub["reasoning_follow_strict"].mean()),
                "response_follow_strict": float(sub["response_follow_strict"].mean()),
                "joint": float(pd.to_numeric(sub["joint_acc_follow"], errors="coerce").mean()),
                "mean_reasoning_words": float(sub["reasoning_word_count"].mean()),
                "bypass_rate": float(sub["bypass_reasoning"].mean()),
            })
    md = pd.DataFrame(rows)
    md["model"] = pd.Categorical(md["model"], MODEL_ORDER, ordered=True)
    md = md.sort_values(["dataset", "model", "mode"])
    md.to_csv(out_dir / "q3_mode_conditioned.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "analysis" / "tables"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_all()
    print(f"loaded {len(df)} rows; reasonif={int((df['dataset']=='reasonif').sum())}, mathif={int((df['dataset']=='mathif').sum())}")
    q1_reasonif(df, out_dir)
    q2_mathif(df, out_dir)
    q3_mechanism(df, out_dir)
    print("wrote q1/q2/q3 tables to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
