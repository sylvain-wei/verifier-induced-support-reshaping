#!/usr/bin/env python
"""Fine-grained aggregation of v2 scored MathIF / ReasonIF rollouts.

Produces 7 CSV tables that answer the two headline questions:

  ReasonIF — did the model learn to follow instructions DURING reasoning, or
             did it simply skip reasoning to dodge the constraint?
  MathIF  —  does IF-RLVR transfer to math-domain IF, or did it stop at
             surface shortcuts on plain IFEval/IFBench?
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import read_records  # noqa: E402


# ---------------------------- helpers ----------------------------

def _load(paths: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.extend(read_records(p))
    df = pd.DataFrame(rows)
    return df


def _mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s.dropna()) else float("nan")


def _rate(s: pd.Series, value: str) -> float:
    return float((s == value).mean()) if len(s) else float("nan")


# ---------------------------- ReasonIF ----------------------------

def aggregate_reasonif_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model_id, checkpoint) headline numbers including the
    `reasoning vs response` IF gap and the `skip-reasoning` rate.
    """
    rows = []
    for (model, step), g in df.groupby(["model_id", "checkpoint_step"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique(),
            "wrapper_rate": _mean(g["wrapper_present"].astype(float, errors="ignore")),
            "skip_reasoning_rate": _mean(g["skip_reasoning"].astype(float, errors="ignore")),
            "mean_reasoning_words": _mean(g["reasoning_word_count"]),
            "mean_response_words": _mean(g["response_word_count"]),
            "acc_pass@1": _mean(g["acc"]),
            "reasoning_strict_pass@1": _mean(g["reasoning_strict_pass"]),
            "response_strict_pass@1": _mean(g["response_strict_pass"]),
            "reasoning_soft_pass@1": _mean(g["reasoning_soft_pass"]),
            "response_soft_pass@1": _mean(g["response_soft_pass"]),
            "reasoning_minus_response": _mean(g["reasoning_strict_pass"]) - _mean(g["response_strict_pass"]),
            "joint_reasoning_pass@1": _mean(g["joint_reasoning_pass"]),
            "joint_response_pass@1": _mean(g["joint_response_pass"]),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
            "dai_rate": _rate(g["opening_mode"], "DAI"),
            "csi_rate": _rate(g["opening_mode"], "CSI"),
            "other_rate": _rate(g["opening_mode"], "Other"),
        })
    return pd.DataFrame(rows).sort_values(["model_id", "checkpoint_step"])


def aggregate_reasonif_by_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """Per constraint class — the central comparison for ReasonIF's 6 categories."""
    rows = []
    for (model, step, cat, name), g in df.groupby(["model_id", "checkpoint_step", "primary_category", "primary_constraint"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "primary_category": cat,
            "primary_constraint": name,
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique(),
            "acc_pass@1": _mean(g["acc"]),
            "reasoning_strict_pass@1": _mean(g["reasoning_strict_pass"]),
            "response_strict_pass@1": _mean(g["response_strict_pass"]),
            "reasoning_minus_response": _mean(g["reasoning_strict_pass"]) - _mean(g["response_strict_pass"]),
            "wrapper_rate": _mean(g["wrapper_present"].astype(float, errors="ignore")),
            "skip_reasoning_rate": _mean(g["skip_reasoning"].astype(float, errors="ignore")),
            "mean_reasoning_words": _mean(g["reasoning_word_count"]),
            "joint_reasoning_pass@1": _mean(g["joint_reasoning_pass"]),
            "joint_response_pass@1": _mean(g["joint_response_pass"]),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
            "dai_rate": _rate(g["opening_mode"], "DAI"),
        })
    return pd.DataFrame(rows).sort_values(["primary_category", "primary_constraint", "model_id", "checkpoint_step"])


def aggregate_reasonif_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Per task source (gpqa / aime / arc / amc / gsm8k) — does the IF behaviour
    differ between math-flavoured (aime/amc/gsm8k) and science (gpqa/arc) prompts?
    """
    rows = []
    for (model, step, src), g in df.groupby(["model_id", "checkpoint_step", "source"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "source": src,
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique(),
            "acc_pass@1": _mean(g["acc"]),
            "reasoning_strict_pass@1": _mean(g["reasoning_strict_pass"]),
            "response_strict_pass@1": _mean(g["response_strict_pass"]),
            "reasoning_minus_response": _mean(g["reasoning_strict_pass"]) - _mean(g["response_strict_pass"]),
            "skip_reasoning_rate": _mean(g["skip_reasoning"].astype(float, errors="ignore")),
            "mean_reasoning_words": _mean(g["reasoning_word_count"]),
            "joint_reasoning_pass@1": _mean(g["joint_reasoning_pass"]),
        })
    return pd.DataFrame(rows).sort_values(["source", "model_id"])


def aggregate_reasonif_skip_decomp(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose response_strict_pass into:
       - skipped-reasoning AND followed-via-response  (constraint trivially holds because no reasoning)
       - real-reasoning AND followed-during-reasoning (genuine following)
    """
    rows = []
    for (model, step), g in df.groupby(["model_id", "checkpoint_step"], dropna=False):
        n = len(g)
        skipped = g[g["skip_reasoning"] == True]  # noqa: E712
        reasoned = g[g["skip_reasoning"] == False]  # noqa: E712
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "n_responses": n,
            "skip_reasoning_rate": len(skipped) / n if n else float("nan"),
            "skip_and_response_pass": _mean(skipped["response_strict_pass"]),
            "skip_and_acc": _mean(skipped["acc"]),
            "reason_and_reasoning_pass": _mean(reasoned["reasoning_strict_pass"]),
            "reason_and_acc": _mean(reasoned["acc"]),
            "reason_and_joint_pass": _mean(reasoned["joint_reasoning_pass"]),
        })
    return pd.DataFrame(rows).sort_values("model_id")


# ---------------------------- MathIF ----------------------------

def aggregate_mathif_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, step), g in df.groupby(["model_id", "checkpoint_step"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique(),
            "acc_pass@1": _mean(g["acc"]),
            "follow_strict_pass@1": _mean(g["follow_strict"]),
            "follow_soft_pass@1": _mean(g["follow_soft"]),
            "joint_pass@1": _mean(g["joint_acc_follow"]),
            "mean_response_words": _mean(g["response_length_words"]),
            "mean_response_chars": _mean(g["response_length_chars"]),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
            "dai_rate": _rate(g["opening_mode"], "DAI"),
            "csi_rate": _rate(g["opening_mode"], "CSI"),
            "other_rate": _rate(g["opening_mode"], "Other"),
        })
    return pd.DataFrame(rows).sort_values(["model_id", "checkpoint_step"])


def aggregate_mathif_per_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten per-constraint passes — the central table for MathIF's 15 constraints."""
    rows = []
    for _, r in df.iterrows():
        for pc in r.get("per_constraint") or []:
            if not isinstance(pc, dict) or not pc.get("supported"):
                continue
            rows.append({
                "model_id": r["model_id"],
                "checkpoint_step": r["checkpoint_step"],
                "source": r.get("source"),
                "n_constraints": r.get("n_constraints"),
                "constraint_name": pc.get("name"),
                "constraint_category": pc.get("category"),
                "passed": int(pc.get("passed") is True),
                "verifier_source": pc.get("source"),
                "acc": r.get("acc"),
                "joint_acc_follow": r.get("joint_acc_follow"),
                "opening_mode": r.get("opening_mode"),
            })
    if not rows:
        return pd.DataFrame()
    cdf = pd.DataFrame(rows)
    grp = cdf.groupby(["model_id", "checkpoint_step", "constraint_category", "constraint_name"], dropna=False)
    return grp.agg(
        n=("passed", "size"),
        pass_rate=("passed", "mean"),
        acc=("acc", "mean"),
        joint=("joint_acc_follow", "mean"),
    ).reset_index().sort_values(["constraint_category", "constraint_name", "model_id"])


def aggregate_mathif_by_source(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, step, src), g in df.groupby(["model_id", "checkpoint_step", "source"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "source": src,
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique(),
            "acc_pass@1": _mean(g["acc"]),
            "follow_strict_pass@1": _mean(g["follow_strict"]),
            "follow_soft_pass@1": _mean(g["follow_soft"]),
            "joint_pass@1": _mean(g["joint_acc_follow"]),
            "mean_response_words": _mean(g["response_length_words"]),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
        })
    return pd.DataFrame(rows).sort_values(["source", "model_id"])


def aggregate_mathif_by_n_constraints(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, step, n), g in df.groupby(["model_id", "checkpoint_step", "n_constraints"], dropna=False):
        rows.append({
            "model_id": model,
            "checkpoint_step": step,
            "n_constraints": n,
            "n_responses": len(g),
            "acc_pass@1": _mean(g["acc"]),
            "follow_strict_pass@1": _mean(g["follow_strict"]),
            "follow_soft_pass@1": _mean(g["follow_soft"]),
            "joint_pass@1": _mean(g["joint_acc_follow"]),
        })
    return pd.DataFrame(rows).sort_values(["n_constraints", "model_id"])


# ---------------------------- main ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathif-inputs", nargs="+", required=True)
    ap.add_argument("--reasonif-inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "analysis" / "tables"))
    ap.add_argument("--prefix", default="mathif_reasonif_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = _load(args.mathif_inputs)
    rdf = _load(args.reasonif_inputs)

    paths = {}

    paths["mathif_overall"] = out_dir / f"{args.prefix}_mathif_overall.csv"
    aggregate_mathif_overall(mdf).to_csv(paths["mathif_overall"], index=False)

    paths["mathif_per_constraint"] = out_dir / f"{args.prefix}_mathif_per_constraint.csv"
    aggregate_mathif_per_constraint(mdf).to_csv(paths["mathif_per_constraint"], index=False)

    paths["mathif_by_source"] = out_dir / f"{args.prefix}_mathif_by_source.csv"
    aggregate_mathif_by_source(mdf).to_csv(paths["mathif_by_source"], index=False)

    paths["mathif_by_n_constraints"] = out_dir / f"{args.prefix}_mathif_by_n_constraints.csv"
    aggregate_mathif_by_n_constraints(mdf).to_csv(paths["mathif_by_n_constraints"], index=False)

    paths["reasonif_overall"] = out_dir / f"{args.prefix}_reasonif_overall.csv"
    aggregate_reasonif_overall(rdf).to_csv(paths["reasonif_overall"], index=False)

    paths["reasonif_by_constraint"] = out_dir / f"{args.prefix}_reasonif_by_constraint.csv"
    aggregate_reasonif_by_constraint(rdf).to_csv(paths["reasonif_by_constraint"], index=False)

    paths["reasonif_by_source"] = out_dir / f"{args.prefix}_reasonif_by_source.csv"
    aggregate_reasonif_by_source(rdf).to_csv(paths["reasonif_by_source"], index=False)

    paths["reasonif_skip_decomp"] = out_dir / f"{args.prefix}_reasonif_skip_decomp.csv"
    aggregate_reasonif_skip_decomp(rdf).to_csv(paths["reasonif_skip_decomp"], index=False)

    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
