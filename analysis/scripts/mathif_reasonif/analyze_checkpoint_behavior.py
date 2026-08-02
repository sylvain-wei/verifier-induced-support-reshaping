#!/usr/bin/env python
"""Aggregate scored MathIF/ReasonIF responses into paper-style tables."""
from __future__ import annotations
import os

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import read_records  # noqa: E402

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))


def _mean(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").mean())


def _rate(s: pd.Series, value: str) -> float:
    if len(s) == 0:
        return float("nan")
    return float((s == value).mean())


def _safe_model_label(row: pd.Series) -> str:
    model = str(row.get("model_id") or "unknown")
    step = int(row.get("checkpoint_step", -1)) if not pd.isna(row.get("checkpoint_step", -1)) else -1
    return f"{model}@{step}" if step >= 0 else model


def _stringify(v):
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    if v is None:
        return ""
    return str(v)


def load_scored(paths: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_records(path))
    if not rows:
        raise SystemExit("No scored records loaded")
    df = pd.DataFrame(rows)
    for col in ("run_id", "model_id", "benchmark", "instruction_type", "opening_mode"):
        if col not in df:
            df[col] = ""
    if "checkpoint_step" not in df:
        df["checkpoint_step"] = -1
    for col in ("acc", "follow_soft", "follow_strict", "joint_acc_follow", "response_length_chars", "response_length_words", "num_segments"):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_step"] = df.apply(_safe_model_label, axis=1)
    return df


def aggregate_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["benchmark", "run_id", "model_id", "checkpoint_step"]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update({
            "n_responses": len(g),
            "n_prompts": g["example_id"].nunique() if "example_id" in g else float("nan"),
            "acc_pass@1": _mean(g["acc"]),
            "follow_soft": _mean(g["follow_soft"]),
            "follow_strict_pass@1": _mean(g["follow_strict"]),
            "joint_pass@1": _mean(g["joint_acc_follow"]),
            "mean_len_chars": _mean(g["response_length_chars"]),
            "mean_len_words": _mean(g["response_length_words"]),
            "mean_num_segments": _mean(g["num_segments"]),
            "dri_rate": _rate(g["opening_mode"], "DRI"),
            "dai_rate": _rate(g["opening_mode"], "DAI"),
            "csi_rate": _rate(g["opening_mode"], "CSI"),
            "other_rate": _rate(g["opening_mode"], "Other"),
        })
        if "first_violation_phase" in g:
            row["early_violation_rate"] = _rate(g["first_violation_phase"].fillna(""), "early")
            row["middle_violation_rate"] = _rate(g["first_violation_phase"].fillna(""), "middle")
            row["late_violation_rate"] = _rate(g["first_violation_phase"].fillna(""), "late")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def aggregate_prompt_support(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["benchmark", "run_id", "model_id", "checkpoint_step", "example_id"]
    for key, g in df.groupby(group_cols, dropna=False):
        k = len(g)
        row = dict(zip(group_cols, key))
        for metric, col in [("acc", "acc"), ("follow", "follow_strict"), ("joint", "joint_acc_follow")]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            cnt = int((vals == 1).sum())
            row[f"{metric}_count"] = cnt
            row[f"{metric}_pass@1"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_best@k"] = 1.0 if cnt > 0 else 0.0 if len(vals) else float("nan")
            row[f"{metric}_bin"] = "all" if cnt == k and k else "zero" if cnt == 0 else "partial"
        row["k"] = k
        it = g["instruction_type"].iloc[0] if "instruction_type" in g else ""
        row["instruction_type"] = _stringify(it)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def aggregate_support_bins(per_prompt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["benchmark", "run_id", "model_id", "checkpoint_step"]
    for key, g in per_prompt.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        for metric in ("acc", "follow", "joint"):
            vc = g[f"{metric}_bin"].value_counts(normalize=False).to_dict()
            row = {**base, "metric": metric, "n_prompts": len(g)}
            for b in ("zero", "partial", "all"):
                row[f"{b}_count"] = int(vc.get(b, 0))
                row[f"{b}_rate"] = float(vc.get(b, 0) / len(g)) if len(g) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["metric"])


def aggregate_mode_conditioned(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["benchmark", "run_id", "model_id", "checkpoint_step", "opening_mode"]
    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update({
            "n_responses": len(g),
            "acc": _mean(g["acc"]),
            "follow_strict": _mean(g["follow_strict"]),
            "follow_soft": _mean(g["follow_soft"]),
            "joint": _mean(g["joint_acc_follow"]),
            "mean_len_chars": _mean(g["response_length_chars"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def aggregate_constraint_categories(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pcs = r.get("per_constraint") or []
        if isinstance(pcs, str):
            try:
                pcs = json.loads(pcs)
            except Exception:
                pcs = []
        for pc in pcs:
            if not isinstance(pc, dict) or not pc.get("supported"):
                continue
            rows.append({
                "benchmark": r.get("benchmark"),
                "run_id": r.get("run_id"),
                "model_id": r.get("model_id"),
                "checkpoint_step": r.get("checkpoint_step"),
                "instruction_type": _stringify(r.get("instruction_type")),
                "constraint_category": _stringify(pc.get("category")),
                "constraint_rule": _stringify(pc.get("rule")),
                "passed": int(pc.get("passed") is True),
            })
    if not rows:
        return pd.DataFrame(columns=["benchmark", "run_id", "model_id", "checkpoint_step", "instruction_type", "constraint_category", "constraint_rule", "n", "pass_rate"])
    cdf = pd.DataFrame(rows)
    group_cols = ["benchmark", "run_id", "model_id", "checkpoint_step", "instruction_type", "constraint_category", "constraint_rule"]
    return cdf.groupby(group_cols, dropna=False).agg(n=("passed", "size"), pass_rate=("passed", "mean")).reset_index().sort_values(group_cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Scored JSONL files from score_mathif/score_reasonif")
    ap.add_argument("--out-dir", default=str(ROOT / "analysis" / "tables"))
    ap.add_argument("--prefix", default="mathif_reasonif")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scored(args.inputs)

    checkpoint = aggregate_checkpoint(df)
    per_prompt = aggregate_prompt_support(df)
    support_bins = aggregate_support_bins(per_prompt)
    mode_cond = aggregate_mode_conditioned(df)
    constraint = aggregate_constraint_categories(df)

    paths = {
        "checkpoint_metrics": out_dir / f"{args.prefix}_checkpoint_metrics.csv",
        "per_prompt_support": out_dir / f"{args.prefix}_per_prompt_support.csv",
        "support_bins": out_dir / f"{args.prefix}_support_bins.csv",
        "mode_conditioned": out_dir / f"{args.prefix}_mode_conditioned.csv",
        "constraint_categories": out_dir / f"{args.prefix}_constraint_categories.csv",
    }
    checkpoint.to_csv(paths["checkpoint_metrics"], index=False)
    per_prompt.to_csv(paths["per_prompt_support"], index=False)
    support_bins.to_csv(paths["support_bins"], index=False)
    mode_cond.to_csv(paths["mode_conditioned"], index=False)
    constraint.to_csv(paths["constraint_categories"], index=False)

    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
