#!/usr/bin/env python
"""WAVE 0.2 — build stratified IFEval-100 / IFBench-100 prompt subsets.

Block A/B all use a 100-prompt subset of IFEval (out of 541) and a 100-prompt
subset of IFBench (out of 300). To keep the constraint-family mix balanced, we
stratify by the *primary* constraint family (= first instruction_id's prefix
before ':').

IFEval families (n=9): detectable_format, keywords, length_constraints,
  change_case, combination, startend, punctuation, detectable_content, language.

IFBench families (n=7): format, words, count, ratio, sentence, custom, repeat.

Sampling rule (per dataset):
  - target_n = 100
  - allocate roughly proportionally: round(100 * count(f) / total). Sum may
    differ from 100 by ±1–2 due to rounding; fix the residual by adding /
    dropping in the largest family.
  - within each family, sample uniformly without replacement (np.random.seed).

Output (one row per prompt):
  - `key`            : original IFEval/IFBench int key
  - `prompt_id`      : str(key) — matches load_rollouts.py prompt_id column
  - `prompt_text`    : first user message content
  - `constraint_family`: primary family (string before ':')
  - `instruction_id_list`: full list
  - `ground_truth`   : reward_model.ground_truth (json string)

Files written:
  - analysis/data/eval_subsets/ifeval_100.jsonl
  - analysis/data/eval_subsets/ifbench_100.jsonl
  - analysis/tables/eval_subsets_summary.csv  (family counts before/after)

Usage:
  python make_eval_subsets.py [--seed 7] [--n 100]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List

import numpy as np
import pandas as pd


ROOT = os.environ.get("PROJECT_ROOT", ".")
IFEVAL_PARQUET  = f"{ROOT}/data/ifeval/test.parquet"
IFBENCH_PARQUET = f"{ROOT}/data/ifbench/test.parquet"
OUT_DIR         = f"{ROOT}/analysis/data/eval_subsets"
SUMMARY_CSV     = f"{ROOT}/analysis/tables/eval_subsets_summary.csv"


def _family(inst_ids) -> str:
    if inst_ids is None or len(inst_ids) == 0:
        return "unknown"
    return str(inst_ids[0]).split(":")[0]


def _proportional_allocation(family_counts: dict, target_n: int) -> dict:
    """Largest-remainder method so the per-family target sums to exactly N."""
    total = sum(family_counts.values())
    raw   = {f: target_n * c / total for f, c in family_counts.items()}
    floor = {f: int(np.floor(v)) for f, v in raw.items()}
    rem   = sorted(((raw[f] - floor[f], f) for f in raw), reverse=True)
    deficit = target_n - sum(floor.values())
    for i in range(deficit):
        floor[rem[i][1]] += 1
    # clamp to family availability
    for f, n in list(floor.items()):
        floor[f] = min(n, family_counts[f])
    # if clamping created a residual, redistribute to families with headroom
    short = target_n - sum(floor.values())
    if short > 0:
        for f, _ in rem:
            head = family_counts[f] - floor[f]
            if head > 0:
                take = min(head, short)
                floor[f] += take
                short -= take
                if short == 0:
                    break
    return floor


def sample_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["constraint_family"] = df["instruction_id_list"].apply(_family)
    fam_counts = Counter(df["constraint_family"])
    alloc = _proportional_allocation(dict(fam_counts), n)

    rng = np.random.default_rng(seed)
    picks: List[int] = []
    for fam, take in alloc.items():
        idx_in_fam = df.index[df["constraint_family"] == fam].to_numpy()
        if take >= len(idx_in_fam):
            picks.extend(idx_in_fam.tolist())
        else:
            chosen = rng.choice(idx_in_fam, size=take, replace=False)
            picks.extend(chosen.tolist())
    sub = df.loc[sorted(picks)].reset_index(drop=True)
    return sub


def write_jsonl(sub: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for _, r in sub.iterrows():
            msgs = list(r["prompt"])
            content = msgs[0]["content"]
            row = {
                "key": int(r["key"]),
                "prompt_id": str(int(r["key"])),
                "prompt_text": content,
                "constraint_family": r["constraint_family"],
                "instruction_id_list": list(r["instruction_id_list"]),
                "ground_truth": r["reward_model"]["ground_truth"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[subset] wrote {n} rows → {out_path}")


def _fam_table(name: str, full: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    full_c = Counter(full["instruction_id_list"].apply(_family))
    sub_c  = Counter(sub["constraint_family"])
    fams = sorted(set(full_c) | set(sub_c))
    return pd.DataFrame([{
        "dataset": name,
        "family": f,
        "full_n":  full_c.get(f, 0),
        "sub_n":   sub_c.get(f, 0),
        "full_pct": full_c.get(f, 0) / max(1, len(full)) * 100,
        "sub_pct":  sub_c.get(f, 0) / max(1, len(sub))  * 100,
    } for f in fams])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    print(f"[subset] reading {IFEVAL_PARQUET}")
    ifeval = pd.read_parquet(IFEVAL_PARQUET)
    print(f"[subset] reading {IFBENCH_PARQUET}")
    ifbench = pd.read_parquet(IFBENCH_PARQUET)

    sub_ifeval  = sample_subset(ifeval,  args.n, args.seed)
    sub_ifbench = sample_subset(ifbench, args.n, args.seed + 1)

    write_jsonl(sub_ifeval,  f"{OUT_DIR}/ifeval_100.jsonl")
    write_jsonl(sub_ifbench, f"{OUT_DIR}/ifbench_100.jsonl")

    summary = pd.concat([
        _fam_table("ifeval",  ifeval,  sub_ifeval),
        _fam_table("ifbench", ifbench, sub_ifbench),
    ], ignore_index=True)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[subset] wrote {SUMMARY_CSV}")

    # print in-terminal preview
    print("\n=== IFEval-100 family breakdown ===")
    print(summary[summary["dataset"] == "ifeval"][["family", "full_n", "sub_n",
                                                   "full_pct", "sub_pct"]]
          .to_string(index=False))
    print("\n=== IFBench-100 family breakdown ===")
    print(summary[summary["dataset"] == "ifbench"][["family", "full_n", "sub_n",
                                                    "full_pct", "sub_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
