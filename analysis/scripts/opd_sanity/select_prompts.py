#!/usr/bin/env python
"""Phase 0 — Select 200 stratified IF-train prompts for the OPD sanity check.

Why this script
---------------
Before running on-policy distillation (OPD) from base into the IF-RLVR teacher,
we want a sanity check on a small but decisive sample: does the IF teacher
prefer contentful responses or shortcut responses? For that comparison we need
a stratified sample across the IF-train constraint families that are most
prone to shortcut exploitation.

Constraint families
-------------------
We bin each IF-train row by its first instruction_id family
(`<family>:<sub>`). The 6 families below were chosen for two reasons: each has
≥3,500 rows of supply in `data/ifeval/train.parquet`, and each is a known
shortcut vector observed in the b2r1 IF-RLVR rollouts from the §H analysis:

    keywords            keyword-dump shortcut
    detectable_format   format-only shortcut (json wrap, brackets, hyphens)
    length_constraints  length-only shortcut (very short / very long)
    combination         repeat-prompt shortcut (the canonical CoT-incompatible)
    change_case         case-only shortcut (all caps / all lower)
    punctuation         punctuation-only shortcut (no_comma, exclamation)

We sample 33–34 per family for 200 prompts total. Seed is fixed for
reproducibility.

Output
------
analysis/data/opd_sanity/selected_prompts.parquet:
    prompt_id          stable hash of the user message
    prompt_text        the raw user message (string, no template)
    instruction_id_list  list[str] parsed from ground_truth
    ground_truth       original string-form ground_truth (passed to verify_ifeval)
    family             one of the 6 families above
    constraint         human-readable constraint description from the parquet
"""
import argparse
import ast
import hashlib
import os
import random
from typing import List, Optional

import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")
TRAIN_PARQUET = f"{ROOT}/data/ifeval/train.parquet"
OUT_PARQUET = f"{ROOT}/analysis/data/opd_sanity/selected_prompts.parquet"

FAMILIES = [
    "keywords",
    "detectable_format",
    "length_constraints",
    "combination",
    "change_case",
    "punctuation",
]
PER_FAMILY = 34  # 6 * 34 = 204 → trim to 200 by family priority
TOTAL = 200
SEED = 1234


def parse_iid_list(gt_str: str) -> Optional[List[str]]:
    try:
        L = ast.literal_eval(gt_str)
        if not L:
            return None
        return list(L[0].get("instruction_id", []))
    except Exception:
        return None


def first_family(iids: Optional[List[str]]) -> Optional[str]:
    if not iids:
        return None
    return iids[0].split(":", 1)[0]


def extract_user_text(messages) -> Optional[str]:
    """`messages` is a numpy array of {role,content} dicts; take the first user."""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", default=TRAIN_PARQUET)
    ap.add_argument("--out_parquet", default=OUT_PARQUET)
    ap.add_argument("--per_family", type=int, default=PER_FAMILY)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print(f"[selection] reading {args.in_parquet}")
    df = pd.read_parquet(args.in_parquet)
    print(f"[selection]   rows: {len(df)}")

    df["instruction_id_list"] = df["ground_truth"].apply(parse_iid_list)
    df["family"] = df["instruction_id_list"].apply(first_family)
    df["prompt_text"] = df["messages"].apply(extract_user_text)

    # Drop rows without prompt or family parse.
    keep = df["prompt_text"].notna() & df["family"].isin(FAMILIES)
    df = df[keep].reset_index(drop=True)
    print(f"[selection]   after family/family-text filter: {len(df)}")

    # Drop duplicate prompt_text (rare but possible).
    df["prompt_id"] = df["prompt_text"].apply(stable_id)
    df = df.drop_duplicates(subset="prompt_id").reset_index(drop=True)
    print(f"[selection]   after de-dup on prompt_text: {len(df)}")

    rng = random.Random(args.seed)
    picks = []
    for fam in FAMILIES:
        fam_df = df[df["family"] == fam]
        n_take = min(args.per_family, len(fam_df))
        idx = sorted(rng.sample(range(len(fam_df)), n_take))
        picks.append(fam_df.iloc[idx])
        print(f"[selection]   family={fam:24s}  pool={len(fam_df):>6d}  took={n_take}")

    selected = pd.concat(picks, ignore_index=True)
    # Trim to TOTAL prioritising even family coverage.
    if len(selected) > args.total:
        # Round-robin trim: drop tail rows from the largest families first.
        counts = selected["family"].value_counts().to_dict()
        drop_ids = []
        while len(selected) - len(drop_ids) > args.total:
            big_fam = max(counts, key=counts.get)
            cand = selected[selected["family"] == big_fam].index.tolist()
            cand = [i for i in cand if i not in drop_ids]
            drop_ids.append(cand[-1])  # drop tail
            counts[big_fam] -= 1
        selected = selected.drop(index=drop_ids).reset_index(drop=True)

    print(f"[selection] final n={len(selected)}")
    print(f"[selection] family distribution:")
    print(selected["family"].value_counts().to_string())

    out_cols = ["prompt_id", "prompt_text", "instruction_id_list",
                "ground_truth", "family", "constraint"]
    out = selected[out_cols].copy()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"[selection] wrote {args.out_parquet}")


if __name__ == "__main__":
    main()
