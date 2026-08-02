#!/usr/bin/env python
"""Build a 128-row stratified MATH-500 validation parquet for E1.

Source
------
${PROJECT_ROOT}/data/math500/test.parquet
  500 rows, schema [data_source, prompt, ability, reward_model, extra_info].
  data_source = "math_dapo", reward_model.style = "rule".
  extra_info has {'index', 'level' ('1'..'5'), 'raw_problem', 'source', 'split', 'type'}.

Sampling strategy
-----------------
Stratified by ``extra_info.level`` (5 difficulty buckets), proportional to
each level's frequency in the source. Counts in source: 1=43 / 2=90 /
3=105 / 4=128 / 5=134 (total 500). For 128 rows: 11 / 23 / 27 / 33 / 34
(rounded to sum 128). Random with a fixed seed for reproducibility.

Why not just AIME for E1 math probe?
------------------------------------
AIME24+25 (60 problems) is hard and high-variance. MATH-500-128 gives a
broader, slightly easier surface to detect a math-support collapse — when
OPD trained on IF-train pulls the student off the math distribution, the
all-wrong group rate on MATH-500 should jump well before AIME.

Output
------
${PROJECT_ROOT}/data/math500/math500_128_opd_val.parquet

Schema is identical to source — no transformation, just a 128-row subsample
with a deterministic seed.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))
SRC_DEFAULT = os.path.join(PROJECT_ROOT, "data", "math500", "test.parquet")
DST_DEFAULT = os.path.join(PROJECT_ROOT, "data", "math500", "math500_128_opd_val.parquet")

# 128-row strata. Computed from source proportions: 43/90/105/128/134 of 500.
STRATA = {"1": 11, "2": 23, "3": 27, "4": 33, "5": 34}  # sums to 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"[build_math500_128] reading {args.src}")
    df = pd.read_parquet(args.src)
    print(f"[build_math500_128]   rows: {len(df):,}  cols: {df.columns.tolist()}")

    if len(df) != 500:
        print(f"[build_math500_128] WARNING: expected 500 rows, got {len(df)}")

    # Pull level out of extra_info
    df = df.copy()
    df["_level"] = df["extra_info"].apply(
        lambda d: str(d.get("level")) if isinstance(d, dict) else None
    )
    print(f"[build_math500_128]   level value_counts: "
          f"{df['_level'].value_counts().sort_index().to_dict()}")

    rng = np.random.default_rng(args.seed)
    chunks = []
    for lvl, n in STRATA.items():
        sub = df[df["_level"] == lvl]
        if len(sub) < n:
            sys.exit(f"[build_math500_128] ERROR: level {lvl} has only {len(sub)} rows; needed {n}")
        idx = rng.choice(sub.index.values, size=n, replace=False)
        chunks.append(sub.loc[idx])
        print(f"[build_math500_128]   level {lvl}: sampled {n} of {len(sub)}")

    out = pd.concat(chunks, axis=0).sort_index().drop(columns=["_level"]).reset_index(drop=True)
    if len(out) != sum(STRATA.values()):
        sys.exit(f"[build_math500_128] ERROR: stratum count mismatch ({len(out)})")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    out.to_parquet(args.dst, index=False)
    print(f"[build_math500_128] wrote {args.dst}  ({len(out)} rows)")

    # Sanity readout
    chk = pd.read_parquet(args.dst)
    print()
    print("--- output sanity ---")
    print(f"  shape: {chk.shape}")
    print(f"  cols: {chk.columns.tolist()}")
    print(f"  data_source: {chk['data_source'].value_counts().to_dict()}")
    print(f"  ability: {chk['ability'].value_counts().to_dict()}")
    rm = chk.iloc[0]["reward_model"]
    print(f"  row 0 reward_model: {rm}")
    print(f"  row 0 prompt[0].content (head): {str(chk.iloc[0]['prompt'][0]['content'])[:120]!r}")


if __name__ == "__main__":
    main()
