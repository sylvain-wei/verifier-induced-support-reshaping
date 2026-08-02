#!/usr/bin/env python
"""Build IF-train parquet for E1 OPD training.

Source
------
${PROJECT_ROOT}/data/ifeval/train.parquet
  95,368 rows already in verl-flat schema:
    key, messages, ground_truth, dataset, constraint_type, constraint,
    data_source, prompt, ability, reward_model, extra_info

  - data_source = "allenai/IF_multi_constraints_upto5"
  - prompt      = [{"role": "user", "content": "<text>"}]
  - reward_model.style = "rule"
  - reward_model.ground_truth = "<stringified IF instruction list>"

Transform
---------
We rewrite ``data_source`` to the short tag ``"ifeval_train"`` so the custom
reward dispatcher (opd_e1_reward_func.py) can route on a clean key. All other
columns are preserved verbatim — the prompt, ground_truth, and extra_info are
already verl-compatible.

Output
------
${PROJECT_ROOT}/opd-lab/data/processed/ifeval_train_opd.parquet

Sanity (printed at end):
  - row count == 95,368
  - data_source value_counts = {"ifeval_train": 95368}
  - first row's reward_model has style="rule" and a non-empty ground_truth string
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))
SRC_DEFAULT = os.path.join(PROJECT_ROOT, "data", "ifeval", "train.parquet")
DST_DEFAULT = os.path.join(PROJECT_ROOT, "opd-lab", "data", "processed", "ifeval_train_opd.parquet")
NEW_DATA_SOURCE = "ifeval_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, write only the first N rows (for smoke).")
    args = ap.parse_args()

    print(f"[build_ifeval] reading {args.src}")
    df = pd.read_parquet(args.src)
    print(f"[build_ifeval]   rows: {len(df):,}  cols: {df.columns.tolist()}")

    # Schema sanity
    required = {"prompt", "data_source", "ability", "reward_model", "extra_info"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[build_ifeval] ERROR: missing required columns: {missing}")

    # Verify reward_model has the expected shape on a sample
    rm0 = df.iloc[0]["reward_model"]
    if not (isinstance(rm0, dict) and "ground_truth" in rm0 and "style" in rm0):
        sys.exit(f"[build_ifeval] ERROR: row 0 reward_model malformed: {rm0!r}")
    if rm0["style"] != "rule":
        print(f"[build_ifeval] WARNING: row 0 reward_model.style={rm0['style']!r} (expected 'rule')")

    # Verify prompt shape (list of {role, content})
    p0 = df.iloc[0]["prompt"]
    if not (len(p0) > 0 and isinstance(p0[0], dict)
            and {"role", "content"}.issubset(p0[0].keys())):
        sys.exit(f"[build_ifeval] ERROR: row 0 prompt malformed: {p0!r}")

    # Rewrite data_source
    print(f"[build_ifeval] rewriting data_source: "
          f"{df['data_source'].value_counts().to_dict()} -> '{NEW_DATA_SOURCE}'")
    df = df.copy()
    df["data_source"] = NEW_DATA_SOURCE

    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
        print(f"[build_ifeval]   --limit applied: keep first {len(df):,} rows")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    df.to_parquet(args.dst, index=False)
    print(f"[build_ifeval] wrote {args.dst}  ({len(df):,} rows)")

    # Final sanity readout
    chk = pd.read_parquet(args.dst)
    print()
    print("--- output sanity ---")
    print(f"  shape: {chk.shape}")
    print(f"  data_source: {chk['data_source'].value_counts().to_dict()}")
    print(f"  ability: {chk['ability'].value_counts().to_dict()}")
    rm = chk.iloc[0]["reward_model"]
    gt = rm["ground_truth"] if isinstance(rm, dict) else "<malformed>"
    print(f"  row 0 reward_model.style: {rm.get('style') if isinstance(rm, dict) else '?'}")
    print(f"  row 0 reward_model.ground_truth (head): {str(gt)[:140]!r}")
    p = chk.iloc[0]["prompt"]
    print(f"  row 0 prompt[0].content (head): {str(p[0]['content'])[:140]!r}")


if __name__ == "__main__":
    main()
