#!/usr/bin/env python3
"""Build OPD-compatible IF val parquets for E3.

Output:
  data/ifeval/ifeval_test_opd_val.parquet   (541 rows, data_source='ifeval_test')
  data/ifbench/ifbench_test_opd_val.parquet (300 rows, data_source='ifbench_test')

Why this script
---------------
The existing data/ifeval/test.parquet and data/ifbench/test.parquet have
data_source='allenai/IF_multi_constraints_upto5' and otherwise correct schema
(prompt, reward_model.ground_truth, etc.) for verl/OPD val. The OPD reward
dispatcher (analysis/scripts/opd_e1/opd_e1_reward_func.py) routes by
exact data_source string match — keeping the original 'allenai/...' string
would force every val score through the math grader (wrong) or require
unstable substring matching. Cleaner: copy + re-stamp data_source, keep
everything else identical. Then add the matching strings to the dispatcher.

Sanity: ground_truth[0] of IFEval-test is the same Python-stringified format
that the bundled IFEval verifier already parses.
"""
from __future__ import annotations
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("PROJECT_ROOT", ".")

SPECS = [
    {
        "src": f"{ROOT}/data/ifeval/test.parquet",
        "dst": f"{ROOT}/data/ifeval/ifeval_test_opd_val.parquet",
        "new_ds": "ifeval_test",
    },
    {
        "src": f"{ROOT}/data/ifbench/test.parquet",
        "dst": f"{ROOT}/data/ifbench/ifbench_test_opd_val.parquet",
        "new_ds": "ifbench_test",
    },
]


def restamp(src: str, dst: str, new_ds: str) -> None:
    if not os.path.isfile(src):
        raise SystemExit(f"missing src: {src}")
    t = pq.read_table(src)
    n = t.num_rows
    new_col = pa.array([new_ds] * n, type=t.schema.field("data_source").type)
    idx = t.schema.get_field_index("data_source")
    t2 = t.set_column(idx, "data_source", new_col)
    # Drop columns that have schema-incompatible struct shapes between IFEval
    # and IFBench (kwargs has wildly different inner fields). The IFEval
    # verifier (verify_ifeval) parses everything it needs from
    # reward_model.ground_truth (a Python-stringified spec), not from the
    # raw kwargs column — so dropping kwargs/instruction_id_list/key is
    # safe for scoring. Required to allow datasets.concatenate_datasets to
    # align with the AIME/MATH val parquets at trainer load time.
    drop = [c for c in ("kwargs", "instruction_id_list", "key") if c in t2.schema.names]
    if drop:
        t2 = t2.drop(drop)
        print(f"  dropped columns: {drop}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(t2, dst, compression="snappy")
    print(f"  wrote {n} rows → {dst}")


def main() -> None:
    for s in SPECS:
        print(f"[build] {os.path.basename(s['src'])} → {os.path.basename(s['dst'])} (data_source='{s['new_ds']}')")
        restamp(**s)
    # Verify
    print("\n[verify] reading back …")
    for s in SPECS:
        t = pq.read_table(s["dst"])
        ds_set = set(t["data_source"].to_pylist())
        print(f"  {os.path.basename(s['dst'])}: rows={t.num_rows} data_source uniq={ds_set}")
        assert ds_set == {s["new_ds"]}, f"data_source not properly stamped on {s['dst']}"
    print("\nOK")


if __name__ == "__main__":
    main()
