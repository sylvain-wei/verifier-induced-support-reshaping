#!/usr/bin/env python
"""Convert /data/ifbench/test.jsonl (300 rows) -> /data/ifbench/test.parquet
matching the schema of /data/ifeval/test.parquet so the same verl reward
function (`if_multi_constraints.compute_score` -> `ifeval/verifier.verify_ifeval`)
can score IFBench rollouts (the verifier auto-falls-back to the IFBench
INSTRUCTION_DICT for OOD instruction ids).

Schema produced (matching ifeval/test.parquet):
  key (int64)
  prompt              : list[{role, content}]
  instruction_id_list : list[str]
  kwargs              : list[dict]   (None/null-fields stripped)
  data_source         : "allenai/IF_multi_constraints_upto5"
  ability             : "instruction_following"
  reward_model        : {"ground_truth": "<repr-list-of-1-dict>", "style": "rule"}
  extra_info          : {"constraint": "", "constraint_type": "eval",
                         "dataset": "allenai/IFBench",
                         "key": "ifbench_<key>"}
"""
import os
import json

import pandas as pd


SRC = os.environ.get("PROJECT_ROOT", ".") + "/data/ifbench/test.jsonl"
DST = os.environ.get("PROJECT_ROOT", ".") + "/data/ifbench/test.parquet"


def strip_nulls(d):
    return {k: v for k, v in d.items() if v is not None}


def main():
    rows = []
    with open(SRC) as f:
        for ln in f:
            o = json.loads(ln)
            inst_ids = list(o["instruction_id_list"])
            kw_clean = [strip_nulls(k) for k in o["kwargs"]]
            # ground_truth uses repr() to match the ifeval format which is a
            # str(list[dict]) parsed by ast.literal_eval inside parse_ground_truth.
            gt_obj = [{"instruction_id": inst_ids, "kwargs": kw_clean}]
            rows.append({
                "key": int(o["key"]),
                "prompt": [{"role": "user", "content": o["prompt"]}],
                "instruction_id_list": inst_ids,
                "kwargs": kw_clean,
                "data_source": "allenai/IF_multi_constraints_upto5",
                "ability": "instruction_following",
                "reward_model": {
                    "ground_truth": repr(gt_obj),
                    "style": "rule",
                },
                "extra_info": {
                    "constraint": "",
                    "constraint_type": "eval",
                    "dataset": "allenai/IFBench",
                    "key": f"ifbench_{o['key']}",
                },
            })

    df = pd.DataFrame(rows)
    df.to_parquet(DST, engine="pyarrow", index=False)
    print(f"[ifbench] wrote {len(df)} rows -> {DST}")
    # quick self-check: round-trip ground_truth via ast.literal_eval
    import ast
    g = ast.literal_eval(df.iloc[0]["reward_model"]["ground_truth"])
    assert isinstance(g, list) and isinstance(g[0], dict) and "instruction_id" in g[0], g
    print(f"[ifbench] schema OK; sample gt[0] = {g[0]}")


if __name__ == "__main__":
    main()
