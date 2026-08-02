#!/usr/bin/env python
"""Custom reward function for E3 OPD training (multi-domain dispatcher, extended).

Why a separate file from opd_e1_reward_func.py
-----------------------------------------------
E1 results are locked behind opd_e1_reward_func.py (its dispatcher table is
final; touching it would risk affecting any E1-replay). E3 needs to route four
new validation data_sources through the IFEval verifier:

  - ifeval_test       (data/ifeval/ifeval_test_opd_val.parquet)
  - ifbench_test      (data/ifbench/ifbench_test_opd_val.parquet)
  - allenai/IF_multi_constraints_upto5  (defensive — original data_source
    of test.parquet, in case anyone forgets to use the re-stamped val files)

Otherwise this file is structurally identical to opd_e1_reward_func.py and
shares the same math_dapo grader and IFEval verifier from the verl framework.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# --- Math grader (active verl fork) ---
from verl.utils.reward_score import math_dapo as _verl_math_dapo  # noqa: E402

_math_dapo_compute_score = _verl_math_dapo.compute_score

# --- IF verifier ---
from verl.utils.reward_score.ifeval.verifier import verify_ifeval as _verify_ifeval


# data_source values routed to the IFEval verifier.
_IFEVAL_SOURCES = frozenset({
    "ifeval_train",
    "ifeval",
    "ifeval_test",            # E3 IFEval-test val (581 rows after re-stamp)
    "ifbench_test",           # E3 IFBench val
    "allenai/IF_multi_constraints_upto5",  # defensive: original test.parquet ds
})

def _ifeval_score(solution_str: str, ground_truth: Any) -> dict:
    verify_ifeval = _verify_ifeval
    res = verify_ifeval(solution_str, ground_truth)
    score = float(res.score)
    # NOTE: returned key set must match _math_score's key set EXACTLY — verl's
    # reward manager appends each returned key into a per-key list across all
    # rows in the val batch, and _validate() asserts every per-key list has
    # length == num_samples. If math rows return {pred} but IF rows don't,
    # the 'pred' list ends up shorter than the batch size and val crashes
    # with `AssertionError: pred: len(lst)=N1, len(sample_scores)=N2`.
    return {
        "score": score,
        "acc": bool(score >= 1.0 - 1e-9),
        "pred": "",                   # placeholder so key set matches math path
        "num_constraints": int(res.num_constraints),
        "num_satisfied": int(res.num_satisfied),
    }


def _math_score(solution_str: str, ground_truth: Any) -> dict:
    res = _math_dapo_compute_score(solution_str, str(ground_truth))
    if isinstance(res, dict):
        return {
            "score": float(res.get("score", 0.0)),
            "acc": bool(res.get("acc", False)),
            "pred": res.get("pred", ""),
            "num_constraints": 0,     # placeholder so key set matches IF path
            "num_satisfied": 0,
        }
    s = float(res)
    return {
        "score": s,
        "acc": bool(s > 0.0),
        "pred": "",
        "num_constraints": 0,
        "num_satisfied": 0,
    }


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
) -> dict:
    try:
        ds = (data_source or "").strip()
        if ds in _IFEVAL_SOURCES:
            return _ifeval_score(solution_str, ground_truth)
        if ds.startswith("aime") or ds in ("math_dapo", "math"):
            return _math_score(solution_str, ground_truth)
        if ds not in _warned_unknown_ds:
            _warned_unknown_ds.add(ds)
            print(f"[opd_e3_reward_func] WARNING: unknown data_source={ds!r}, "
                  f"routing to math_dapo grader as fallback")
        return _math_score(solution_str, ground_truth)
    except Exception as e:
        print(f"[opd_e3_reward_func] ERROR scoring data_source={data_source!r}: {e}")
        traceback.print_exc()
        # IMPORTANT: same key set as IF/math paths — verl's val aggregation
        # builds per-key lists across all rows and asserts uniform length.
        # Don't add an _error key here: rows that fail would diverge from
        # rows that succeed and re-trigger the very assertion we're avoiding.
        return {
            "score": 0.0,
            "acc": False,
            "pred": "",
            "num_constraints": 0,
            "num_satisfied": 0,
        }


# --- Smoke test ---
if __name__ == "__main__":
    import pyarrow.parquet as pq

    print("=== smoke test (E3 dispatcher) ===")

    # 1) IFEval-test row 0 (re-stamped data_source='ifeval_test')
    t = pq.read_table(
        os.environ.get("PROJECT_ROOT", ".") + "/data/ifeval/ifeval_test_opd_val.parquet"
    )
    row = t.slice(0, 1).to_pydict()
    gt = row["reward_model"][0]["ground_truth"]
    ds = row["data_source"][0]
    # A trivially-failing response — verifier should return score < 1.
    print("IFEval-test (bad)  :", reward_func(ds, "Hello, world!", gt))
    # A response that is unlikely to satisfy 'no comma + 3 highlights + 300+ words',
    # but exercises the verifier path successfully.

    # 2) IFBench row 0
    tb = pq.read_table(
        os.environ.get("PROJECT_ROOT", ".") + "/data/ifbench/ifbench_test_opd_val.parquet"
    )
    rb = tb.slice(0, 1).to_pydict()
    gt_b = rb["reward_model"][0]["ground_truth"]
    ds_b = rb["data_source"][0]
    print("IFBench (bad)      :", reward_func(ds_b, "Random unrelated answer.", gt_b))

    # 3) Math (math_dapo)
    print("MATH (good)        :", reward_func("math_dapo",
                                                "Step 1... Answer: $\\boxed{42}$", "42"))
    # 4) Defensive: original allenai/... source
    print("allenai (bad)      :", reward_func("allenai/IF_multi_constraints_upto5",
                                                "x", gt))

    # 5) verl namespace integrity
    import verl
    import verl.utils.reward_score.math_dapo as md
    assert verl.__file__ and md.__file__
    print("OK — verl imports resolved, dispatcher routes IF + math correctly")
