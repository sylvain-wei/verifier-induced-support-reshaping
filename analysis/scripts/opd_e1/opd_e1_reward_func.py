#!/usr/bin/env python
"""Custom reward function for E1 OPD training (multi-domain dispatcher).

Why this file
-------------
E1 trains on IF-train (95k rows) and validates on AIME24 / AIME25 / MATH-500-128.
verl's built-in dispatcher (verl.utils.reward_score.__init__.default_compute_score)
routes ``aime*`` and ``math_dapo`` to math_dapo.compute_score, but it has no IF
verifier branch. We register THIS file via ``custom_reward_function.path`` so
verl uses it for every batch and skips the built-in dispatcher entirely.

Routing
-------
- ``data_source == "ifeval_train"`` (or ``"ifeval"``)  → IFEval verifier
- ``data_source.startswith("aime")``                   → math_dapo grader
- ``data_source in {"math_dapo", "math"}``             → math_dapo grader
- anything else                                         → math_dapo grader as a
                                                           forgiving fallback,
                                                           warned once per source
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# --- Math grader: eager import from verl fork ---
from verl.utils.reward_score import math_dapo as _verl_math_dapo  # noqa: E402

_math_dapo_compute_score = _verl_math_dapo.compute_score

# --- IF verifier ---
from verl.utils.reward_score.ifeval.verifier import verify_ifeval as _verify_ifeval
_warned_unknown_ds: set[str] = set()


def _ifeval_score(solution_str: str, ground_truth: Any) -> dict:
    """Score one IF response. Returns {score, acc, num_constraints, num_satisfied}."""
    verify_ifeval = _verify_ifeval
    res = verify_ifeval(solution_str, ground_truth)
    score = float(res.score)
    return {
        "score": score,
        "acc": bool(score >= 1.0 - 1e-9),
        "num_constraints": int(res.num_constraints),
        "num_satisfied": int(res.num_satisfied),
    }


def _math_score(solution_str: str, ground_truth: Any) -> dict:
    """Score one math response via math_dapo. Returns {score, acc, pred}."""
    res = _math_dapo_compute_score(solution_str, str(ground_truth))
    if isinstance(res, dict):
        return {
            "score": float(res.get("score", 0.0)),
            "acc": bool(res.get("acc", False)),
            "pred": res.get("pred", ""),
        }
    return {"score": float(res), "acc": bool(float(res) > 0.0), "pred": ""}


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
) -> dict:
    """verl-compatible custom reward function. Returns dict with score + acc."""
    try:
        ds = (data_source or "").strip()
        if ds in ("ifeval_train", "ifeval"):
            return _ifeval_score(solution_str, ground_truth)
        if ds.startswith("aime") or ds in ("math_dapo", "math"):
            return _math_score(solution_str, ground_truth)
        if ds not in _warned_unknown_ds:
            _warned_unknown_ds.add(ds)
            print(f"[opd_e1_reward_func] WARNING: unknown data_source={ds!r}, "
                  f"routing to math_dapo grader as fallback")
        return _math_score(solution_str, ground_truth)
    except Exception as e:
        print(f"[opd_e1_reward_func] ERROR scoring data_source={data_source!r}: {e}")
        traceback.print_exc()
        return {"score": 0.0, "acc": False, "_error": repr(e)[:200]}


# --- Smoke test (run directly: python opd_e1_reward_func.py) ---
if __name__ == "__main__":
    print("=== smoke test ===")

    # 1) IFEval row (from data/ifeval/train.parquet row 0). Two constraints:
    # detectable_format:sentence_hyphens (sentences joined by hyphens, no spaces)
    # last_word:last_word_answer (last word == 'brief')
    if_gt = ("[{'instruction_id': ['detectable_format:sentence_hyphens', "
             "'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]")
    if_resp_good = "this-is-a-sentence-brief"
    if_resp_bad = "This is a normal sentence with spaces and the wrong last word"
    print("IF (good)   :", reward_func("ifeval_train", if_resp_good, if_gt))
    print("IF (bad)    :", reward_func("ifeval_train", if_resp_bad, if_gt))

    # 2) Math (math_dapo)
    print("MATH (good) :", reward_func("math_dapo",
                                        "Step 1... Answer: $\\boxed{42}$",
                                        "42"))
    print("MATH (bad)  :", reward_func("math_dapo",
                                        "Answer: $\\boxed{0}$",
                                        "42"))

    # 3) AIME (data_source.startswith("aime"))
    print("AIME (good) :", reward_func("aime24",
                                        "...Answer: $\\boxed{540}$",
                                        "540"))

    # 4) Unknown — fallback path
    print("UNK         :", reward_func("totally_unknown",
                                        "Answer: $\\boxed{0}$",
                                        "0"))

    # 5) Verify import paths
    import verl
    import verl.utils.reward_score.math_dapo as md
    print("verl     :", verl.__file__)
    print("math_dapo:", md.__file__)
