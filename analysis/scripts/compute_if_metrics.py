#!/usr/bin/env python
"""Compute IF metrics across all (run, step) and per-constraint pass rates.

For each (run, benchmark in {ifeval, ifbench}, step), produces:

  - per-rollout score (already in jsonl) — pass@1 / best@32 / strict_pass@1 / strict_best@32
  - per-constraint pass/fail by re-running `verify_ifeval._check_one` on each
    rollout's instruction_id_list

Outputs:
  analysis/tables/if_metrics_per_step.csv         (run, step, benchmark, mean_score, pass@1, best@32, strict_pass@1, strict_best@32, mean_len)
  analysis/tables/if_metrics_per_prompt.csv       (run, step, benchmark, prompt_id, key, n, pass_count_strict, mean_score, max_score)
  analysis/tables/constraint_category_perf.csv    (run, step, benchmark, category, num_evals, pass_rate)

Reuses load_rollouts.load_run + verifier `_check_one` for per-constraint pass.

Usage:
  python analysis/scripts/compute_if_metrics.py            # all runs, all steps, both benches
  python analysis/scripts/compute_if_metrics.py --runs b1r1_Qwen3-8B-Base_math7.5k_local_H20 \
       --benchmarks ifeval --steps 0,220
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, os.path.join(ROOT, "verl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_rollouts import RUNS, load_run, list_run_steps  # noqa: E402

OUT_TABLES = f"{ROOT}/analysis/tables"


# -----------------------------------------------------------------------------
# Per-constraint check (caches).
# -----------------------------------------------------------------------------
_check_one_fn = None


def _get_check_one():
    global _check_one_fn
    if _check_one_fn is None:
        from verl.utils.reward_score.ifeval.verifier import _check_one  # noqa
        _check_one_fn = _check_one
    return _check_one_fn


def _category_of(inst_id: str) -> str:
    """Return the colon-prefix category, e.g. 'length_constraints' from
    'length_constraints:number_words'. Falls back to the full id."""
    return inst_id.split(":", 1)[0] if ":" in inst_id else inst_id


def _parse_kwargs(ground_truth) -> tuple[list, list]:
    """Return (instruction_ids, kwargs_list) from the ground_truth string.

    ground_truth format (matches ifeval/test.parquet):
      "[{'instruction_id': [...], 'kwargs': [...]}]"
    """
    obj = ground_truth
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        return [], []
    return list(obj.get("instruction_id", [])), list(obj.get("kwargs", []))


def per_constraint_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-constraint pass info to a DataFrame.

    For ifbench rows, ground_truth was not loaded; reconstruct kwargs from the
    test.parquet via key. For ifeval rows, ground_truth came from the join.
    """
    check_one = _get_check_one()

    # Build IFBench key->(inst_ids, kwargs) lookup.
    ifb_test_path = f"{ROOT}/data/ifbench/test.parquet"
    ifb_lookup = {}
    if os.path.exists(ifb_test_path):
        ifb_df = pd.read_parquet(ifb_test_path)
        for _, r in ifb_df.iterrows():
            ifb_lookup[int(r["key"])] = {
                "instruction_id": list(r["instruction_id_list"]),
                "kwargs": [{kk: vv for kk, vv in (k or {}).items() if vv is not None}
                           for k in list(r["kwargs"])],
            }

    out_rows = []
    for _, r in df.iterrows():
        bench = r["benchmark"]
        inst_ids: list = []
        kwargs_list: list = []
        if bench == "ifeval":
            gt = r.get("ground_truth")
            if gt is None:
                continue
            inst_ids, kwargs_list = _parse_kwargs(gt)
        elif bench == "ifbench":
            key = r.get("key")
            try:
                key = int(key)
            except (ValueError, TypeError):
                continue
            meta = ifb_lookup.get(key)
            if not meta:
                continue
            inst_ids = meta["instruction_id"]
            kwargs_list = meta["kwargs"]
        else:
            continue

        # pad kwargs
        if len(kwargs_list) < len(inst_ids):
            kwargs_list = list(kwargs_list) + [None] * (len(inst_ids) - len(kwargs_list))

        for inst_id, kwargs in zip(inst_ids, kwargs_list):
            try:
                ok = check_one(inst_id, kwargs or {}, r["response"])
            except Exception:
                ok = False
            out_rows.append({
                "run_name": r["run_name"],
                "checkpoint_step": r["checkpoint_step"],
                "benchmark": bench,
                "prompt_id": r["prompt_id"],
                "sample_id": r["sample_id"],
                "instruction_id": inst_id,
                "category": _category_of(inst_id),
                "passed": int(bool(ok)),
            })
    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Aggregations.
# -----------------------------------------------------------------------------
def aggregate_per_step(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, step, bench), g in df.groupby(["run_name", "checkpoint_step", "benchmark"]):
        # group by prompt -> n rollouts each
        per_prompt = g.groupby("prompt_id")["score"]
        pass_at_1 = float(per_prompt.mean().mean())
        best_at_n = float(per_prompt.max().mean())
        strict_pass_at_1 = float(per_prompt.apply(lambda s: (s >= 1.0).mean()).mean())
        strict_best_at_n = float(per_prompt.apply(lambda s: (s >= 1.0).any()).mean())
        mean_len = float(g["response_length_chars"].mean())
        rows.append({
            "run_name": run, "checkpoint_step": step, "benchmark": bench,
            "n_prompts": g["prompt_id"].nunique(),
            "n_rollouts": len(g),
            "pass@1": pass_at_1,
            "best@n": best_at_n,
            "strict_pass@1": strict_pass_at_1,
            "strict_best@n": strict_best_at_n,
            "mean_len_chars": mean_len,
        })
    return pd.DataFrame(rows).sort_values(["run_name", "benchmark", "checkpoint_step"])


def aggregate_per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, step, bench, pid), g in df.groupby(["run_name", "checkpoint_step", "benchmark", "prompt_id"]):
        scores = g["score"].values
        rows.append({
            "run_name": run, "checkpoint_step": step, "benchmark": bench,
            "prompt_id": pid, "key": g["key"].iloc[0],
            "n": len(g),
            "pass_count_strict": int((scores >= 1.0).sum()),
            "mean_score": float(scores.mean()),
            "max_score": float(scores.max()),
        })
    return pd.DataFrame(rows)


def aggregate_constraint_category(constraint_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, step, bench, cat), g in constraint_df.groupby(
        ["run_name", "checkpoint_step", "benchmark", "category"]
    ):
        rows.append({
            "run_name": run, "checkpoint_step": step, "benchmark": bench,
            "category": cat, "num_evals": len(g),
            "pass_rate": float(g["passed"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["run_name", "benchmark", "category", "checkpoint_step"])


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                    help="comma-separated run names; default = all known runs")
    ap.add_argument("--benchmarks", default="ifeval,ifbench",
                    help="comma-separated subset of {ifeval, ifbench}")
    ap.add_argument("--steps", default=None,
                    help="comma-separated step list; default = all available")
    ap.add_argument("--skip_per_constraint", action="store_true",
                    help="Skip constraint-category breakdown (faster).")
    ap.add_argument("--out_step", default=f"{OUT_TABLES}/if_metrics_per_step.csv")
    ap.add_argument("--out_prompt", default=f"{OUT_TABLES}/if_metrics_per_prompt.csv")
    ap.add_argument("--out_cat", default=f"{OUT_TABLES}/constraint_category_perf.csv")
    args = ap.parse_args()

    os.makedirs(OUT_TABLES, exist_ok=True)
    requested_steps = ([int(s) for s in args.steps.split(",")]
                       if args.steps else None)
    runs = (args.runs.split(",") if args.runs
            else list(RUNS.keys()))
    benches = args.benchmarks.split(",")

    all_rollouts = []
    for run in runs:
        # IFEval rollouts live inside val_rollout (mixed with AIME).
        if "ifeval" in benches:
            steps = requested_steps or list_run_steps(run, "val_rollout")
            if steps:
                df = load_run(run, kind="val_rollout", steps=steps)
                if len(df):
                    df = df[df["benchmark"] == "ifeval"]
                    print(f"[{run}] ifeval: {len(df)} rows across {df['checkpoint_step'].nunique()} steps")
                    all_rollouts.append(df)
        if "ifbench" in benches:
            steps_b = requested_steps or list_run_steps(run, "val_rollout_ifbench")
            if steps_b:
                df = load_run(run, kind="val_rollout_ifbench", steps=steps_b)
                if len(df):
                    print(f"[{run}] ifbench: {len(df)} rows across {df['checkpoint_step'].nunique()} steps")
                    all_rollouts.append(df)

    if not all_rollouts:
        print("[error] no rollouts loaded.")
        return
    big = pd.concat(all_rollouts, ignore_index=True)
    print(f"\n[total] {len(big)} rollouts loaded")

    print("\n[aggregate per-step]")
    step_df = aggregate_per_step(big)
    step_df.to_csv(args.out_step, index=False)
    print(f"  wrote {args.out_step} ({len(step_df)} rows)")

    print("\n[aggregate per-prompt]")
    prompt_df = aggregate_per_prompt(big)
    prompt_df.to_csv(args.out_prompt, index=False)
    print(f"  wrote {args.out_prompt} ({len(prompt_df)} rows)")

    if not args.skip_per_constraint:
        print("\n[per-constraint reverification]")
        cdf = per_constraint_eval(big)
        if len(cdf):
            cat = aggregate_constraint_category(cdf)
            cat.to_csv(args.out_cat, index=False)
            print(f"  wrote {args.out_cat} ({len(cat)} rows)")
        else:
            print("  [warn] empty constraint dataframe — skipped category file")


if __name__ == "__main__":
    main()
