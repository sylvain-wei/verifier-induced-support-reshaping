#!/usr/bin/env python
"""Unified loader for all rollout JSONLs across b1r1 / b2r1 / b4r1 runs.

Reads a single JSONL or a whole directory of JSONLs and returns a pandas
DataFrame with a normalized schema (one row per (prompt, sample)). Auto-detects
the rollout type:

  - val_rollout AIME row : input contains "Solve the following math problem step by step"
  - val_rollout IFEval row : everything else in val_rollout/{run}/*.jsonl
  - val_rollout_ifbench/{run}/{step}.jsonl : already pure IFBench (instruction_id_list/key present)
  - train_rollout/{run}/{step}.jsonl       : pure on-policy training rollouts
                                             (group_size=16, no `reward` field)

For IFEval rows, the `instruction_id_list` is *not* in the rollout jsonl, so it
is recovered by joining the IFEval prompt text against
data/ifeval/test.parquet. The first user-message content is extracted from the
`input` field by stripping the "user\\n" prefix and "\\nassistant\\n" suffix.

Usage
-----
  # CLI: dump consolidated parquet for one run
  python load_rollouts.py --run b2r1_Qwen3-8B-Base_IFTrain_local_H20 \
      --steps 0,100 --max_prompts 5 --out /tmp/sanity.parquet

  # Library
  from load_rollouts import load_run
  df = load_run("b1r1_Qwen3-8B-Base_math7.5k_local_H20",
                training_type="math_rlvr",
                kind="val_rollout",          # AIME+IFEval mixed
                steps=[0, 220])
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from typing import Iterable, Optional

import pandas as pd


ROOT = os.environ.get("PROJECT_ROOT", ".")
AIME_MARKER = "Solve the following math problem step by step"

VAL_ROLLOUT_DIR        = f"{ROOT}/rollout/val_rollout/DAPO_sh_repro"
VAL_ROLLOUT_IFBENCH_DIR= f"{ROOT}/rollout/val_rollout_ifbench"
TRAIN_ROLLOUT_DIR      = f"{ROOT}/rollout/train_rollout/DAPO_sh_repro"

IFEVAL_TEST_PARQUET = f"{ROOT}/data/ifeval/test.parquet"

# Run -> (training_type, base_model_name, max_step) registry. base_model_name
# is informational only (used to set the model_name column of the consolidated
# DataFrame).
RUNS = {
    "b1r1_Qwen3-8B-Base_math7.5k_local_H20":     ("math_rlvr", "Qwen3-8B-Base"),
    "b1r1_Qwen2.5-math-7B_math7.5k_local_H20":   ("math_rlvr", "Qwen2.5-Math-7B"),
    "b2r1_Qwen3-8B-Base_IFTrain_local_H20":      ("if_rlvr",   "Qwen3-8B-Base"),
    "b2r1_Qwen2.5-math-7B_IFTrain_local_H20":    ("if_rlvr",   "Qwen2.5-Math-7B"),
    "b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20": ("if_then_math", "Qwen3-8B-Base"),
}


# -----------------------------------------------------------------------------
# IFEval prompt -> instruction_id_list join cache.
# -----------------------------------------------------------------------------
_ifeval_prompt_to_inst = None


def _load_ifeval_prompt_map():
    """Return dict[prompt_text -> dict(key, instruction_id_list, ground_truth)]."""
    global _ifeval_prompt_to_inst
    if _ifeval_prompt_to_inst is not None:
        return _ifeval_prompt_to_inst
    df = pd.read_parquet(IFEVAL_TEST_PARQUET)
    out = {}
    for _, row in df.iterrows():
        # row['prompt'] is array of {role, content}; take the first user msg
        msgs = list(row["prompt"])
        # b1r1/b2r1 val rollouts strip role+content together: "user\n<content>\nassistant\n"
        text = msgs[0]["content"]
        gt = row["reward_model"]["ground_truth"]
        out[text] = {
            "key": int(row["key"]),
            "instruction_id_list": list(row["instruction_id_list"]),
            "ground_truth": gt,
        }
    _ifeval_prompt_to_inst = out
    return out


def _extract_user_content_from_input(s: str) -> str:
    """val_rollout `input` is 'user\\n<content>\\nassistant\\n'. Recover content.

    For multi-message prompts the joiner is '\\n' between role+content blocks,
    but the IFEval val rollouts in this repo always use a single user message,
    so we only need to peel the first 'user\\n' and the trailing '\\nassistant\\n'.
    """
    s2 = s
    if s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    # peel one trailing '\nassistant\n' (the actual format in jsonls)
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def _hash_prompt(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# JSONL -> DataFrame.
# -----------------------------------------------------------------------------
def _classify_row(o: dict, kind: str) -> str:
    """Return one of {'aime','ifeval','ifbench','math_train'}."""
    if kind == "val_rollout_ifbench":
        return "ifbench"
    if kind == "train_rollout":
        return "math_train"
    # val_rollout: AIME marker decides
    if AIME_MARKER in o["input"]:
        return "aime"
    return "ifeval"


def load_jsonl(
    path: str,
    run_name: str,
    training_type: str,
    kind: str,
    step: int,
    model_name: str,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """Load one rollout jsonl into the unified schema."""
    rows = []
    if_map = None
    seen_prompts_per_bench: dict = {}  # only used when max_prompts is set
    sample_idx_per_input: dict = {}    # hash(input) -> running counter

    with open(path) as f:
        for ln in f:
            o = json.loads(ln)
            bench = _classify_row(o, kind)

            input_text = o["input"]
            ph = _hash_prompt(input_text)

            if max_prompts is not None:
                seen = seen_prompts_per_bench.setdefault(bench, set())
                if len(seen) >= max_prompts and ph not in seen:
                    continue
                seen.add(ph)

            sample_id = sample_idx_per_input.get(ph, 0)
            sample_idx_per_input[ph] = sample_id + 1

            inst_ids = None
            key = None
            ground_truth = None
            prompt_text = input_text

            if bench == "ifeval":
                if if_map is None:
                    if_map = _load_ifeval_prompt_map()
                content = _extract_user_content_from_input(input_text)
                meta = if_map.get(content)
                if meta is not None:
                    inst_ids = meta["instruction_id_list"]
                    key = meta["key"]
                    ground_truth = meta["ground_truth"]
                prompt_text = content
            elif bench == "ifbench":
                inst_ids = list(o.get("instruction_id_list", []))
                key = o.get("key")
                prompt_text = _extract_user_content_from_input(input_text)
            elif bench == "aime":
                prompt_text = _extract_user_content_from_input(input_text)
            else:  # math_train
                prompt_text = _extract_user_content_from_input(input_text)

            response = o["output"]
            score = float(o.get("score", 0.0))
            reward = float(o["reward"]) if "reward" in o else score
            # AIME / math has acc=bool; IFEval/IFBench has acc=float (==score) → coerce
            acc_raw = o.get("acc", None)
            if acc_raw is None:
                acc = bool(score >= 1.0) if bench in ("ifeval", "ifbench") else False
            elif isinstance(acc_raw, bool):
                acc = acc_raw
            else:
                acc = bool(float(acc_raw) >= 1.0)

            rows.append({
                "run_name": run_name,
                "model_name": model_name,
                "training_type": training_type,
                "checkpoint_step": step,
                "benchmark": bench,
                "prompt_id": (str(key) if key is not None else ph),
                "sample_id": sample_id,
                "prompt": prompt_text,
                "response": response,
                "reward": reward,
                "acc": acc,
                "score": score,
                "extracted_answer": o.get("pred"),
                "instruction_id_list": inst_ids,
                "key": key,
                "ground_truth": ground_truth,
                "response_length_chars": len(response),
                "response_length_ws_tokens": len(response.split()),
            })
    return pd.DataFrame(rows)


def load_run(
    run_name: str,
    training_type: Optional[str] = None,
    kind: str = "val_rollout",   # 'val_rollout' / 'val_rollout_ifbench' / 'train_rollout'
    steps: Optional[Iterable[int]] = None,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate all jsonls for one run into a single DataFrame."""
    if run_name not in RUNS:
        raise ValueError(f"Unknown run {run_name}; known: {list(RUNS)}")
    tt, model_name = RUNS[run_name]
    training_type = training_type or tt

    if kind == "val_rollout":
        d = f"{VAL_ROLLOUT_DIR}/{run_name}"
    elif kind == "val_rollout_ifbench":
        d = f"{VAL_ROLLOUT_IFBENCH_DIR}/{run_name}"
    elif kind == "train_rollout":
        d = f"{TRAIN_ROLLOUT_DIR}/{run_name}"
    else:
        raise ValueError(f"Unknown kind {kind}")
    if not os.path.isdir(d):
        return pd.DataFrame()

    files = sorted(
        glob.glob(os.path.join(d, "*.jsonl")),
        key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)),
    )
    out = []
    target_steps = set(steps) if steps is not None else None
    for path in files:
        m = re.match(r"(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        if target_steps is not None and step not in target_steps:
            continue
        df = load_jsonl(path, run_name, training_type, kind, step, model_name, max_prompts)
        out.append(df)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def list_run_steps(run_name: str, kind: str) -> list[int]:
    if kind == "val_rollout":
        d = f"{VAL_ROLLOUT_DIR}/{run_name}"
    elif kind == "val_rollout_ifbench":
        d = f"{VAL_ROLLOUT_IFBENCH_DIR}/{run_name}"
    elif kind == "train_rollout":
        d = f"{TRAIN_ROLLOUT_DIR}/{run_name}"
    else:
        raise ValueError
    if not os.path.isdir(d):
        return []
    steps = []
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        m = re.match(r"(\d+)\.jsonl$", os.path.basename(p))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


# -----------------------------------------------------------------------------
# CLI / sanity-check entry point.
# -----------------------------------------------------------------------------
def _sanity():
    """Run plan-verification 1+2: row-count breakdown + IFEval inst-id join."""
    print("=== sanity 1: load b2r1 Qwen3-8B-Base step 0 + step 100, 5 prompts each ===")
    df = load_run(
        "b2r1_Qwen3-8B-Base_IFTrain_local_H20",
        kind="val_rollout",
        steps=[0, 100],
        max_prompts=5,
    )
    print(f"loaded {len(df)} rows")
    print(df.groupby(["checkpoint_step", "benchmark"]).size().unstack(fill_value=0))
    if (df["benchmark"] == "ifeval").any():
        ifr = df[df["benchmark"] == "ifeval"]
        hit = ifr["instruction_id_list"].apply(lambda x: x is not None).mean()
        print(f"IFEval inst_id_list join hit rate: {hit:.3f}  (rows={len(ifr)})")

    print("\n=== sanity 2: full step-0 row counts (no max_prompts) ===")
    df0 = load_run(
        "b2r1_Qwen3-8B-Base_IFTrain_local_H20",
        kind="val_rollout",
        steps=[0],
    )
    print(df0.groupby(["benchmark"]).size())

    print("\n=== sanity 3: train_rollout schema ===")
    dft = load_run(
        "b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20",
        kind="train_rollout",
        steps=[1],
    )
    print(f"loaded {len(dft)} rows; group_size sample: ", end="")
    if len(dft):
        gs = dft.groupby("prompt_id").size().head(5).tolist()
        print(gs)
        print(dft.head(2).to_dict(orient="records"))

    print("\n=== sanity 4: verify_ifeval round-trip (1 row) ===")
    sys.path.insert(0, os.path.join(ROOT, "verl"))
    from verl.utils.reward_score.ifeval.verifier import verify_ifeval
    if (df["benchmark"] == "ifeval").any():
        r = df[(df["benchmark"] == "ifeval")
               & df["instruction_id_list"].apply(lambda x: x is not None)
               & (df["ground_truth"].apply(lambda x: x is not None))].iloc[0]
        res = verify_ifeval(r["response"], r["ground_truth"])
        print(f"  inst_ids={r['instruction_id_list']}")
        print(f"  reported score={r['score']:.4f} ; verifier rerun: "
              f"score={res.score:.4f} num_constraints={res.num_constraints} "
              f"num_satisfied={res.num_satisfied}")
        # cross-check
        ok = abs(res.score - r["score"]) < 1e-3
        print(f"  match={'OK' if ok else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="Run name; if absent, --sanity is implied")
    ap.add_argument("--kind", default="val_rollout",
                    choices=["val_rollout", "val_rollout_ifbench", "train_rollout"])
    ap.add_argument("--steps", default=None,
                    help="comma-separated step list (e.g. 0,100); default = all")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="Output parquet path (defaults: print head only)")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    if args.sanity or args.run is None:
        _sanity()
        return

    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    df = load_run(args.run, kind=args.kind, steps=steps, max_prompts=args.max_prompts)
    print(f"loaded {len(df)} rows")
    print(df.groupby(["checkpoint_step", "benchmark"]).size().unstack(fill_value=0))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
