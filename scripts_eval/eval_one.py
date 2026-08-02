#!/usr/bin/env python
"""
Evaluate one HuggingFace model checkpoint on AIME24 (math_dapo) with n=32 rollouts,
response_length=12288, temperature=1.0, top_p=0.7, top_k=-1 (val_kwargs from training script).

Writes a JSON summary to --out_json with per-prompt pass@1, best@32 and dataset-level metrics.
"""
import argparse
import json
import os
import re
import sys

# Set multiprocessing start method BEFORE importing torch or anything that touches CUDA.
# vLLM requires 'spawn' when the parent process already has CUDA initialized (e.g. through
# `import verl...` which pulls in torch+cuda).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd

# verl's DAPO extractor (same one used during training val)
sys.path.insert(0, os.environ.get("PROJECT_ROOT", ".") + "/verl_new")
from verl.utils.reward_score.math_dapo import compute_score  # noqa: E402

from vllm import LLM, SamplingParams


def build_prompts(df, tokenizer_path):
    """Turn the 'prompt' array-of-dict column into string prompts the same way verl does at val.

    Training prompt format is: 'user\n<content>\nassistant\n' (no chat-template applied for Base),
    matching what we saw in rollout jsonl files.
    """
    prompts = []
    for _, row in df.iterrows():
        msgs = row["prompt"]
        # msgs is a list-like of dicts with 'role' and 'content'
        parts = []
        for m in msgs:
            parts.append(f"{m['role']}\n{m['content']}")
        parts.append("assistant\n")
        prompts.append("\n".join(parts))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--response_length", type=int, default=12288)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_rollout_jsonl", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"[eval] loaded {len(df)} prompts from {args.data_path}")

    prompts = build_prompts(df, args.model_path)

    # vLLM engine
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_prompt_length + args.response_length,
        enforce_eager=False,
        enable_chunked_prefill=True,
        max_num_seqs=128,
        trust_remote_code=False,
        disable_log_stats=True,
        seed=args.seed,
    )

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.response_length,
        seed=args.seed,
    )

    print(f"[eval] generating: n={args.n} T={args.temperature} top_p={args.top_p} resp_len={args.response_length}")
    outputs = llm.generate(prompts, sp)

    # Score every (prompt, rollout)
    per_prompt = []
    all_rollouts = []
    for i, out in enumerate(outputs):
        gt = str(df.iloc[i]["reward_model"]["ground_truth"])
        rollouts = []
        for cand in out.outputs:
            text = cand.text
            res = compute_score(text, gt)
            rollouts.append({
                "score": res["score"],
                "acc": bool(res["acc"]),
                "pred": res["pred"],
                "len": len(text),
            })
            if args.out_rollout_jsonl:
                all_rollouts.append({
                    "prompt_idx": i,
                    "gt": gt,
                    "output": text,
                    "score": res["score"],
                    "acc": bool(res["acc"]),
                    "pred": res["pred"],
                })
        accs = [r["acc"] for r in rollouts]
        pass_at_1 = sum(1 for a in accs if a) / len(accs)
        best_at_32 = 1.0 if any(accs) else 0.0
        per_prompt.append({
            "prompt_idx": i,
            "gt": gt,
            "n": len(rollouts),
            "pass@1": pass_at_1,
            "best@n": best_at_32,
            "mean_len": float(np.mean([r["len"] for r in rollouts])),
            "invalid_frac": sum(1 for r in rollouts if r["pred"] == "[INVALID]") / len(rollouts),
            "truncated_frac": sum(1 for r in rollouts if r["len"] >= (args.response_length * 3)) / len(rollouts),
        })

    # aggregate
    mean_pass1 = float(np.mean([p["pass@1"] for p in per_prompt]))
    mean_bestn = float(np.mean([p["best@n"] for p in per_prompt]))
    mean_len = float(np.mean([p["mean_len"] for p in per_prompt]))
    mean_invalid = float(np.mean([p["invalid_frac"] for p in per_prompt]))

    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "response_length": args.response_length,
        "metrics": {
            "pass@1": mean_pass1,
            "best@32": mean_bestn,
            "mean_output_char_len": mean_len,
            "invalid_frac": mean_invalid,
        },
        "per_prompt": per_prompt,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {args.out_json}")
    print(f"[eval] pass@1={mean_pass1:.4f}  best@32={mean_bestn:.4f}  "
          f"mean_len={mean_len:.0f}  invalid={mean_invalid:.3f}")

    if args.out_rollout_jsonl:
        with open(args.out_rollout_jsonl, "w") as f:
            for r in all_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
