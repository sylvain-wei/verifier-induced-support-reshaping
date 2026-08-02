#!/usr/bin/env python
"""Evaluate one HF checkpoint on IFBench (test.parquet, 300 prompts) with n=32 rollouts.

Sampling matches b2r1 training-time val:
  T=1.0, top_p=0.7, top_k=-1, max_resp=8192, max_prompt=2048, n=32, seed=1234.

Prompt format matches b2r1 training-time val (no chat template, since the model
is a Base model — verl trained with: 'user\\n<content>\\nassistant\\n').

Scoring uses the bundled verl `if_multi_constraints.compute_score`, which delegates
to `ifeval/verifier.verify_ifeval`. The verifier auto-falls-back to the IFBench
INSTRUCTION_DICT for OOD instruction ids.

Outputs (both required):
  --out_rollout_jsonl : 300*32 = 9 600 rows, schema aligned with b2r1 training jsonl:
        {"input","output","score","step","reward","acc","pred",
         "instruction_id_list","key"}
  --out_json          : per-step aggregate (pass@1, best@32, mean_len, ...).
"""
import argparse
import json
import os
import sys

# vLLM requires 'spawn' when CUDA may already be initialized in the parent.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd

# Use the bundled verl copy because it carries the IFBench instruction registry.
sys.path.insert(0, os.path.join(os.environ.get("PROJECT_ROOT", "."), "verl"))
from verl.utils.reward_score.if_multi_constraints import compute_score  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402


def build_prompts(df):
    """Same as eval_one.py: 'user\\n<content>\\nassistant\\n' (no chat template).

    Matches b2r1 training-time val concatenation observed in the existing rollout
    jsonls under DAPO_sh_repro/.
    """
    prompts = []
    for _, row in df.iterrows():
        msgs = row["prompt"]
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
    ap.add_argument("--out_rollout_jsonl", required=True)
    ap.add_argument("--step", type=int, required=True,
                    help="Training step label for the rollout jsonl (also used as the 'step' field).")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--response_length", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"[ifbench] loaded {len(df)} prompts from {args.data_path}")

    prompts = build_prompts(df)

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

    print(f"[ifbench] generating: n={args.n} T={args.temperature} top_p={args.top_p} resp_len={args.response_length}")
    outputs = llm.generate(prompts, sp)

    # Score every (prompt, rollout) and emit aligned jsonl.
    os.makedirs(os.path.dirname(args.out_rollout_jsonl) or ".", exist_ok=True)
    fout = open(args.out_rollout_jsonl, "w")

    per_prompt = []
    for i, out in enumerate(outputs):
        row = df.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        data_source = row["data_source"]
        inst_ids = list(row["instruction_id_list"])
        key = row["extra_info"]["key"] if isinstance(row["extra_info"], dict) and "key" in row["extra_info"] else int(row["key"])
        rollouts = []
        for cand in out.outputs:
            text = cand.text
            res = compute_score(data_source=data_source, solution_str=text, ground_truth=gt)
            score = float(res["score"])
            rollouts.append({"score": score, "len": len(text)})
            fout.write(json.dumps({
                "input": prompts[i],
                "output": text,
                "score": score,
                "step": int(args.step),
                "reward": score,
                "acc": bool(score >= 1.0),
                "pred": "[N/A]",
                "instruction_id_list": inst_ids,
                "key": key,
            }, ensure_ascii=False) + "\n")
        scores = [r["score"] for r in rollouts]
        n = len(rollouts)
        # IF-style metrics: pass@1 = mean(score), best@n = max(score) (partial-credit aware).
        pass_at_1 = float(np.mean(scores))
        best_at_n = float(max(scores))
        # Strict variant: any rollout fully satisfies all constraints (score >= 1.0).
        strict_pass_at_1 = float(np.mean([1.0 if s >= 1.0 else 0.0 for s in scores]))
        strict_best_at_n = 1.0 if any(s >= 1.0 for s in scores) else 0.0
        per_prompt.append({
            "prompt_idx": i,
            "key": key,
            "n": n,
            "pass@1": pass_at_1,
            "best@n": best_at_n,
            "strict_pass@1": strict_pass_at_1,
            "strict_best@n": strict_best_at_n,
            "mean_len": float(np.mean([r["len"] for r in rollouts])),
        })
    fout.close()
    print(f"[ifbench] wrote {args.out_rollout_jsonl}")

    mean_pass1 = float(np.mean([p["pass@1"] for p in per_prompt]))
    mean_bestn = float(np.mean([p["best@n"] for p in per_prompt]))
    mean_strict_pass1 = float(np.mean([p["strict_pass@1"] for p in per_prompt]))
    mean_strict_bestn = float(np.mean([p["strict_best@n"] for p in per_prompt]))
    mean_len = float(np.mean([p["mean_len"] for p in per_prompt]))

    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "step": int(args.step),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "response_length": args.response_length,
        "metrics": {
            # Stable aggregate keys for downstream table generation.
            "pass@1": mean_pass1,
            "best@32": mean_bestn,
            "strict_pass@1": mean_strict_pass1,
            "strict_best@32": mean_strict_bestn,
            "mean_output_char_len": mean_len,
            "invalid_frac": 0.0,  # IF benchmarks have no invalid extraction concept
        },
        "per_prompt": per_prompt,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ifbench] wrote {args.out_json}")
    print(f"[ifbench] pass@1={mean_pass1:.4f}  best@32={mean_bestn:.4f}  "
          f"strict_pass@1={mean_strict_pass1:.4f}  strict_best@32={mean_strict_bestn:.4f}  "
          f"mean_len={mean_len:.0f}")


if __name__ == "__main__":
    main()
