#!/usr/bin/env python
"""C2 — Forced-DRI math probe (128-prompt math-7.5k probe under DRI prefix).

Same as analysis/scripts/eval_one_math_support.py but injects a forced prefix
after the closing `assistant\\n`, and prepends that prefix to the response
before passing to the scorer (so 'Answer:' continuation gets matched correctly).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(PROJECT_ROOT, "verl"))
from verl.utils.reward_score.math_dapo import compute_score  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402


def build_prompts(df, prefix: str = ""):
    prompts = []
    for _, row in df.iterrows():
        msgs = row["prompt"]
        parts = []
        for m in msgs:
            parts.append(f"{m['role']}\n{m['content']}")
        parts.append("assistant\n")
        s = "\n".join(parts)
        if prefix:
            s = s + prefix
        prompts.append(s)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", default=os.path.join(PROJECT_ROOT, "analysis", "data", "math_support_probe_128.parquet"))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_rollout_jsonl", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--n", type=int, default=16)
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
    print(f"[c2] loaded {len(df)} prompts; prefix={args.prefix!r}")
    prompts = build_prompts(df, prefix=args.prefix)
    if args.prefix:
        print(f"[c2] first prompt tail: ...{prompts[0][-180:]!r}")

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
    outputs = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(args.out_rollout_jsonl) or ".", exist_ok=True)
    fout = open(args.out_rollout_jsonl, "w")
    per_prompt = []
    for i, out in enumerate(outputs):
        row = df.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        prompt_id = row["prompt_id"]
        rollouts = []
        for cand in out.outputs:
            text = cand.text
            full_text = (args.prefix + text) if args.prefix else text
            res = compute_score(solution_str=full_text, ground_truth=gt)
            score = float(res.get("score", 0.0))
            pred = res.get("pred", "[N/A]")
            acc = bool(res.get("acc", score >= 1.0))
            rollouts.append({"score": score, "len": len(text), "acc": acc})
            fout.write(json.dumps({
                "prompt_id": prompt_id,
                "input": prompts[i],
                "prefix": args.prefix,
                "output_continuation": text,
                "scored_text_head": full_text[:200],
                "score": score,
                "step": int(args.step),
                "reward": score,
                "acc": acc,
                "pred": pred,
                "ground_truth": str(gt),
            }, ensure_ascii=False) + "\n")
        accs = np.array([1.0 if r["acc"] else 0.0 for r in rollouts])
        n = len(rollouts)
        n_correct = int(accs.sum())
        per_prompt.append({
            "prompt_idx": i,
            "prompt_id": prompt_id,
            "n": n,
            "n_correct": n_correct,
            "all_wrong":  bool(n_correct == 0),
            "all_correct": bool(n_correct == n),
            "mixed":       bool(0 < n_correct < n),
            "pass@1":   float(accs.mean()),
            "best@n":   float(accs.max()),
            "mean_len": float(np.mean([r["len"] for r in rollouts])),
        })
    fout.close()
    print(f"[c2] wrote {args.out_rollout_jsonl}")

    pp = pd.DataFrame(per_prompt)
    metrics = {
        "all_wrong_rate":   float(pp["all_wrong"].mean()),
        "all_correct_rate": float(pp["all_correct"].mean()),
        "mixed_rate":       float(pp["mixed"].mean()),
        "pass@1":           float(pp["pass@1"].mean()),
        "best@n":           float(pp["best@n"].mean()),
        "mean_output_char_len": float(pp["mean_len"].mean()),
    }
    summary = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "step": int(args.step),
        "prefix": args.prefix,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "response_length": args.response_length,
        "metrics": metrics,
        "per_prompt": per_prompt,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[c2] wrote {args.out_json}")
    print(f"[c2] all_wrong={metrics['all_wrong_rate']:.3f}  "
          f"mixed={metrics['mixed_rate']:.3f}  "
          f"all_correct={metrics['all_correct_rate']:.3f}  "
          f"pass@1={metrics['pass@1']:.3f}  best@n={metrics['best@n']:.3f}")


if __name__ == "__main__":
    main()
