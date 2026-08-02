#!/usr/bin/env python
"""B1/B2 — Forced-prefix decoding eval on AIME (vLLM).

Drop-in for the standard eval_one.py with one extra flag: --prefix STRING.
The prefix is concatenated to each prompt right after the closing
"assistant\\n" — i.e. the model continues from that prefix.

Usage examples
--------------
B1 (forced DRI on b2r1-100):
  python b1_forced_prefix_eval.py \\
      --model_path .../b2r1.../global_step_100/actor/huggingface \\
      --data_path  "${PROJECT_ROOT}/data/aime24/test.parquet" \\
      --out_json   eval_results/disentangle/b1_b2r1_100_dri_n8.json \\
      --out_rollout_jsonl eval_results/disentangle/b1_b2r1_100_dri_n8.jsonl \\
      --n 8 --prefix $'Let me solve this step by step.\\n\\n' \\
      --tp 4

B2 (forced DAI on b0):
  ... --model_path .../models/Qwen3-8B-Base ... --prefix "Answer: " --n 8 ...

For computing best@k for k <= n, we report a curve (k=1,2,4,8,16,32 capped at n).
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
    """Apply the standard 'user\\n<content>\\nassistant\\n' template, then
    append the forced prefix to make the model continue from it."""
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


def best_at_k(accs: list, k: int) -> float:
    """Probability that any of k samples is correct (uniform-without-replacement
    over the n samples). Closed form via combinations: 1 - C(n-c,k)/C(n,k)
    where c = number of correct. We use the unbiased estimator the field uses."""
    n = len(accs)
    c = sum(1 for a in accs if a)
    if k > n:
        return float(c > 0)
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    # 1 - C(n-c, k) / C(n, k); compute in log-space to avoid overflow.
    from math import lgamma
    def lg_comb(a, b):
        if b < 0 or b > a:
            return float("-inf")
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    return float(1.0 - np.exp(lg_comb(n - c, k) - lg_comb(n, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_rollout_jsonl", default=None)
    ap.add_argument("--prefix", default="",
                    help="String forced after `assistant\\n`. Empty = free decoding.")
    ap.add_argument("--label", default=None,
                    help="Run label written into out_json. Defaults to model_path.")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--response_length", type=int, default=12288)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"[bX] loaded {len(df)} prompts from {args.data_path}")
    prompts = build_prompts(df, prefix=args.prefix)
    if args.prefix:
        print(f"[bX] forced prefix: {args.prefix!r}")
        # Sanity check: print head of first prompt.
        head = prompts[0][-200:]
        print(f"[bX] first prompt tail: ...{head!r}")

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
    print(f"[bX] generating: n={args.n} T={args.temperature} top_p={args.top_p} resp={args.response_length}")
    outputs = llm.generate(prompts, sp)

    per_prompt = []
    rollout_rows = []
    for i, out in enumerate(outputs):
        gt = str(df.iloc[i]["reward_model"]["ground_truth"])
        rollouts = []
        for cand in out.outputs:
            text = cand.text
            # When --prefix is given, the model continues from it. The scorer
            # extracts the answer from the FULL response; to match that, we
            # prepend the forced prefix to the model's continuation before
            # scoring. (The verifier looks for "Answer:" / "\\boxed{}" anywhere
            # in the string, so prepending matters when the model continues
            # the sentence.)
            full_text = (args.prefix + text) if args.prefix else text
            res = compute_score(full_text, gt)
            rollouts.append({
                "score": res["score"],
                "acc": bool(res["acc"]),
                "pred": res["pred"],
                "len": len(text),
            })
            if args.out_rollout_jsonl:
                rollout_rows.append({
                    "prompt_idx": i,
                    "gt": gt,
                    "prefix": args.prefix,
                    "output_continuation": text,
                    "scored_text_head": full_text[:200],
                    "score": res["score"],
                    "acc": bool(res["acc"]),
                    "pred": res["pred"],
                })
        accs = [r["acc"] for r in rollouts]
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= len(rollouts)]
        best_curve = {f"best@{k}": best_at_k(accs, k) for k in ks}
        per_prompt.append({
            "prompt_idx": i,
            "gt": gt,
            "n": len(rollouts),
            "pass@1": sum(accs) / len(accs),
            "best@n": 1.0 if any(accs) else 0.0,
            "mean_len": float(np.mean([r["len"] for r in rollouts])),
            "invalid_frac": sum(1 for r in rollouts if r["pred"] == "[INVALID]") / len(rollouts),
            **best_curve,
        })

    pp = pd.DataFrame(per_prompt)
    metrics = {
        "pass@1":   float(pp["pass@1"].mean()),
        "best@n":   float(pp["best@n"].mean()),
        "mean_len": float(pp["mean_len"].mean()),
    }
    for col in pp.columns:
        if col.startswith("best@"):
            metrics[col] = float(pp[col].mean())
    summary = {
        "label":          args.label or args.model_path,
        "model_path":     args.model_path,
        "data_path":      args.data_path,
        "prefix":         args.prefix,
        "n":              args.n,
        "temperature":    args.temperature,
        "top_p":          args.top_p,
        "response_length": args.response_length,
        "metrics":        metrics,
        "per_prompt":     per_prompt,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bX] wrote {args.out_json}")
    print(f"[bX] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                 if k.startswith("best") or k == "pass@1"))

    if args.out_rollout_jsonl:
        os.makedirs(os.path.dirname(args.out_rollout_jsonl) or ".", exist_ok=True)
        with open(args.out_rollout_jsonl, "w") as f:
            for r in rollout_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[bX] wrote {args.out_rollout_jsonl}")


if __name__ == "__main__":
    main()
