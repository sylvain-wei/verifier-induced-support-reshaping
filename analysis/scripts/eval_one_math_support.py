#!/usr/bin/env python
"""Math-7.5k support probe inference.

For one HF checkpoint, run n=16 rollouts on the 128-prompt fixed probe sampled
from math 7.5k (analysis/data/math_support_probe_128.parquet) and score each
rollout with the math_dapo verifier.

Sampling matches DAPO training (n=16, T=1.0, top_p=0.7, top_k=-1,
max_resp=8192, max_prompt=2048). Prompt format matches verl base-model training:
"user\\n<content>\\nassistant\\n" with no chat template.

Outputs:
  --out_rollout_jsonl : 128*16 = 2 048 rows; schema:
        {"prompt_id","input","output","score","step","reward","acc","pred",
         "ground_truth"}
  --out_json          : per-step aggregate with group-level stats
        (all_wrong_rate, all_correct_rate, mixed_rate, mean_group_pass,
         pass@1, best@n, mean_output_char_len, ...) — these are now genuine
        unfiltered measurements, not artifacts of DAPO's dynamic-sampling
        filter.

Score is binary {0, 1} per rollout because math_dapo.compute_score returns
acc=bool. We treat score>=1.0 as "correct" so the same downstream pipeline
works for IF too.
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


def build_prompts(df):
    """user\\n<content>\\nassistant\\n (matches verl base-model training)."""
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
    ap.add_argument("--data_path", default=os.path.join(PROJECT_ROOT, "analysis", "data", "math_support_probe_128.parquet"))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_rollout_jsonl", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run_name", required=True,
                    help="Run identifier, written into out_json.")
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
    print(f"[probe] loaded {len(df)} prompts from {args.data_path}")

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

    print(f"[probe] generating: n={args.n} T={args.temperature} top_p={args.top_p} resp_len={args.response_length}")
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
            res = compute_score(solution_str=text, ground_truth=gt)
            # math_dapo returns {"score": 1.0/0.0, "acc": True/False, "pred": ...}
            score = float(res.get("score", 0.0))
            pred = res.get("pred", "[N/A]")
            acc = bool(res.get("acc", score >= 1.0))
            rollouts.append({"score": score, "len": len(text), "acc": acc})
            fout.write(json.dumps({
                "prompt_id": prompt_id,
                "input": prompts[i],
                "output": text,
                "score": score,
                "step": int(args.step),
                "reward": score,
                "acc": acc,
                "pred": pred,
                "ground_truth": str(gt),
            }, ensure_ascii=False) + "\n")
        # NB: math_dapo.compute_score returns score = +1/-1 (not +1/0); use
        # the boolean `acc` field to define the binary correctness used by
        # group-level diagnostics. Raw `score` is preserved in the jsonl.
        accs = np.array([1.0 if r["acc"] else 0.0 for r in rollouts])
        n = len(rollouts)
        n_correct = int(accs.sum())
        all_wrong = bool(n_correct == 0)
        all_correct = bool(n_correct == n)
        mixed = bool(0 < n_correct < n)
        per_prompt.append({
            "prompt_idx": i,
            "prompt_id": prompt_id,
            "n": n,
            "n_correct": n_correct,
            "all_wrong": all_wrong,
            "all_correct": all_correct,
            "mixed": mixed,
            "pass@1": float(accs.mean()),         # binary acc, in [0, 1]
            "best@n": float(accs.max()),          # 0 or 1
            "mean_len": float(np.mean([r["len"] for r in rollouts])),
        })
    fout.close()
    print(f"[probe] wrote {args.out_rollout_jsonl}")

    pp = pd.DataFrame(per_prompt)
    all_wrong_rate = float(pp["all_wrong"].mean())
    all_correct_rate = float(pp["all_correct"].mean())
    mixed_rate = float(pp["mixed"].mean())
    mean_pass1 = float(pp["pass@1"].mean())
    mean_bestn = float(pp["best@n"].mean())
    mean_len = float(pp["mean_len"].mean())

    summary = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "step": int(args.step),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "response_length": args.response_length,
        "metrics": {
            "all_wrong_rate": all_wrong_rate,
            "all_correct_rate": all_correct_rate,
            "mixed_rate": mixed_rate,
            "pass@1": mean_pass1,
            "best@n": mean_bestn,
            "mean_output_char_len": mean_len,
        },
        "per_prompt": per_prompt,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe] wrote {args.out_json}")
    print(f"[probe] all_wrong={all_wrong_rate:.3f} all_correct={all_correct_rate:.3f} "
          f"mixed={mixed_rate:.3f} pass@1={mean_pass1:.3f} best@n={mean_bestn:.3f} "
          f"mean_len={mean_len:.0f}")


if __name__ == "__main__":
    main()
