#!/usr/bin/env python
"""Phase 1 — Sample n=32 responses per prompt with vLLM.

Why this script
---------------
For the OPD sanity check we need rollouts from two policies on the same 200
stratified IF-train prompts:
  base       — Qwen3-8B-Base (the OPD student start)
  teacher_if — b2r1 step 100 (IF-RLVR final ckpt = the OPD teacher)

The base rollouts feed the main S_ratio comparison; the teacher rollouts feed
counter-experiment (a) — what does the teacher's own generation pool look like
under our DeepSeek shortcut/contentful judge?

Sampling matches the b2r1 training-time validation observed in existing rollout
JSONLs (T=1.0, top_p=0.7, top_k=-1, max_resp=8192, n=32, seed=1234), so the
distributions we measure here are commensurable with the analysis already in
the report.

Outputs
-------
rollout/opd_sanity/{model_tag}/responses.jsonl
    one row per rollout: {prompt_id, sample_id, family, input, output,
                          score, num_constraints, num_satisfied, acc,
                          response_chars, instruction_id_list, ground_truth}

Usage
-----
  # base
  python sample_responses.py --model_tag base \\
      --model_path "${PROJECT_ROOT}/models/Qwen3-8B-Base"
  # IF teacher
  python sample_responses.py --model_tag teacher_if \\
      --model_path "${PROJECT_ROOT}/checkpoints/verl_exp/DAPO_sh_repro/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"
"""
import argparse
import json
import os
import sys

# vLLM requires 'spawn' when CUDA may already be initialized.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd  # noqa: E402

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, os.path.join(ROOT, "verl"))
from verl.utils.reward_score.ifeval.verifier import verify_ifeval  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402


def build_prompts(df: pd.DataFrame):
    """`'user\\n<content>\\nassistant\\n'` — same template as b2r1 training-time val.

    The IF-train messages array has exactly one user turn (we verified earlier).
    """
    prompts = []
    for _, row in df.iterrows():
        user = row["prompt_text"]
        prompts.append(f"user\n{user}\nassistant\n")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True,
                    help="Subdirectory name under rollout/opd_sanity/. e.g. base, teacher_if")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompts_parquet",
                    default=f"{ROOT}/analysis/data/opd_sanity/selected_prompts.parquet")
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/rollout/opd_sanity")
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

    df = pd.read_parquet(args.prompts_parquet)
    print(f"[sample] loaded {len(df)} prompts from {args.prompts_parquet}")

    prompts = build_prompts(df)

    out_dir = os.path.join(args.out_dir, args.model_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "responses.jsonl")

    print(f"[sample] loading model {args.model_path}")
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

    print(f"[sample] generating: n={args.n} T={args.temperature} top_p={args.top_p}")
    outputs = llm.generate(prompts, sp)

    fout = open(out_jsonl, "w")
    n_pass = 0
    n_total = 0
    for i, out in enumerate(outputs):
        row = df.iloc[i]
        gt = row["ground_truth"]
        inst_ids = list(row["instruction_id_list"])
        for sid, cand in enumerate(out.outputs):
            text = cand.text
            res = verify_ifeval(text, gt)
            score = float(res.score)
            fout.write(json.dumps({
                "prompt_id": row["prompt_id"],
                "sample_id": sid,
                "family": row["family"],
                "input": prompts[i],
                "output": text,
                "score": score,
                "num_constraints": int(res.num_constraints),
                "num_satisfied": int(res.num_satisfied),
                "acc": bool(score >= 1.0),
                "response_chars": len(text),
                "instruction_id_list": inst_ids,
                "ground_truth": gt,
            }, ensure_ascii=False) + "\n")
            n_total += 1
            if score >= 1.0:
                n_pass += 1
    fout.close()
    print(f"[sample] wrote {out_jsonl}")
    print(f"[sample] n_total={n_total}  n_strict_pass={n_pass}  strict_rate={n_pass/n_total:.4f}")


if __name__ == "__main__":
    main()
