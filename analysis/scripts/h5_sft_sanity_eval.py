#!/usr/bin/env python
"""h5 — SFT-only sanity eval: small AIME + IFEval forward to confirm SFT didn't
break the model and to measure pre-RL baseline.

For a single SFT'd ckpt: generate n=8 rollouts on AIME-24 (30 prompts) and on
IFEval (sample 50 prompts), compute best@8, pass@1, opening-mode rates.

Output: analysis/data/h5_sft_sanity/<tag>.parquet (per-rollout) and
        analysis/tables/h5_sft_sanity_summary.csv (one row per ckpt).

Usage:
    python analysis/scripts/h5_sft_sanity_eval.py \
        --tag h1_dri50_S1 \
        --ckpt "${CHECKPOINT_DIR}"
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Short tag for output files")
    ap.add_argument("--ckpt", required=True, help="Path to ckpt with config.json + safetensors")
    ap.add_argument("--aime", default=f"{ROOT}/data/aime24/test.parquet")
    ap.add_argument("--ifeval", default=f"{ROOT}/data/ifeval/test.parquet")
    ap.add_argument("--n_per_prompt", type=int, default=8,
                    help="Rollouts per prompt (best@n)")
    ap.add_argument("--ifeval_n_prompts", type=int, default=50,
                    help="Subsample IFEval to N prompts (saves time vs all 540)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--out_dir", default=f"{ROOT}/analysis/data/h5_sft_sanity")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_parquet = os.path.join(args.out_dir, f"{args.tag}.parquet")

    if os.path.exists(out_parquet):
        print(f"[h5:{args.tag}] {out_parquet} exists, skipping")
        return

    from vllm import LLM, SamplingParams

    print(f"[h5:{args.tag}] loading vLLM model from {args.ckpt}")
    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=1,
        max_model_len=args.max_tokens + 2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
        dtype="bfloat16",
    )
    print(f"[h5:{args.tag}] model loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(
        n=args.n_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    rows = []

    # ---- AIME ----
    aime = pd.read_parquet(args.aime)
    aime_prompts = []
    for i, row in aime.iterrows():
        chat = list(row["prompt"])
        user_content = next(m["content"] for m in chat if m["role"] == "user")
        prompt_text = f"user\n{user_content}\nassistant\n"
        aime_prompts.append(prompt_text)
    print(f"[h5:{args.tag}] AIME: {len(aime_prompts)} prompts, n={args.n_per_prompt}")
    t0 = time.time()
    aime_outs = llm.generate(aime_prompts, sp)
    print(f"[h5:{args.tag}] AIME generated in {time.time()-t0:.1f}s")

    for i, out in enumerate(aime_outs):
        prompt_id = str(aime.iloc[i].get("id", i))
        gold = aime.iloc[i]["reward_model"]["ground_truth"] \
            if isinstance(aime.iloc[i]["reward_model"], dict) else None
        for s, sample_out in enumerate(out.outputs):
            rows.append(dict(
                tag=args.tag,
                benchmark="aime",
                prompt_idx=int(i),
                prompt_id=prompt_id,
                sample_idx=int(s),
                response=sample_out.text,
                gold=gold,
                n_resp_tokens=len(sample_out.token_ids),
            ))

    # ---- IFEval (subsample) ----
    ifeval = pd.read_parquet(args.ifeval)
    if len(ifeval) > args.ifeval_n_prompts:
        ifeval = ifeval.sample(n=args.ifeval_n_prompts, random_state=0).reset_index(drop=True)
    ifeval_prompts = []
    for i, row in ifeval.iterrows():
        if "prompt" in row and isinstance(row["prompt"], (list, tuple)):
            chat = list(row["prompt"])
            user_content = next(m["content"] for m in chat if m["role"] == "user")
        else:
            user_content = row.get("prompt_text", row.get("prompt", ""))
            if isinstance(user_content, (list, tuple)):
                user_content = next(m["content"] for m in user_content if m["role"] == "user")
        prompt_text = f"user\n{user_content}\nassistant\n"
        ifeval_prompts.append(prompt_text)
    print(f"[h5:{args.tag}] IFEval: {len(ifeval_prompts)} prompts (sub of {len(pd.read_parquet(args.ifeval))})")
    t0 = time.time()
    if_outs = llm.generate(ifeval_prompts, sp)
    print(f"[h5:{args.tag}] IFEval generated in {time.time()-t0:.1f}s")
    for i, out in enumerate(if_outs):
        prompt_id = str(i)
        for s, sample_out in enumerate(out.outputs):
            rows.append(dict(
                tag=args.tag,
                benchmark="ifeval",
                prompt_idx=int(i),
                prompt_id=prompt_id,
                sample_idx=int(s),
                response=sample_out.text,
                gold=None,
                n_resp_tokens=len(sample_out.token_ids),
            ))

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"[h5:{args.tag}] wrote {out_parquet} ({len(df)} rows)")


if __name__ == "__main__":
    main()
