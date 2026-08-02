#!/usr/bin/env python
"""F1 helper — precompute top-1/2 token IDs at AIME position-1 for both
primary and intervention models. Output JSON with structure:
    {
      "model_tag": "B_q3" | "I_q3",
      "data_path": "...",
      "tokens": {
        prompt_id: {top1_id, top1_text, top2_id, top2_text}
      }
    }

Used to (a) feed single_token_C1/C2 with the intervention's top-1 as a
*string prefix* in vLLM, and (b) feed random_top2 with primary's own top-2.

Why precompute as strings: vLLM doesn't expose a per-step LogitsProcessor
forcing the FIRST token; the standard hack is to append the desired token's
text to the prompt. For Qwen3-family vocab this round-trips correctly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get("PROJECT_ROOT", ".")

CKPT_BASE = f"{REPO}/checkpoints/verl_exp/DAPO_sh_repro"
MODEL_PATHS = {
    "B_q3": f"{REPO}/models/Qwen3-8B-Base",
    "M_q3": f"{CKPT_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface",
    "I_q3": f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface",
    "B_q25m": f"{REPO}/models/Qwen2.5-Math-7B",
    "M_q25m": f"{CKPT_BASE}/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface",
    "I_q25m": f"{CKPT_BASE}/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface",
}


def aime_prompt_id(s: str) -> str:
    if s.startswith("user\n"):
        s = s[len("user\n"):]
    if s.endswith("\nassistant\n"):
        s = s[:-len("\nassistant\n")]
    elif s.endswith("assistant\n"):
        s = s[:-len("assistant\n")]
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()[:16]


def build_prompt(row) -> str:
    parts = []
    for m in row["prompt"]:
        parts.append(f"{m['role']}\n{m['content']}")
    parts.append("assistant\n")
    return "\n".join(parts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True, choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--data_path",
                    default=f"{REPO}/data/aime24/test.parquet")
    ap.add_argument("--out_dir",
                    default=f"{REPO}/analysis/data/f1_intervention")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--max_ctx", type=int, default=4096)
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.gpu_id}"

    df = pd.read_parquet(args.data_path)
    print(f"[f1_topk] {args.model_tag}: {len(df)} prompts on {device}")

    model_path = MODEL_PATHS[args.model_tag]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=False, low_cpu_mem_usage=True)
    model.eval()

    out = {"model_tag": args.model_tag, "data_path": args.data_path, "tokens": {}}
    for idx, row in df.iterrows():
        prompt = build_prompt(row)
        pid = aime_prompt_id(prompt)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] > args.max_ctx:
            ids = ids[:, :args.max_ctx]
        logits = model(ids).logits[0, -1, :]
        tk = torch.topk(logits, 4)
        ids_top4 = [int(x) for x in tk.indices.tolist()]
        texts_top4 = [tok.decode([i], skip_special_tokens=False) for i in ids_top4]
        out["tokens"][pid] = {
            "prompt_idx": int(idx),
            "top1_id": ids_top4[0], "top1_text": texts_top4[0],
            "top2_id": ids_top4[1], "top2_text": texts_top4[1],
            "top3_id": ids_top4[2], "top3_text": texts_top4[2],
            "top4_id": ids_top4[3], "top4_text": texts_top4[3],
        }
        if (idx + 1) % 5 == 0 or idx == len(df) - 1:
            print(f"[f1_topk]   {idx+1}/{len(df)}  pid={pid}  top1={texts_top4[0]!r}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"first_token_topk__{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[f1_topk] wrote {out_path}")


if __name__ == "__main__":
    main()
