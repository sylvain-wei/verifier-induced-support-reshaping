#!/usr/bin/env python
"""h2 — Per-(model, AIME prompt) first-token routing-projection extractor.

For each (model, prompt) pair, runs ONE teacher-forced HF forward pass on the
prompt envelope, takes logits at the position right after the assistant header
(the first response token), softmax-converts, and computes:
    logp_DRI = log Σ_{t∈T_DRI} p(t)
    logp_DAI = log Σ_{t∈T_DAI} p(t)

Output: analysis/data/h2_first_token_proj.parquet with columns:
    [model_tag, prompt_idx, prompt_id, logp_DRI, logp_DAI, top1_id, top1_text,
     top1_logp, total_logp_routing, top1_class]

Routing token IDs (validated on Qwen3-8B-Base tokenizer):
    T_DRI = {1249 "To", 8304 "Step", 10061 "Let", 1654 "We"}
    T_DAI = {16141 "Answer", 785 "The", 25 ":"}

Usage:
    # Single GPU on a single model:
    CUDA_VISIBLE_DEVICES=0 python h2_first_token_logits.py --model_tag B_q3
    # Multi-model batch (8 GPUs, one model per GPU):
    bash analysis/scripts/h2_first_token_logits.sh
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("PROJECT_ROOT", ".")

# Validated against Qwen3-8B-Base tokenizer (tokenizer == Qwen2.5-Math by inspection).
# Extended DRI vocab matches the prefix list in classify_opening_modes.py
# (deliberative openers including "Alright", "Okay", problem-restatement openers).
T_DRI = {
    1249:  "To",
    8304:  "Step",
    10061: "Let",
    1654:  "We",
    71486: "Alright",
    32313: "Okay",
    3925:  "OK",
    5979:  "Right",
    4416:  "So",
    7039:  "Now",
    22043: "Given",
    37175: "Consider",
    35338: "Define",
    9112:  "Note",
    5338:  "First",
    9885:  "Find",
    46254: "Compute",
}
# DAI tokens (direct-answer openers).
T_DAI = {16141: "Answer", 785: "The", 25: ":"}

MODEL_PATHS = {
    "B_q3": f"{ROOT}/models/Qwen3-8B-Base",
    "M_q3": (f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
             "b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface"),
    "I_q3": (f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
             "b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"),
    "B_q25m": f"{ROOT}/models/Qwen2.5-Math-7B",
    "M_q25m": (f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
               "b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface"),
    "I_q25m": (f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
               "b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface"),
}


def build_prompt_text(prompt_field, tokenizer):
    """Apply chat template to convert chat-list to bare-template prompt text.

    Use the bare RL template `user\\n{content}\\nassistant\\n` directly,
    bypassing model-specific ChatML, to keep h2 on-distribution with eval rollouts.
    """
    if isinstance(prompt_field, (list, tuple, np.ndarray)):
        chat = list(prompt_field)
    elif isinstance(prompt_field, str):
        chat = [{"role": "user", "content": prompt_field}]
    else:
        raise TypeError(f"unsupported prompt field type: {type(prompt_field)}")
    user_content = next(m["content"] for m in chat if m["role"] == "user")
    return f"user\n{user_content}\nassistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True, help="Tag in MODEL_PATHS")
    ap.add_argument("--model_path", default=None,
                    help="Override path; otherwise look up MODEL_PATHS[model_tag]")
    ap.add_argument("--data", default=f"{ROOT}/data/aime24/test.parquet")
    ap.add_argument("--out_dir", default=f"{ROOT}/analysis/data/h2_first_token_proj")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model_tag}.parquet")
    if os.path.exists(out_path):
        print(f"[h2:{args.model_tag}] already exists at {out_path}, skipping")
        return

    model_path = args.model_path or MODEL_PATHS[args.model_tag]
    print(f"[h2:{args.model_tag}] model_path={model_path}")

    df = pd.read_parquet(args.data)
    print(f"[h2:{args.model_tag}] {len(df)} prompts loaded from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    print(f"[h2:{args.model_tag}] tokenizer loaded; vocab={len(tokenizer)}")

    # Confirm routing token ids decode as expected (sanity check across q3/q25m).
    for tid, lbl in {**T_DRI, **T_DAI}.items():
        decoded = tokenizer.decode([tid])
        if decoded.strip() != lbl.strip() and decoded != lbl:
            print(f"  WARN routing token mismatch: id={tid} expected={lbl!r} "
                  f"decoded={decoded!r}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"[h2:{args.model_tag}] model loaded in {time.time()-t0:.1f}s")

    rows = []
    dri_ids = list(T_DRI.keys())
    dai_ids = list(T_DAI.keys())

    for i, row in df.iterrows():
        prompt_text = build_prompt_text(row["prompt"], tokenizer)
        ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = ids["input_ids"].to("cuda")

        with torch.no_grad():
            out = model(input_ids=input_ids)
        # logits shape: (1, L, V); the last position predicts the *next* (= first response) token.
        last_logits = out.logits[0, -1, :].float()
        logp = torch.log_softmax(last_logits, dim=-1)

        # Sum p(token) over routing classes -> log(sum) via logsumexp on log-probs.
        logp_dri = float(torch.logsumexp(logp[dri_ids], dim=-1))
        logp_dai = float(torch.logsumexp(logp[dai_ids], dim=-1))

        top1_id = int(logp.argmax())
        top1_logp = float(logp[top1_id])
        top1_text = tokenizer.decode([top1_id])
        if top1_id in T_DRI:
            top1_class = "DRI"
        elif top1_id in T_DAI:
            top1_class = "DAI"
        else:
            top1_class = "Other"

        prompt_id = row.get("id", i)

        rows.append(dict(
            model_tag=args.model_tag,
            prompt_idx=int(i),
            prompt_id=str(prompt_id),
            logp_DRI=logp_dri,
            logp_DAI=logp_dai,
            top1_id=top1_id,
            top1_text=top1_text,
            top1_logp=top1_logp,
            top1_class=top1_class,
        ))

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[h2:{args.model_tag}] wrote {out_path} ({len(out_df)} rows)")
    # Quick on-screen summary.
    counts = out_df.top1_class.value_counts().to_dict()
    print(f"[h2:{args.model_tag}] top1_class counts: {counts}")
    print(f"[h2:{args.model_tag}] mean logp_DRI={out_df.logp_DRI.mean():.3f}, "
          f"mean logp_DAI={out_df.logp_DAI.mean():.3f}")


if __name__ == "__main__":
    main()
