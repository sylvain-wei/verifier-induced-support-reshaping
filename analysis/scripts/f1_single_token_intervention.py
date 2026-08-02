#!/usr/bin/env python
"""F1 / Block C P0 — single-token causal intervention on AIME.

Plan §6 spec
------------
- C.1: primary = B_q3, intervention = I_q3, position-1 forced to intervention's
  top-1 token under the same prompt → primary's free decode for rest.
  Expect: DAI rate jumps 5% → ≥80%, AIME best@32 collapses toward I_q3 level.
- C.2: primary = I_q3, intervention = B_q3, symmetric setup.
  Expect: DRI rate recovers from ~5% → ≥30%, AIME best@32 partially restored.
- C.3 baselines (run on the SAME 30 AIME prompts × 32 samples each):
    - free          : no intervention, just primary's free decode.
    - random_top2   : at position 1 force primary's OWN top-2 token (control:
                      shows whether "any non-greedy first token" is enough).
    - forced_DRI    : prefix "Let me solve this step by step.\n\n" forced
                      string (reproduces B1 with cleaner orthogonal control).
    - forced_DAI    : prefix "Answer: " forced string.

Implementation
--------------
We use HuggingFace transformers (NOT vLLM) so the LogitsProcessor for
exact single-token forcing is straightforward:

    class ForceFirstTokenLogitsProcessor(LogitsProcessor):
        def __init__(self, target_id):
            self.target_id = target_id
            self.fired = False
        def __call__(self, input_ids, scores):
            if not self.fired:
                self.fired = True
                neg = torch.full_like(scores, -1e9)
                neg[..., self.target_id] = 0.0
                return neg
            return scores

For each prompt:
  1. Run intervention model forward on prompt once → top-1 token at position 1.
  2. Run primary model with the LogitsProcessor + multinomial sampling
     (n=32, T=1.0, top_p=0.7) → 32 free-decode continuations conditioned on
     the forced first token.
  3. Score each continuation with the AIME verifier (math_dapo).
  4. Classify opening mode (DRI/DAI/CSI/Other) on each response.

For the random_top2 control we instead force primary's OWN top-2 (skipping
its top-1 = greedy choice). That's a sanity check that the effect we see
isn't merely "any first-token diversity".

The free / forced_DRI / forced_DAI conditions don't need an intervention
model — they're produced by primary's vanilla generate (with optional
prepended forced_prefix string).

Output
------
analysis/data/f1_intervention/{condition}__shard{i}of{N}.parquet — schema:
    condition (str), primary_tag (str), intervention_tag (str|None),
    prompt_id (str), prompt_idx (int), sample_id (int),
    forced_token_id (int|None), forced_token_text (str|None),
    response (str), response_chars (int), n_resp_tokens (int),
    score (float), acc (bool), pred (str),
    mode (str)  -- from classify_opening_modes

Plus a per-prompt aggregate / per-condition summary file at
analysis/tables/f1_intervention_summary.csv.

Usage
-----
    # smoke / dry-run (1 prompt × 4 sample, fast)
    python f1_single_token_intervention.py --smoke

    # one (condition, shard) at a time, used by run_wave3_block_c.sh
    python f1_single_token_intervention.py \\
        --condition single_token_C1 \\
        --shard_id 0 --n_shards 8 \\
        --gpu_id 0
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

REPO = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, f"{REPO}/analysis/scripts")
from classify_opening_modes import classify_mode  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "verl"))
from verl.utils.reward_score.math_dapo import compute_score  # noqa: E402

OUT_DIR = f"{REPO}/analysis/data/f1_intervention"
AIME_PARQUET = f"{REPO}/data/aime24/test.parquet"

CKPT_BASE = f"{REPO}/checkpoints/verl_exp/DAPO_sh_repro"
MODEL_PATHS = {
    "B_q3": f"{REPO}/models/Qwen3-8B-Base",
    "M_q3": f"{CKPT_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface",
    "I_q3": f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface",
}


# Plan §6 condition table.
CONDITIONS = {
    # condition_name : (primary, intervention, force_prefix, force_token_strategy)
    "free":             ("B_q3", None,    "", None),
    "single_token_C1":  ("B_q3", "I_q3",  "", "intervention_top1"),
    "single_token_C2":  ("I_q3", "B_q3",  "", "intervention_top1"),
    "random_top2_B_q3": ("B_q3", None,    "", "primary_top2"),
    "random_top2_I_q3": ("I_q3", None,    "", "primary_top2"),
    "forced_DRI_B_q3":  ("B_q3", None,    "Let me solve this step by step.\n\n", None),
    "forced_DAI_B_q3":  ("B_q3", None,    "Answer: ", None),
    "forced_DRI_I_q3":  ("I_q3", None,    "Let me solve this step by step.\n\n", None),
    "forced_DAI_I_q3":  ("I_q3", None,    "Answer: ", None),
}


# ---------------------------------------------------------------------
# Prompt building (matches b1_forced_prefix_eval.py)
# ---------------------------------------------------------------------
def build_prompt(row, prefix: str = "") -> str:
    parts = []
    for m in row["prompt"]:
        parts.append(f"{m['role']}\n{m['content']}")
    parts.append("assistant\n")
    s = "\n".join(parts)
    if prefix:
        s = s + prefix
    return s


def aime_prompt_id(prompt_text: str) -> str:
    """Use just the user content (between 'user\\n' and '\\nassistant\\n')."""
    s = prompt_text
    if s.startswith("user\n"):
        s = s[len("user\n"):]
    if s.endswith("\nassistant\n"):
        s = s[:-len("\nassistant\n")]
    elif s.endswith("assistant\n"):
        s = s[:-len("assistant\n")]
    s = s.strip()
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------
# Single-token forcing logits processor
# ---------------------------------------------------------------------
class ForceFirstTokenLogitsProcessor(LogitsProcessor):
    """At the very first generation step, force ``target_id`` (set all other
    logits to a large negative). Subsequent steps return scores unchanged.

    NB: HF will broadcast this across sample dims, so we just compare on the
    seq-length axis: we count 'first step' as the first call to __call__.
    """
    def __init__(self, target_id: int):
        self.target_id = int(target_id)
        self.fired = False

    def __call__(self, input_ids, scores):
        if self.fired:
            return scores
        self.fired = True
        neg = torch.full_like(scores, -1e9)
        neg[..., self.target_id] = 0.0
        return neg


# ---------------------------------------------------------------------
# Intervention top-1 lookup
# ---------------------------------------------------------------------
@torch.no_grad()
def intervention_top1_token_id(model, tokenizer, prompt: str,
                                device: str, max_ctx: int = 4096) -> int:
    """Forward the intervention model on `prompt` once, return its top-1
    token id at position 1 (= first response token)."""
    ids = tokenizer(prompt, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    if ids.shape[1] > max_ctx:
        ids = ids[:, :max_ctx]
    out = model(ids)
    last_logits = out.logits[0, -1, :]
    return int(torch.argmax(last_logits).item())


@torch.no_grad()
def primary_topk_at_first(model, tokenizer, prompt: str,
                          device: str, K: int = 4,
                          max_ctx: int = 4096):
    ids = tokenizer(prompt, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    if ids.shape[1] > max_ctx:
        ids = ids[:, :max_ctx]
    out = model(ids)
    last_logits = out.logits[0, -1, :]
    tk = torch.topk(last_logits, K)
    return [int(x) for x in tk.indices.tolist()]


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
@torch.no_grad()
def hf_generate_n(model, tokenizer, prompt: str, n: int,
                  temperature: float, top_p: float,
                  max_new_tokens: int, force_token_id: Optional[int],
                  device: str, max_prompt_len: int = 4096,
                  seed: int = 1234) -> List[str]:
    """Sample `n` completions of `prompt` under primary model. If
    force_token_id is not None, force the first generated token to be it."""
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    if inputs["input_ids"].shape[1] > max_prompt_len:
        inputs["input_ids"] = inputs["input_ids"][:, :max_prompt_len]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_prompt_len]
    p_len = int(inputs["input_ids"].shape[1])

    procs = LogitsProcessorList()
    if force_token_id is not None:
        procs.append(ForceFirstTokenLogitsProcessor(force_token_id))

    # Each sample needs an independent ForceFirstTokenLogitsProcessor (since
    # `fired` is stateful). Easiest: loop n times. For small n this is fine.
    decoded: List[str] = []
    for s in range(n):
        torch.manual_seed(seed + s)
        if force_token_id is not None:
            procs = LogitsProcessorList(
                [ForceFirstTokenLogitsProcessor(force_token_id)])
        else:
            procs = LogitsProcessorList()
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
            logits_processor=procs,
        )
        new_ids = out[0, p_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        decoded.append(text)
    return decoded


# ---------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=str, default="single_token_C1",
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--data_path", default=AIME_PARQUET)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="1 prompt × 4 sample, very fast end-to-end test.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.smoke:
        args.n_samples = 4
        args.max_new_tokens = 256

    primary_tag, intervention_tag, force_prefix, force_strategy = CONDITIONS[args.condition]
    primary_path = MODEL_PATHS[primary_tag]
    intervention_path = MODEL_PATHS[intervention_tag] if intervention_tag else None

    # GPU pinning. Honor CUDA_VISIBLE_DEVICES if set; otherwise use --gpu_id.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.gpu_id}"
    print(f"[f1] condition={args.condition}  primary={primary_tag}  "
          f"intervention={intervention_tag}  force_strategy={force_strategy}  "
          f"force_prefix={force_prefix!r}  device={device}")

    # Load AIME prompts
    df = pd.read_parquet(args.data_path)
    if args.smoke:
        df = df.head(1).reset_index(drop=True)
    if args.n_shards > 1:
        df = df.iloc[args.shard_id::args.n_shards].reset_index(drop=True)
    print(f"[f1] handling {len(df)} prompts (shard {args.shard_id}/{args.n_shards})")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{args.condition}__shard{args.shard_id}of{args.n_shards}.parquet")
    if os.path.exists(out_path):
        print(f"[f1] {out_path} exists; skipping. Delete to re-run.")
        return

    # Tokenizers / models
    print(f"[f1] loading primary model {primary_path}")
    primary_tok = AutoTokenizer.from_pretrained(primary_path, trust_remote_code=False)
    primary_model = AutoModelForCausalLM.from_pretrained(
        primary_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=False, low_cpu_mem_usage=True)
    primary_model.eval()

    interv_model = interv_tok = None
    if intervention_path:
        print(f"[f1] loading intervention model {intervention_path}")
        interv_tok = AutoTokenizer.from_pretrained(intervention_path, trust_remote_code=False)
        interv_model = AutoModelForCausalLM.from_pretrained(
            intervention_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=False, low_cpu_mem_usage=True)
        interv_model.eval()

    # Determine the target token id per prompt (if needed)
    rows = []
    t0 = time.time()
    for idx, row in df.iterrows():
        prompt = build_prompt(row, prefix=force_prefix)
        pid = aime_prompt_id(prompt)
        gt = str(row["reward_model"]["ground_truth"])

        force_token_id = None
        force_token_text = None
        if force_strategy == "intervention_top1" and interv_model is not None:
            # Use intervention's tokenizer for its forward; both should be
            # Qwen-family with shared vocab so id is mutually compatible.
            tid = intervention_top1_token_id(
                interv_model, interv_tok, prompt, device=device)
            force_token_id = tid
            force_token_text = primary_tok.decode([tid], skip_special_tokens=False)
        elif force_strategy == "primary_top2":
            tk = primary_topk_at_first(primary_model, primary_tok, prompt,
                                        device=device, K=2)
            tid = int(tk[1])
            force_token_id = tid
            force_token_text = primary_tok.decode([tid], skip_special_tokens=False)

        # Generate n samples
        completions = hf_generate_n(
            primary_model, primary_tok, prompt,
            n=args.n_samples,
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            force_token_id=force_token_id,
            device=device, seed=args.seed,
        )

        for sid, text in enumerate(completions):
            full = (force_prefix + text) if force_prefix else text
            res = compute_score(full, gt)
            rows.append({
                "condition": args.condition,
                "primary_tag": primary_tag,
                "intervention_tag": intervention_tag,
                "prompt_id": pid,
                "prompt_idx": int(idx),
                "sample_id": sid,
                "forced_token_id": force_token_id,
                "forced_token_text": force_token_text,
                "force_prefix": force_prefix,
                "response": text,
                "response_chars": len(text),
                "n_resp_tokens": len(primary_tok.encode(text, add_special_tokens=False)),
                "score": float(res["score"]),
                "acc": bool(res["acc"]),
                "pred": str(res["pred"]),
                "mode": classify_mode(text),
            })
        elapsed = time.time() - t0
        rate = (idx + 1) / max(elapsed, 1e-6)
        eta = (len(df) - idx - 1) / max(rate, 1e-6)
        n_correct = sum(1 for r in rows[-args.n_samples:] if r["acc"])
        print(f"[f1]   prompt {idx+1}/{len(df)}  pid={pid}  "
              f"forced_tok={force_token_text!r}  acc={n_correct}/{args.n_samples}  "
              f"rate={rate:.2f}p/s  ETA={eta/60:.1f}m")

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[f1] wrote {out_path}  ({len(rows)} rows)")
    del primary_model
    if interv_model is not None:
        del interv_model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
