#!/usr/bin/env python
"""A1 — Teacher-forced log-prob of known-correct DRI traces (b0 vs b2r1-100).

Loads (correct, DRI-classified) AIME rollouts from b1r1 step 0 (which equals b0),
then computes token-averaged log-prob (nats/tok) of the response under both b0
and b2r1-100 via HuggingFace transformers (NOT vLLM — we need exact teacher-
forced log-probs).

Decision rule:
  - Δ ≤ 0.05 nats/tok → DRI generation distribution intact (mode-loss reading)
  - 0.05 < Δ ≤ 0.20  → ambiguous; expand cap to 128 + add b1r1-100 control
  - Δ > 0.20         → real knowledge degradation (re-scope)

Usage
-----
  python a1_teacher_forced_logprob.py \\
      --b0     "${PROJECT_ROOT}/models/Qwen3-8B-Base" \\
      --b2r1   .../b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface \\
      --rollouts .../b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl \\
      --cap 64 \\
      --out_pairs   analysis/data/a1_logprob.parquet \\
      --out_summary analysis/tables/a1_summary.csv
"""
import argparse
import hashlib
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# Make our local opening-mode classifier importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_opening_modes import classify_mode  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))

AIME_MARKER = "Solve the following math problem step by step"


def _extract_user_content(s: str) -> str:
    """val_rollout `input` is 'user\\n<content>\\nassistant\\n'."""
    s2 = s
    if s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def load_correct_dri_traces(jsonl_path: str, max_per_prompt: int = 2,
                            cap: int = 64) -> List[dict]:
    """Read existing b0 AIME rollouts; keep (acc=True, mode='DRI') pairs.

    Returns list of dicts with keys:
        prompt_id (md5-style hash from input), prompt_text (raw with template),
        user_content, response, response_chars.
    """
    out: List[dict] = []
    per_prompt: dict = {}
    with open(jsonl_path) as f:
        for ln in f:
            o = json.loads(ln)
            inp = o.get("input", "")
            if AIME_MARKER not in inp:
                continue
            acc_raw = o.get("acc")
            if isinstance(acc_raw, bool):
                acc = acc_raw
            else:
                try:
                    acc = bool(float(o.get("score", 0.0)) >= 1.0)
                except Exception:
                    continue
            if not acc:
                continue
            resp = o.get("output", "")
            if not resp:
                continue
            mode = classify_mode(resp)
            if mode != "DRI":
                continue
            uc = _extract_user_content(inp)
            pid = hashlib.md5(uc.encode("utf-8")).hexdigest()[:16]
            n_seen = per_prompt.get(pid, 0)
            if n_seen >= max_per_prompt:
                continue
            per_prompt[pid] = n_seen + 1
            out.append({
                "prompt_id": pid,
                "prompt_text": inp,
                "user_content": uc,
                "response": resp,
                "response_chars": len(resp),
            })
            if len(out) >= cap:
                break
    return out


def _score_one_pair(model, tokenizer, prompt: str, response: str,
                    device: str = "cuda") -> Tuple[float, int]:
    """Token-averaged log-prob (nats/tok) of `response` given `prompt`.

    Returns (mean_nats_per_token, n_response_tokens).
    """
    # Tokenize separately so we know the prompt boundary.
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    full = prompt + response
    full_ids = tokenizer(full, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= p_len:
        return float("nan"), 0
    # Truncate to model's context length to avoid OOM on long responses.
    max_ctx = getattr(model.config, "max_position_embeddings", 32768) or 32768
    max_ctx = min(max_ctx, 16384)  # safety
    if full_ids.shape[1] > max_ctx:
        full_ids = full_ids[:, :max_ctx]
    n_resp = full_ids.shape[1] - p_len

    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        # logits[:, t-1] predicts token t. So response logits are at positions
        # p_len-1 .. T-2, response tokens are at positions p_len .. T-1.
        log_probs = torch.log_softmax(logits[0, p_len - 1: -1, :].float(), dim=-1)
        target = full_ids[0, p_len:]
        token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mean_lp = token_lp.mean().item()
    return mean_lp, int(n_resp)


def score_logprobs(model_path: str, pairs: List[dict],
                   dtype=torch.bfloat16) -> List[Tuple[float, int]]:
    """Load HF model, score all pairs, return list of (mean_lp, n_resp_tokens)."""
    print(f"[a1] loading model {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="balanced",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    results = []
    for i, p in enumerate(pairs):
        mean_lp, n_tok = _score_one_pair(model, tokenizer,
                                         p["prompt_text"], p["response"])
        results.append((mean_lp, n_tok))
        if (i + 1) % 8 == 0 or i == len(pairs) - 1:
            print(f"[a1]   scored {i+1}/{len(pairs)}  mean_lp={mean_lp:.4f}  n_tok={n_tok}")
    # Free.
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", required=True)
    ap.add_argument("--b2r1", required=True)
    ap.add_argument("--rollouts", required=True,
                    help="b0 AIME rollouts JSONL (b1r1 step 0)")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--max_per_prompt", type=int, default=8)
    ap.add_argument("--out_pairs", default=os.path.join(PROJECT_ROOT, "analysis", "data", "a1_logprob.parquet"))
    ap.add_argument("--out_summary", default=os.path.join(PROJECT_ROOT, "analysis", "tables", "a1_summary.csv"))
    args = ap.parse_args()

    pairs = load_correct_dri_traces(args.rollouts, args.max_per_prompt, args.cap)
    print(f"[a1] loaded {len(pairs)} (correct, DRI) reference pairs")
    if not pairs:
        print("[a1] ERROR: no pairs loaded; check rollouts path / classifier")
        sys.exit(1)

    # Score both checkpoints, sequentially (one model on GPU at a time).
    res_b0 = score_logprobs(args.b0, pairs)
    res_b2 = score_logprobs(args.b2r1, pairs)

    rows = []
    for p, (lp0, n0), (lp2, n2) in zip(pairs, res_b0, res_b2):
        rows.append({
            "prompt_id": p["prompt_id"],
            "user_content_head": p["user_content"][:120],
            "response_chars": p["response_chars"],
            "n_resp_tokens_b0": n0,
            "n_resp_tokens_b2r1": n2,
            "lp_b0": lp0,
            "lp_b2r1": lp2,
            "delta": lp0 - lp2,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    df.to_parquet(args.out_pairs, index=False)
    print(f"[a1] wrote {args.out_pairs}")

    delta_mean = float(df["delta"].mean())
    summary = pd.DataFrame([
        {"model": "b0",       "mean_lp": float(df["lp_b0"].mean()),
         "std_lp": float(df["lp_b0"].std()),
         "n_pairs": len(df)},
        {"model": "b2r1-100", "mean_lp": float(df["lp_b2r1"].mean()),
         "std_lp": float(df["lp_b2r1"].std()),
         "n_pairs": len(df)},
        {"model": "delta_b0_minus_b2r1",
         "mean_lp": delta_mean,
         "std_lp": float(df["delta"].std()),
         "n_pairs": len(df)},
    ])
    summary.to_csv(args.out_summary, index=False)
    print(f"[a1] wrote {args.out_summary}")
    print(f"[a1] Δ (b0 − b2r1) mean = {delta_mean:.4f} nats/tok over {len(df)} pairs")
    if delta_mean <= 0.05:
        verdict = "INTACT (mode-loss reading)"
    elif delta_mean <= 0.20:
        verdict = "AMBIGUOUS (expand cap=128, add b1r1-100 control)"
    else:
        verdict = "REAL KNOWLEDGE DEGRADATION (re-scope)"
    print(f"[a1] verdict: {verdict}")


if __name__ == "__main__":
    main()
