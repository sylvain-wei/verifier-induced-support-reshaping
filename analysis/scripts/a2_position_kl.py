#!/usr/bin/env python
"""A2 — First-token KL decomposition (b0 vs b2r1-100).

For each (prompt, reference DRI response) pair from A1, run a single forward
pass through both models and compute a top-K KL between b2r1-100 and b0 at:
  - position 1 (first response token, immediately after `assistant\\n`)
  - positions 8, 16, 32, 64 of the reference response (clamped to length)

Decision rule:
  - kl_pos1 / mean(kl_pos>=8) >= 5×  → KL concentrated at routing token (mode-prior shift)
  - ratio < 2×                       → diffuse (reframe)

Top-K KL implementation: take union of top-K tokens under both p and q, mass-
renormalize within that union, compute KL(p || q) and KL(q || p). We report
both and treat the symmetric mean as the headline KL.
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_opening_modes import classify_mode  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))

AIME_MARKER = "Solve the following math problem step by step"


def _extract_user_content(s: str) -> str:
    s2 = s
    if s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def load_canonical_traces(jsonl_path: str, cap: int = 30) -> List[dict]:
    """Pick one (correct, DRI) response per AIME prompt — longest one."""
    by_prompt: Dict[str, dict] = {}
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
            if not resp or classify_mode(resp) != "DRI":
                continue
            uc = _extract_user_content(inp)
            pid = hashlib.md5(uc.encode("utf-8")).hexdigest()[:16]
            cur = by_prompt.get(pid)
            if cur is None or len(resp) > len(cur["response"]):
                by_prompt[pid] = {
                    "prompt_id": pid,
                    "prompt_text": inp,
                    "user_content": uc,
                    "response": resp,
                }
    out = list(by_prompt.values())[:cap]
    return out


def topk_logits_at_positions(model, tokenizer, prompt: str, response: str,
                             positions: List[int], K: int = 64,
                             device: str = "cuda"):
    """Returns dict[pos] -> (top_ids: np.array(K,), top_logits: np.array(K,)).

    Position 1 here means the FIRST response token (i.e., the model's
    predicted distribution over the very first token after the prompt).
    Position k means the predicted distribution over the k-th response token.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    full = prompt + response
    full_ids = tokenizer(full, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    max_ctx = 16384
    if full_ids.shape[1] > max_ctx:
        full_ids = full_ids[:, :max_ctx]
    n_resp = full_ids.shape[1] - p_len
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if n_resp <= 0:
        return out

    with torch.no_grad():
        outp = model(full_ids)
        logits = outp.logits[0]  # (T, V)

    for pos in positions:
        # Predicted distribution over the pos-th response token.
        # Position index in logits is (p_len - 1) + (pos - 1) = p_len + pos - 2.
        # If pos exceeds n_resp, clamp.
        eff_pos = min(pos, n_resp)
        idx = p_len - 2 + eff_pos
        if idx < 0 or idx >= logits.shape[0]:
            continue
        v = logits[idx].float()
        topk = torch.topk(v, K)
        out[pos] = (topk.indices.cpu().numpy(), topk.values.cpu().numpy())
    return out


def topk_union_kl(b0_top, b2_top, K_eff: int = 64) -> Dict[str, float]:
    """Compute mass-renormalized top-K KL between two distributions, given
    each model's (top_ids, top_logits). Returns {kl_b2_b0, kl_b0_b2, kl_sym}."""
    ids0, lg0 = b0_top
    ids1, lg1 = b2_top
    union = sorted(set(ids0.tolist()) | set(ids1.tolist()))
    # Build full softmax probs over each model's own top-K then re-normalize
    # to the union — tokens outside top-K get probability 0.
    p_dict = {int(i): float(l) for i, l in zip(ids0, lg0)}
    q_dict = {int(i): float(l) for i, l in zip(ids1, lg1)}
    # Convert raw logits to probabilities via softmax over the model's own top-K
    p_logits = np.array([p_dict.get(i, -1e9) for i in union])
    q_logits = np.array([q_dict.get(i, -1e9) for i in union])
    # softmax to get unnormalized within union, then renormalize.
    p = np.exp(p_logits - p_logits.max())
    q = np.exp(q_logits - q_logits.max())
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-12
    kl_p_q = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    kl_q_p = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
    return {
        "kl_b0_b2r1": kl_p_q,    # KL(b0 || b2r1)
        "kl_b2r1_b0": kl_q_p,    # KL(b2r1 || b0)
        "kl_sym":     0.5 * (kl_p_q + kl_q_p),
    }


def collect_topk(model_path: str, pairs: List[dict], positions: List[int],
                 K: int = 64) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """For one model, return prompt_id -> {pos -> (top_ids, top_logits)}."""
    print(f"[a2] loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    out: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for i, p in enumerate(pairs):
        out[p["prompt_id"]] = topk_logits_at_positions(
            model, tokenizer, p["prompt_text"], p["response"], positions, K=K,
        )
        if (i + 1) % 6 == 0 or i == len(pairs) - 1:
            print(f"[a2]   {i+1}/{len(pairs)} done")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", required=True)
    ap.add_argument("--b2r1", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--positions", default="1,8,16,32,64")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--out_pairs", default=os.path.join(PROJECT_ROOT, "analysis", "data", "a2_kl.parquet"))
    ap.add_argument("--out_summary", default=os.path.join(PROJECT_ROOT, "analysis", "tables", "a2_summary.csv"))
    args = ap.parse_args()

    positions = [int(s) for s in args.positions.split(",") if s.strip()]
    pairs = load_canonical_traces(args.rollouts, cap=args.cap)
    print(f"[a2] loaded {len(pairs)} canonical (correct, DRI) traces")
    if not pairs:
        print("[a2] ERROR: no traces loaded")
        sys.exit(1)

    b0_topk = collect_topk(args.b0, pairs, positions, K=args.K)
    b2_topk = collect_topk(args.b2r1, pairs, positions, K=args.K)

    rows = []
    for p in pairs:
        pid = p["prompt_id"]
        for pos in positions:
            if pos not in b0_topk[pid] or pos not in b2_topk[pid]:
                continue
            kl = topk_union_kl(b0_topk[pid][pos], b2_topk[pid][pos], K_eff=args.K)
            rows.append({"prompt_id": pid, "position": pos, **kl})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    df.to_parquet(args.out_pairs, index=False)
    print(f"[a2] wrote {args.out_pairs}")

    summary = (df.groupby("position")
                 .agg(mean_kl_sym=("kl_sym", "mean"),
                      p50=("kl_sym", "median"),
                      p90=("kl_sym", lambda s: float(np.percentile(s, 90))),
                      mean_kl_b2_b0=("kl_b2r1_b0", "mean"),
                      mean_kl_b0_b2=("kl_b0_b2r1", "mean"),
                      n=("kl_sym", "size"))
                 .reset_index())
    summary.to_csv(args.out_summary, index=False)
    print(f"[a2] wrote {args.out_summary}")
    pos1 = float(summary[summary["position"] == 1]["mean_kl_sym"].iloc[0]) \
        if (summary["position"] == 1).any() else float("nan")
    tail = summary[summary["position"] >= 8]["mean_kl_sym"].mean() \
        if (summary["position"] >= 8).any() else float("nan")
    ratio = pos1 / tail if tail > 0 else float("nan")
    print(f"[a2] mean_kl_pos1 = {pos1:.4f}   mean_kl_pos>=8 = {tail:.4f}   ratio = {ratio:.2f}×")
    if not np.isnan(ratio):
        if ratio >= 5:
            verdict = "POSITIVE: KL concentrated at routing token (mode-prior shift)"
        elif ratio >= 2:
            verdict = "PARTIAL: opening-window KL > tail; reframe"
        else:
            verdict = "DIFFUSE: combine with A1 reading"
        print(f"[a2] verdict: {verdict}")


if __name__ == "__main__":
    main()
