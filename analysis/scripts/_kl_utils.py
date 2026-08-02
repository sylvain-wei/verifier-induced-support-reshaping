#!/usr/bin/env python
"""Shared KL/JS utilities for Block A/B/C/D analyses.

Extracted from `a2_position_kl.py` so that the d1–d5, e1, f1, g1 scripts can
share one canonical implementation of:

  - `topk_logits_at_positions(model, tokenizer, prompt, response, positions, K)`
    teacher-forced single forward; returns {pos -> (top_ids, top_logits)}.
  - `topk_union_kl(p_top, q_top)`  — symmetric top-K KL on the mass-renormalized
    union (matches A2's original definition).
  - `topk_union_js(p_top, q_top)`  — symmetric Jensen–Shannon divergence on the
    same mass-renormalized union, base-e nats.

Conventions
-----------
- `*_top` is the pair `(top_ids: np.ndarray[int64, K], top_logits: np.ndarray[float, K])`
  returned by `torch.topk` over the raw logit vector.
- All probabilities are recomputed by softmax over each model's own top-K, then
  zero-padded onto the union. Tokens outside both top-Ks contribute 0 to the
  divergence; this is the same convention as A2 (so position-1 KL = 201.9× tail
  is reproduced exactly with these helpers).
- JS uses `m = 0.5 * (p + q)` and returns `0.5*KL(p||m) + 0.5*KL(q||m)` in nats.
  Bounded above by `ln 2 ≈ 0.6931`.
- Both functions use `eps = 1e-12` for numerical safety.

Position semantics
------------------
A "position" here is the 1-indexed *response token* position. Position 1 means
the model's predicted distribution over the very first token after the prompt.
The corresponding row in `model(full_ids).logits[0]` is `p_len - 2 + pos`,
where `p_len = len(tokenize(prompt))`. See A2's main script for the derivation.

Self-test
---------
Run `python _kl_utils.py` for a synthetic sanity check.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


TopK = Tuple[np.ndarray, np.ndarray]  # (top_ids, top_logits)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Single-forward top-K extractor.
# ---------------------------------------------------------------------------
def topk_logits_at_positions(
    model,
    tokenizer,
    prompt: str,
    response: str,
    positions: List[int],
    K: int = 64,
    device: str = "cuda",
    max_ctx: int = 16384,
) -> Dict[int, TopK]:
    """Run one forward over `prompt + response`; return top-K at each response
    position in `positions` (1-indexed; position 1 = first response token).

    If `pos > n_resp`, that position is silently skipped.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    full_ids = tokenizer(prompt + response, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    if full_ids.shape[1] > max_ctx:
        full_ids = full_ids[:, :max_ctx]
    n_resp = full_ids.shape[1] - p_len
    out: Dict[int, TopK] = {}
    if n_resp <= 0:
        return out

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (T, V)

    for pos in positions:
        if pos < 1 or pos > n_resp:
            continue
        idx = p_len - 2 + pos
        if idx < 0 or idx >= logits.shape[0]:
            continue
        v = logits[idx].float()
        tk = torch.topk(v, K)
        out[pos] = (tk.indices.cpu().numpy(), tk.values.cpu().numpy())
    return out


def topk_logits_all_positions(
    model,
    tokenizer,
    prompt: str,
    response: str,
    K: int = 64,
    device: str = "cuda",
    max_ctx: int = 16384,
) -> List[TopK]:
    """Variant: return top-K for EVERY response position (1..n_resp). Used by
    Block A's per-token JS sparsity histograms (D.1, D.2).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    full_ids = tokenizer(prompt + response, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    if full_ids.shape[1] > max_ctx:
        full_ids = full_ids[:, :max_ctx]
    n_resp = full_ids.shape[1] - p_len
    out: List[TopK] = []
    if n_resp <= 0:
        return out

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (T, V)

    # response positions 1..n_resp → indices p_len-1 .. p_len-2+n_resp
    start = p_len - 1
    end = p_len - 2 + n_resp + 1
    slab = logits[start:end].float()  # (n_resp, V)
    tk = torch.topk(slab, K, dim=-1)
    ids_np = tk.indices.cpu().numpy()
    lg_np = tk.values.cpu().numpy()
    for i in range(n_resp):
        out.append((ids_np[i], lg_np[i]))
    return out


# ---------------------------------------------------------------------------
# Union-renormalized distributions.
# ---------------------------------------------------------------------------
def _union_probs(p_top: TopK, q_top: TopK) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p, q), both of shape (|union|,), each normalized to sum 1."""
    ids_p, lg_p = p_top
    ids_q, lg_q = q_top
    union = sorted(set(ids_p.tolist()) | set(ids_q.tolist()))
    p_map = {int(i): float(l) for i, l in zip(ids_p, lg_p)}
    q_map = {int(i): float(l) for i, l in zip(ids_q, lg_q)}
    # tokens not in a model's top-K are zero-probability under that model
    p_logits = np.array([p_map.get(i, -1e9) for i in union])
    q_logits = np.array([q_map.get(i, -1e9) for i in union])
    p = np.exp(p_logits - p_logits.max())
    p = p / p.sum()
    q = np.exp(q_logits - q_logits.max())
    q = q / q.sum()
    return p, q


def topk_union_kl(p_top: TopK, q_top: TopK) -> Dict[str, float]:
    """Symmetric top-K KL on the mass-renormalized union (nats).

    Returns {kl_pq, kl_qp, kl_sym}. `kl_sym = 0.5*(KL(p||q) + KL(q||p))`.
    Matches A2's original `topk_union_kl` exactly (only the dict keys differ
    from A2's hardcoded `kl_b0_b2r1` / `kl_b2r1_b0` / `kl_sym`).
    """
    p, q = _union_probs(p_top, q_top)
    kl_pq = float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))
    kl_qp = float(np.sum(q * (np.log(q + EPS) - np.log(p + EPS))))
    return {"kl_pq": kl_pq, "kl_qp": kl_qp, "kl_sym": 0.5 * (kl_pq + kl_qp)}


def topk_union_js(p_top: TopK, q_top: TopK) -> float:
    """Jensen–Shannon divergence on the mass-renormalized union (nats).

    `0.5*KL(p||m) + 0.5*KL(q||m)` with `m = 0.5*(p+q)`. Bounded in [0, ln 2].
    """
    p, q = _union_probs(p_top, q_top)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p + EPS) - np.log(m + EPS))))
    kl_qm = float(np.sum(q * (np.log(q + EPS) - np.log(m + EPS))))
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------
def _selftest():
    """Sanity: identical top-K → KL=JS=0; disjoint top-K → JS=ln 2."""
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    same = (ids, logits)
    kl = topk_union_kl(same, same)
    js = topk_union_js(same, same)
    assert abs(kl["kl_sym"]) < 1e-8, f"identical KL should be 0, got {kl}"
    assert abs(js) < 1e-8, f"identical JS should be 0, got {js}"
    print(f"[ok] identical:    KL_sym={kl['kl_sym']:.2e}  JS={js:.2e}")

    # disjoint supports with non-overlapping ids
    p = (np.array([1, 2], dtype=np.int64), np.array([5.0, 0.0], dtype=np.float32))
    q = (np.array([3, 4], dtype=np.int64), np.array([5.0, 0.0], dtype=np.float32))
    js = topk_union_js(p, q)
    # disjoint → JS ≈ ln 2 (only true when each side puts almost all mass on
    # a single distinct id; our 5-vs-0 logit gives ~99.3% mass on top-1, so JS
    # should be very close to ln 2 ≈ 0.693)
    assert abs(js - np.log(2)) < 0.02, f"disjoint JS should ≈ ln2, got {js}"
    print(f"[ok] disjoint:     JS={js:.4f}  (target ≈ ln 2 = {np.log(2):.4f})")

    # JS symmetric
    js_pq = topk_union_js(p, q)
    js_qp = topk_union_js(q, p)
    assert abs(js_pq - js_qp) < 1e-10, "JS should be symmetric"
    print(f"[ok] symmetric:    JS(p,q)={js_pq:.4f} == JS(q,p)={js_qp:.4f}")

    # KL symmetric in our sym version
    kl_pq = topk_union_kl(p, q)
    kl_qp = topk_union_kl(q, p)
    assert abs(kl_pq["kl_sym"] - kl_qp["kl_sym"]) < 1e-10
    print(f"[ok] KL sym key:   KL_sym(p,q)={kl_pq['kl_sym']:.4f} "
          f"== KL_sym(q,p)={kl_qp['kl_sym']:.4f}")

    # Partial overlap: shared top-1
    p = (np.array([1, 2, 3], dtype=np.int64), np.array([5.0, 1.0, 0.0], dtype=np.float32))
    q = (np.array([1, 4, 5], dtype=np.int64), np.array([5.0, 1.0, 0.0], dtype=np.float32))
    js = topk_union_js(p, q)
    kl = topk_union_kl(p, q)
    print(f"[ok] shared top-1: JS={js:.4f}  KL_sym={kl['kl_sym']:.4f}  "
          f"(both should be > 0 but < ln 2)")
    assert 0 < js < np.log(2)
    assert kl["kl_sym"] > 0

    # KL >= JS (well-known bound: JS ≤ (1/2)(KL(p||q) + KL(q||p)) / 2 is not
    # the bound; actually JS ≤ ln 2, no direct ordering with KL_sym in general.
    # Just check finiteness.)
    print(f"[ok] all self-tests passed.")


if __name__ == "__main__":
    _selftest()
