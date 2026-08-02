#!/usr/bin/env python
"""h3 — lm_head weight-delta SVD for Figure 1 panel (d).

Goal: load lm_head.weight from B_q3 / M_q3 / I_q3 safetensors, compute
    ΔW_I = lm_head[I_q3] - lm_head[B_q3]
    ΔW_M = lm_head[M_q3] - lm_head[B_q3]
randomized SVD k=32 on each. Output:
  - top-32 singular values (per delta)
  - top-3 left-singular vectors' cosine with the unembedding rows of the
    7 routing tokens (4 DRI + 3 DAI), where DRI={To, Step, Let, We} and
    DAI={Answer, The, ":"}.
  - top-3 cumulative variance fraction (low-rank-ness check).

Output: analysis/data/h3_lm_head_svd.parquet (one row per (delta, component, token)).

Usage:
    python analysis/scripts/h3_lm_head_svd.py
    # or with explicit ckpts:
    python analysis/scripts/h3_lm_head_svd.py --b_path ... --m_path ... --i_path ...

CPU-only; ~15-30 min total for both deltas.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# Routing tokens validated against Qwen3-8B-Base tokenizer (single-token IDs).
T_DRI = {1249: "To", 8304: "Step", 10061: "Let", 1654: "We"}
T_DAI = {16141: "Answer", 785: "The", 25: ":"}

ROOT = os.environ.get("PROJECT_ROOT", ".")
DEFAULT_B = f"{ROOT}/models/Qwen3-8B-Base"
DEFAULT_M = (
    f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
    "b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface"
)
DEFAULT_I = (
    f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/"
    "b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface"
)


def load_lm_head(model_dir: str) -> np.ndarray:
    """Load just lm_head.weight from a HF safetensors checkpoint dir.

    Returns float32 numpy array of shape (V, d). Uses the torch backend so we
    can correctly handle bf16-stored ckpts (verl saves bf16; numpy backend can't
    read bf16 directly).
    """
    import torch
    from safetensors import safe_open
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    shard_name = idx["weight_map"]["lm_head.weight"]
    shard_path = os.path.join(model_dir, shard_name)
    t0 = time.time()
    with safe_open(shard_path, framework="pt") as f:
        w = f.get_tensor("lm_head.weight")
    w_np = w.to(torch.float32).cpu().numpy()
    print(f"  loaded lm_head from {os.path.basename(model_dir)}: shape={w_np.shape}, "
          f"src_dtype={w.dtype}, {time.time()-t0:.1f}s")
    return w_np


def randomized_svd(M: np.ndarray, k: int = 32, n_oversamples: int = 16,
                   n_iter: int = 4, seed: int = 0):
    """Randomized SVD of M ∈ R^{V x d}, returning top-k components.

    Returns (U[:, :k], s[:k], Vt[:k, :]).
    """
    V, d = M.shape
    rng = np.random.default_rng(seed)
    p = k + n_oversamples
    Omega = rng.standard_normal((d, p), dtype=np.float32)
    Y = M @ Omega                          # (V, p)
    # Power iteration for accuracy
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Z = M.T @ Q                        # (d, p)
        Q2, _ = np.linalg.qr(Z)
        Y = M @ Q2                         # (V, p)
    Q, _ = np.linalg.qr(Y)                 # (V, p)
    B = Q.T @ M                            # (p, d)
    U_b, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    return U[:, :k], s[:k], Vt[:k, :]


def cos_to_unembedding_rows(u: np.ndarray, W_unembed: np.ndarray, token_ids):
    """Cosine similarity between u (R^V) and the rows of W_unembed (R^{V x d}) at token_ids.

    The 'rows of W_unembed' are vectors in R^d; but u is in R^V. They are NOT
    in the same space. The correct geometric object is: u indexes into V (token
    space), so cos(u, e_token) is just u[token] / ||u||.
    """
    u_norm = np.linalg.norm(u) + 1e-12
    out = {}
    for t in token_ids:
        # cos(u, one_hot_t) = u[t] / ||u||  (since one_hot has unit norm)
        out[t] = float(u[t] / u_norm)
    return out


def analyze_delta(name: str, dW: np.ndarray, k: int = 32):
    """Run SVD on dW, summarize top components, return list of result rows."""
    V, d = dW.shape
    fro = float(np.linalg.norm(dW))
    print(f"\n[{name}] dW shape={dW.shape} ||dW||_F={fro:.4f}")
    t0 = time.time()
    U, s, Vt = randomized_svd(dW, k=k, n_iter=4, seed=42)
    print(f"[{name}] SVD k={k} done in {time.time()-t0:.1f}s")
    s_norm = s / fro
    var_frac = (s ** 2) / (fro ** 2)
    cum_var3 = float(var_frac[:3].sum())
    cum_var10 = float(var_frac[:10].sum())
    print(f"[{name}] top-3 cum var fraction = {cum_var3:.4f}; top-10 = {cum_var10:.4f}")
    print(f"[{name}] s/||dW||_F top-5: {s_norm[:5].tolist()}")

    rows = []
    # Per-component scree row + per-(component, routing-token) cosine row.
    for i in range(k):
        rows.append(dict(
            delta_name=name,
            kind="scree",
            component=i + 1,
            sigma=float(s[i]),
            sigma_over_fro=float(s_norm[i]),
            var_fraction=float(var_frac[i]),
            cum_var_fraction=float(var_frac[: i + 1].sum()),
            token_id=-1, token_label="", token_class="", cosine=float("nan"),
        ))
    for ci in range(min(3, k)):
        u = U[:, ci]
        for tid, lbl in T_DRI.items():
            cos = float(u[tid] / (np.linalg.norm(u) + 1e-12))
            rows.append(dict(
                delta_name=name, kind="cos_DRI", component=ci + 1,
                sigma=float(s[ci]), sigma_over_fro=float(s_norm[ci]),
                var_fraction=float(var_frac[ci]),
                cum_var_fraction=float(var_frac[: ci + 1].sum()),
                token_id=int(tid), token_label=lbl, token_class="DRI", cosine=cos,
            ))
        for tid, lbl in T_DAI.items():
            cos = float(u[tid] / (np.linalg.norm(u) + 1e-12))
            rows.append(dict(
                delta_name=name, kind="cos_DAI", component=ci + 1,
                sigma=float(s[ci]), sigma_over_fro=float(s_norm[ci]),
                var_fraction=float(var_frac[ci]),
                cum_var_fraction=float(var_frac[: ci + 1].sum()),
                token_id=int(tid), token_label=lbl, token_class="DAI", cosine=cos,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b_path", default=DEFAULT_B)
    ap.add_argument("--m_path", default=DEFAULT_M)
    ap.add_argument("--i_path", default=DEFAULT_I)
    ap.add_argument("--out", default=f"{ROOT}/analysis/data/h3_lm_head_svd.parquet")
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    print("[h3] loading lm_head from 3 ckpts...")
    W_B = load_lm_head(args.b_path)
    W_M = load_lm_head(args.m_path)
    W_I = load_lm_head(args.i_path)

    if W_B.shape != W_M.shape or W_B.shape != W_I.shape:
        raise RuntimeError(
            f"lm_head shape mismatch: B={W_B.shape} M={W_M.shape} I={W_I.shape}"
        )

    rows = []
    rows += analyze_delta("M_minus_B", W_M - W_B, k=args.k)
    rows += analyze_delta("I_minus_B", W_I - W_B, k=args.k)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[h3] wrote {args.out} ({len(df)} rows)")
    # Headline summary
    print("\n[h3] SUMMARY")
    for delta in ("M_minus_B", "I_minus_B"):
        sub = df[(df.delta_name == delta) & (df.kind == "scree")].sort_values("component")
        cum3 = float(sub.iloc[2].cum_var_fraction)
        cum10 = float(sub.iloc[9].cum_var_fraction)
        print(f"  {delta}: top-3 cum var = {cum3:.4f}, top-10 = {cum10:.4f}")
    for cls in ("cos_DRI", "cos_DAI"):
        for delta in ("M_minus_B", "I_minus_B"):
            sub = df[(df.delta_name == delta) & (df.kind == cls) & (df.component == 1)]
            mean = float(sub.cosine.abs().mean())
            print(f"  {delta} component#1 {cls}: |cos| mean = {mean:.4f}")


if __name__ == "__main__":
    main()
