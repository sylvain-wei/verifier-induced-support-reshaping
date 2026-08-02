#!/usr/bin/env python
"""Precompute per-token JS / KL for one (base, rl, dataset) triple and write
a compact parquet that the d1-d5 / e1 scripts can re-read without ever
touching the bulky topk arrays again.

Output schema (one row per response token):
    base_tag, rl_tag, dataset, prompt_id, sample_id, t,
    n_resp_tokens, rel_pos, token_id_B,
    logp_B, logp_RL, js, kl_B_RL, kl_RL_B

For the 10 pairs we care about this collapses ~6 GB of topk arrays down to
~300-500 MB of scalar JS columns total — d-scripts then load those instantly.

Usage:
    python precompute_js.py --base B_q3 --rl M_q3 --dataset aime
    # writes analysis/data/wave2_js/{base}__{rl}__{ds}.parquet

Bottleneck previously was the Python-level row loop in _js_join. Here we
keep the same logic but factored so multiple pair-jobs can run in parallel
via the orchestrator (each pair is CPU-only, ~one core).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kl_utils import topk_union_kl, topk_union_js  # noqa: E402

REPO = os.environ.get("PROJECT_ROOT", ".")
WAVE1 = f"{REPO}/analysis/data/wave1"
WAVE15 = f"{REPO}/analysis/data/wave1_5"
OUT_DIR = f"{REPO}/analysis/data/wave2_js"


def _vector_js_kl(ids_b_arr, lg_b_arr, ids_r_arr, lg_r_arr):
    """Row-wise JS / KL. Numpy-based but still a Python loop over rows.

    Returns (js, kl_B_RL, kl_RL_B) arrays of length n.
    """
    n = len(ids_b_arr)
    js = np.empty(n, dtype=np.float32)
    kl_BR = np.empty(n, dtype=np.float32)
    kl_RB = np.empty(n, dtype=np.float32)
    for i in range(n):
        ib = np.asarray(ids_b_arr[i], dtype=np.int64)
        lb = np.asarray(lg_b_arr[i], dtype=np.float32)
        ir = np.asarray(ids_r_arr[i], dtype=np.int64)
        lr = np.asarray(lg_r_arr[i], dtype=np.float32)
        kl = topk_union_kl((ib, lb), (ir, lr))
        js[i] = topk_union_js((ib, lb), (ir, lr))
        kl_BR[i] = kl["kl_pq"]
        kl_RB[i] = kl["kl_qp"]
        if i % 200000 == 0 and i > 0:
            print(f"    [precompute] {i}/{n} ({i/n*100:.1f}%)")
    return js, kl_BR, kl_RB


def precompute(base: str, rl: str, ds: str,
               wave1_dir: str = WAVE1, wave15_dir: str = WAVE15,
               out_dir: str = OUT_DIR, force: bool = False) -> str:
    out_path = os.path.join(out_dir, f"{base}__{rl}__{ds}.parquet")
    if os.path.exists(out_path) and not force:
        print(f"[precompute_js] {out_path} exists; skip (use --force)")
        return out_path
    os.makedirs(out_dir, exist_ok=True)

    pB = os.path.join(wave1_dir, f"{base}__{ds}.parquet")
    pR = os.path.join(wave15_dir, f"{rl}_on_B__{ds}.parquet")
    if not os.path.exists(pB):
        raise FileNotFoundError(pB)
    if not os.path.exists(pR):
        raise FileNotFoundError(pR)

    t0 = time.time()
    print(f"[precompute_js] loading {pB}")
    dfB = pd.read_parquet(pB)
    print(f"[precompute_js] loading {pR}")
    dfR = pd.read_parquet(pR)

    dfB = dfB.rename(columns={"logp": "logp_B",
                              "topk_ids": "ids_B",
                              "topk_logits": "lg_B",
                              "token_id": "token_id_B"})[
        ["prompt_id", "sample_id", "t",
         "token_id_B", "logp_B", "ids_B", "lg_B"]
    ]
    dfR = dfR.rename(columns={"logp": "logp_RL",
                              "topk_ids": "ids_R",
                              "topk_logits": "lg_R"})[
        ["prompt_id", "sample_id", "t", "logp_RL", "ids_R", "lg_R"]
    ]
    j = dfB.merge(dfR, on=["prompt_id", "sample_id", "t"], how="inner")
    if j.empty:
        print(f"[precompute_js] empty join for {base}/{rl}/{ds}")
        return out_path
    n_resp = (j.groupby(["prompt_id", "sample_id"])["t"]
                .max().rename("n_resp_tokens").reset_index())
    j = j.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    j["rel_pos"] = j["t"] / j["n_resp_tokens"]

    print(f"[precompute_js] joined {len(j)} rows  load_time={time.time()-t0:.1f}s, "
          f"computing JS/KL...")
    t1 = time.time()
    js, kl_BR, kl_RB = _vector_js_kl(j["ids_B"].values,
                                     j["lg_B"].values,
                                     j["ids_R"].values,
                                     j["lg_R"].values)
    print(f"[precompute_js] JS done in {time.time()-t1:.1f}s "
          f"({len(j)/(time.time()-t1):.0f} rows/s)")
    j["js"] = js
    j["kl_B_RL"] = kl_BR
    j["kl_RL_B"] = kl_RB
    j["base_tag"] = base
    j["rl_tag"] = rl
    j["dataset"] = ds

    keep = ["base_tag", "rl_tag", "dataset",
            "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
            "token_id_B", "logp_B", "logp_RL",
            "js", "kl_B_RL", "kl_RL_B"]
    j[keep].to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path) / 1e6
    print(f"[precompute_js] wrote {out_path}  ({sz:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["aime", "ifeval", "ifbench"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    precompute(args.base, args.rl, args.dataset, force=args.force)


if __name__ == "__main__":
    main()
