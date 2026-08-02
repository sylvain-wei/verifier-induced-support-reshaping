#!/usr/bin/env python
"""WAVE 5 — Precompute per-token JS / KL for one (base, run-step20, dataset)
triple, mirroring precompute_js.py but with explicit input/output paths so
the predictive routing-JS analysis (h11/h12/h13) can reuse the same JS
computation logic without disturbing wave2_js outputs.

Output schema mirrors wave2_js/*.parquet exactly:
    base_tag, rl_tag, dataset, prompt_id, sample_id, t,
    n_resp_tokens, rel_pos, token_id_B,
    logp_B, logp_RL, js, kl_B_RL, kl_RL_B

Usage:
    python precompute_js_predictive.py \\
        --base_tag B_q3 \\
        --base_parquet analysis/data/wave1/B_q3__aime.parquet \\
        --rl_tag I_q3_step20_S1soft \\
        --rl_parquet analysis/data/wave5_predictive/I_q3_step20_S1soft__aime.parquet \\
        --dataset aime \\
        --out_dir analysis/data/wave5_predictive_js
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


def _vector_js_kl(ids_b_arr, lg_b_arr, ids_r_arr, lg_r_arr):
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
            print(f"    [precompute] {i}/{n} ({i/n*100:.1f}%)", flush=True)
    return js, kl_BR, kl_RB


def precompute(base_tag, base_parquet, rl_tag, rl_parquet, dataset,
               out_dir, force=False):
    out_path = os.path.join(out_dir, f"{base_tag}__{rl_tag}__{dataset}.parquet")
    if os.path.exists(out_path) and not force:
        print(f"[precompute] {out_path} exists; skip (use --force)")
        return out_path
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(base_parquet):
        raise FileNotFoundError(base_parquet)
    if not os.path.exists(rl_parquet):
        raise FileNotFoundError(rl_parquet)

    t0 = time.time()
    print(f"[precompute] loading base: {base_parquet}")
    dfB = pd.read_parquet(base_parquet)
    print(f"[precompute] loading rl  : {rl_parquet}")
    dfR = pd.read_parquet(rl_parquet)

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
        print(f"[precompute] empty join for {base_tag}/{rl_tag}/{dataset}")
        return out_path

    n_resp = (j.groupby(["prompt_id", "sample_id"])["t"]
                .max().rename("n_resp_tokens").reset_index())
    j = j.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    j["rel_pos"] = j["t"] / j["n_resp_tokens"]

    print(f"[precompute] joined {len(j)} rows  load_time={time.time()-t0:.1f}s, "
          f"computing JS/KL...", flush=True)
    t1 = time.time()
    js, kl_BR, kl_RB = _vector_js_kl(j["ids_B"].values,
                                     j["lg_B"].values,
                                     j["ids_R"].values,
                                     j["lg_R"].values)
    print(f"[precompute] JS done in {time.time()-t1:.1f}s "
          f"({len(j)/(time.time()-t1):.0f} rows/s)")

    j["js"] = js
    j["kl_B_RL"] = kl_BR
    j["kl_RL_B"] = kl_RB
    j["base_tag"] = base_tag
    j["rl_tag"] = rl_tag
    j["dataset"] = dataset

    keep = ["base_tag", "rl_tag", "dataset",
            "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
            "token_id_B", "logp_B", "logp_RL",
            "js", "kl_B_RL", "kl_RL_B"]
    j[keep].to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path) / 1e6
    print(f"[precompute] wrote {out_path}  ({sz:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tag", required=True)
    ap.add_argument("--base_parquet", required=True)
    ap.add_argument("--rl_tag", required=True)
    ap.add_argument("--rl_parquet", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["aime", "ifeval", "ifbench"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    precompute(args.base_tag, args.base_parquet,
               args.rl_tag, args.rl_parquet,
               args.dataset, args.out_dir, args.force)


if __name__ == "__main__":
    main()
