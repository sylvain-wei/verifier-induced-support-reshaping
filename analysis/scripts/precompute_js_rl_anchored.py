#!/usr/bin/env python
"""Precompute per-token JS / KL for one RL-anchored (rl_tag, dataset) pair.

Difference from `precompute_js.py`:
- Anchor trajectory is the RL model's OWN rollout (M_q3-rollout / I_q3-rollout
  / same for q25m), not B's. So both distributions are evaluated at the SAME
  prefix = the RL rollout response.
- "RL distribution side" = WAVE 1's `wave1/{rl_tag}__{ds}.parquet`
  (RL forwarded over its own rollout; already exists).
- "Base distribution side" = WAVE 1.5 RL-anchored
  `wave1_5_rl_anchored/B_{lineage}_on_{rl_tag}__{ds}.parquet`
  (base forwarded over the RL rollout; produced by run_wave1_5_rl_anchored.sh).

Output sits in `analysis/data/wave2_js_rl_anchored/`, schema identical to
`wave2_js/` (same columns), so e1_outcome_divergence_v2 can re-use the same
loader code.

Usage:
    python precompute_js_rl_anchored.py --base B_q3 --rl M_q3 --dataset aime
    # -> analysis/data/wave2_js_rl_anchored/B_q3__M_q3__aime.parquet
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
WAVE15_RLA = f"{REPO}/analysis/data/wave1_5_rl_anchored"
OUT_DIR = f"{REPO}/analysis/data/wave2_js_rl_anchored"


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
            print(f"    [precompute_rla] {i}/{n}")
    return js, kl_BR, kl_RB


def precompute(base: str, rl: str, ds: str,
               force: bool = False) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{base}__{rl}__{ds}.parquet")
    if os.path.exists(out_path) and not force:
        print(f"[rla] {out_path} exists; skip")
        return out_path

    # RL side = WAVE 1 (RL model on RL's own rollout)
    pR = f"{WAVE1}/{rl}__{ds}.parquet"
    # Base side = WAVE 1.5 RL-anchored (base on RL rollout)
    pB = f"{WAVE15_RLA}/{base}_on_{rl}__{ds}.parquet"
    if not os.path.exists(pR):
        raise FileNotFoundError(pR)
    if not os.path.exists(pB):
        raise FileNotFoundError(pB)

    t0 = time.time()
    dfB = pd.read_parquet(pB).rename(columns={
        "logp": "logp_B", "topk_ids": "ids_B", "topk_logits": "lg_B",
        "token_id": "token_id_B",
    })[["prompt_id", "sample_id", "t", "token_id_B", "logp_B", "ids_B", "lg_B"]]
    dfR = pd.read_parquet(pR).rename(columns={
        "logp": "logp_RL", "topk_ids": "ids_R", "topk_logits": "lg_R",
    })[["prompt_id", "sample_id", "t", "logp_RL", "ids_R", "lg_R"]]

    j = dfB.merge(dfR, on=["prompt_id", "sample_id", "t"], how="inner")
    if j.empty:
        print("[rla] empty join")
        return out_path
    n_resp = (j.groupby(["prompt_id", "sample_id"])["t"]
                .max().rename("n_resp_tokens").reset_index())
    j = j.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    j["rel_pos"] = j["t"] / j["n_resp_tokens"]

    print(f"[rla] joined {len(j)} rows in {time.time()-t0:.1f}s; computing JS/KL …")
    t1 = time.time()
    js, kl_BR, kl_RB = _vector_js_kl(j.ids_B.values, j.lg_B.values,
                                     j.ids_R.values, j.lg_R.values)
    print(f"[rla] JS done in {time.time()-t1:.1f}s "
          f"({len(j)/(time.time()-t1):.0f} rows/s)")
    j["js"] = js
    j["kl_B_RL"] = kl_BR
    j["kl_RL_B"] = kl_RB
    j["base_tag"] = base
    j["rl_tag"] = rl
    j["dataset"] = ds

    keep = ["base_tag", "rl_tag", "dataset",
            "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
            "token_id_B", "logp_B", "logp_RL", "js", "kl_B_RL", "kl_RL_B"]
    j[keep].to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path) / 1e6
    print(f"[rla] wrote {out_path}  ({sz:.1f} MB)")
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
