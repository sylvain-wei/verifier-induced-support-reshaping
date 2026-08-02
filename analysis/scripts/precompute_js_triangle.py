#!/usr/bin/env python
"""Precompute the three same-prefix JS edges (B-M, B-I, M-I) for one
(lineage, dataset) tuple. Output schema is one row per response token with
columns js_BM, js_BI, js_MI plus prompt_id/sample_id/t/n_resp/rel_pos.

Lets d5_js_triangle.py use a fast pandas-only aggregation path.

Usage:
    python precompute_js_triangle.py --base B_q3 --m M_q3 --i I_q3 --dataset aime
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kl_utils import topk_union_js  # noqa: E402

REPO = os.environ.get("PROJECT_ROOT", ".")
WAVE1 = f"{REPO}/analysis/data/wave1"
WAVE15 = f"{REPO}/analysis/data/wave1_5"
OUT_DIR = f"{REPO}/analysis/data/wave2_js_tri"


def precompute(base: str, m: str, i: str, ds: str,
               force: bool = False) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{base}__{m}__{i}__{ds}.parquet")
    if os.path.exists(out_path) and not force:
        print(f"[tri] {out_path} exists; skip")
        return out_path

    pB = f"{WAVE1}/{base}__{ds}.parquet"
    pM = f"{WAVE15}/{m}_on_B__{ds}.parquet"
    pI = f"{WAVE15}/{i}_on_B__{ds}.parquet"
    for p in (pB, pM, pI):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    t0 = time.time()
    dfB = pd.read_parquet(pB)[["prompt_id", "sample_id", "t",
                                "token_id", "logp", "topk_ids", "topk_logits"]]
    dfM = pd.read_parquet(pM)[["prompt_id", "sample_id", "t",
                                "logp", "topk_ids", "topk_logits"]]
    dfI = pd.read_parquet(pI)[["prompt_id", "sample_id", "t",
                                "logp", "topk_ids", "topk_logits"]]
    print(f"[tri] loaded 3 parquets in {time.time()-t0:.1f}s")

    dfB = dfB.rename(columns={"token_id": "token_id_B", "logp": "logp_B",
                              "topk_ids": "ids_B", "topk_logits": "lg_B"})
    dfM = dfM.rename(columns={"logp": "logp_M", "topk_ids": "ids_M",
                              "topk_logits": "lg_M"})
    dfI = dfI.rename(columns={"logp": "logp_I", "topk_ids": "ids_I",
                              "topk_logits": "lg_I"})
    j = (dfB.merge(dfM, on=["prompt_id", "sample_id", "t"], how="inner")
            .merge(dfI, on=["prompt_id", "sample_id", "t"], how="inner"))
    if j.empty:
        print("[tri] empty join")
        return out_path

    n_resp = j.groupby(["prompt_id", "sample_id"])["t"].max().rename("n_resp_tokens").reset_index()
    j = j.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    j["rel_pos"] = j["t"] / j["n_resp_tokens"]

    n = len(j)
    js_BM = np.empty(n, dtype=np.float32)
    js_BI = np.empty(n, dtype=np.float32)
    js_MI = np.empty(n, dtype=np.float32)
    t1 = time.time()
    for k in range(n):
        ib = np.asarray(j.ids_B.values[k], dtype=np.int64)
        lb = np.asarray(j.lg_B.values[k], dtype=np.float32)
        im = np.asarray(j.ids_M.values[k], dtype=np.int64)
        lm = np.asarray(j.lg_M.values[k], dtype=np.float32)
        ii = np.asarray(j.ids_I.values[k], dtype=np.int64)
        li = np.asarray(j.lg_I.values[k], dtype=np.float32)
        js_BM[k] = topk_union_js((ib, lb), (im, lm))
        js_BI[k] = topk_union_js((ib, lb), (ii, li))
        js_MI[k] = topk_union_js((im, lm), (ii, li))
        if k % 200000 == 0 and k > 0:
            print(f"  [tri] {k}/{n} ({k/n*100:.1f}%)  rate={k/(time.time()-t1):.0f}/s")
    print(f"[tri] computed JS_3 in {time.time()-t1:.1f}s")

    j["js_BM"] = js_BM
    j["js_BI"] = js_BI
    j["js_MI"] = js_MI
    j["base_tag"] = base
    j["m_tag"] = m
    j["i_tag"] = i
    j["dataset"] = ds

    keep = ["base_tag", "m_tag", "i_tag", "dataset",
            "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
            "token_id_B", "logp_B", "logp_M", "logp_I",
            "js_BM", "js_BI", "js_MI"]
    j[keep].to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path) / 1e6
    print(f"[tri] wrote {out_path}  ({sz:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--m", required=True)
    ap.add_argument("--i", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["aime", "ifeval", "ifbench"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    precompute(args.base, args.m, args.i, args.dataset, force=args.force)


if __name__ == "__main__":
    main()
