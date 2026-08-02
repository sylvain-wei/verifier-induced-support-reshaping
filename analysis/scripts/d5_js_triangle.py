#!/usr/bin/env python
"""D.5 — Cross-policy JS triangle: JS(B,M), JS(B,I), JS(M,I).

All three pairs evaluated on the SAME prefix — B's rollout — so the three
edges are directly comparable per-position. We can then ask whether M's
divergence direction equals I's (same edit, different magnitude) or
different (orthogonal sparse edits).

Inputs:
    WAVE 1: B_q3__{ds}.parquet, B_q25m__{ds}.parquet
    WAVE 1.5: M_q3_on_B / I_q3_on_B / M_q25m_on_B / I_q25m_on_B __ {ds}.parquet

Outputs:
    analysis/data/d5_js_triangle.parquet  — per-token, all three edges
    analysis/tables/d5_triangle_buckets.csv — by position bucket
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kl_utils import topk_union_js  # noqa: E402
from _js_join import assign_position_bucket, DEFAULT_WAVE1, DEFAULT_WAVE15  # noqa: E402

REPO = os.environ.get("PROJECT_ROOT", ".")
OUT_PARQUET = f"{REPO}/analysis/data/d5_js_triangle.parquet"
OUT_TABLE = f"{REPO}/analysis/tables/d5_triangle_buckets.csv"

LINEAGES = [
    # (base_tag, m_tag, i_tag, datasets)
    ("B_q3", "M_q3", "I_q3", ["aime", "ifeval", "ifbench"]),
    ("B_q25m", "M_q25m", "I_q25m", ["aime", "ifeval", "ifbench"]),
]


def load_three_way(base_tag: str, m_tag: str, i_tag: str, ds: str,
                   max_samples: int | None = None,
                   tri_dir: str = f"{REPO}/analysis/data/wave2_js_tri",
                   ) -> pd.DataFrame:
    """Fast path: read the precomputed triangle parquet if present."""
    fast = os.path.join(tri_dir,
                        f"{base_tag}__{m_tag}__{i_tag}__{ds}.parquet")
    if os.path.exists(fast):
        df = pd.read_parquet(fast)
        if max_samples is not None:
            keys = (df[["prompt_id", "sample_id"]].drop_duplicates()
                      .sort_values(["prompt_id", "sample_id"]).head(max_samples))
            df = df.merge(keys, on=["prompt_id", "sample_id"])
        return df

    # Slow path: re-join the three WAVE 1 / WAVE 1.5 topk parquets.
    pB = f"{DEFAULT_WAVE1}/{base_tag}__{ds}.parquet"
    pM = f"{DEFAULT_WAVE15}/{m_tag}_on_B__{ds}.parquet"
    pI = f"{DEFAULT_WAVE15}/{i_tag}_on_B__{ds}.parquet"
    if not (os.path.exists(pB) and os.path.exists(pM) and os.path.exists(pI)):
        missing = [p for p in (pB, pM, pI) if not os.path.exists(p)]
        raise FileNotFoundError("; ".join(missing))

    dfB = pd.read_parquet(pB)
    dfM = pd.read_parquet(pM)
    dfI = pd.read_parquet(pI)
    if max_samples is not None:
        keys = (dfB[["prompt_id", "sample_id"]].drop_duplicates()
                  .sort_values(["prompt_id", "sample_id"]).head(max_samples))
        dfB = dfB.merge(keys, on=["prompt_id", "sample_id"])
        dfM = dfM.merge(keys, on=["prompt_id", "sample_id"])
        dfI = dfI.merge(keys, on=["prompt_id", "sample_id"])

    dfB = dfB.rename(columns={"topk_ids": "ids_B", "topk_logits": "lg_B",
                              "logp": "logp_B", "token_id": "token_id_B"})
    dfM = dfM.rename(columns={"topk_ids": "ids_M", "topk_logits": "lg_M",
                              "logp": "logp_M"})
    dfI = dfI.rename(columns={"topk_ids": "ids_I", "topk_logits": "lg_I",
                              "logp": "logp_I"})
    j = (dfB[["prompt_id", "sample_id", "t", "token_id_B",
              "logp_B", "ids_B", "lg_B"]]
         .merge(dfM[["prompt_id", "sample_id", "t", "logp_M",
                     "ids_M", "lg_M"]],
                on=["prompt_id", "sample_id", "t"], how="inner")
         .merge(dfI[["prompt_id", "sample_id", "t", "logp_I",
                     "ids_I", "lg_I"]],
                on=["prompt_id", "sample_id", "t"], how="inner"))
    if j.empty:
        return j

    n_resp = j.groupby(["prompt_id", "sample_id"])["t"].max().rename("n_resp_tokens").reset_index()
    j = j.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    j["rel_pos"] = j["t"] / j["n_resp_tokens"]

    js_BM = np.empty(len(j), dtype=np.float32)
    js_BI = np.empty(len(j), dtype=np.float32)
    js_MI = np.empty(len(j), dtype=np.float32)
    for k, (ib, lb, im, lm, ii, li) in enumerate(zip(
            j["ids_B"].values, j["lg_B"].values,
            j["ids_M"].values, j["lg_M"].values,
            j["ids_I"].values, j["lg_I"].values)):
        ib = np.asarray(ib, dtype=np.int64)
        lb = np.asarray(lb, dtype=np.float32)
        im = np.asarray(im, dtype=np.int64)
        lm = np.asarray(lm, dtype=np.float32)
        ii = np.asarray(ii, dtype=np.int64)
        li = np.asarray(li, dtype=np.float32)
        js_BM[k] = topk_union_js((ib, lb), (im, lm))
        js_BI[k] = topk_union_js((ib, lb), (ii, li))
        js_MI[k] = topk_union_js((im, lm), (ii, li))

    j["js_BM"] = js_BM
    j["js_BI"] = js_BI
    j["js_MI"] = js_MI
    j = j.drop(columns=["ids_B", "lg_B", "ids_M", "lg_M", "ids_I", "lg_I"])
    j["base_tag"] = base_tag
    j["m_tag"] = m_tag
    j["i_tag"] = i_tag
    j["dataset"] = ds
    return j


def aggregate_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_bucket"] = [
        assign_position_bucket(int(t), int(n)) for t, n in
        zip(df["t"].values, df["n_resp_tokens"].values)
    ]
    g = (df.groupby(["base_tag", "m_tag", "i_tag", "dataset", "pos_bucket"])
           .agg(n_tokens=("js_BM", "size"),
                mean_js_BM=("js_BM", "mean"),
                mean_js_BI=("js_BI", "mean"),
                mean_js_MI=("js_MI", "mean"),
                median_js_BM=("js_BM", "median"),
                median_js_BI=("js_BI", "median"),
                median_js_MI=("js_MI", "median"))
           .reset_index())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples_per_pair", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    parts = []
    for (b, m, i, datasets) in LINEAGES:
        for ds in datasets:
            try:
                df = load_three_way(b, m, i, ds,
                                    max_samples=args.max_samples_per_pair)
            except FileNotFoundError as e:
                print(f"[d5] SKIP ({b},{ds}): {e}")
                continue
            if df.empty:
                continue
            parts.append(df)
            print(f"[d5] ({b},{ds}): n_tok={len(df)}  "
                  f"mean JS(B,M)={df['js_BM'].mean():.4f}  "
                  f"JS(B,I)={df['js_BI'].mean():.4f}  "
                  f"JS(M,I)={df['js_MI'].mean():.4f}")

    if not parts:
        print("[d5] nothing to do")
        return
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT_PARQUET, index=False)
    print(f"[d5] wrote {OUT_PARQUET}")

    agg = aggregate_by_bucket(full)
    agg.to_csv(OUT_TABLE, index=False)
    print(f"[d5] wrote {OUT_TABLE}")
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
