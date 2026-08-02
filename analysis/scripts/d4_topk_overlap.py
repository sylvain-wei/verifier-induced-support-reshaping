#!/usr/bin/env python
"""D.4 — Top-K overlap & rank reordering on high-JS positions.

For each (B, RL) × dataset, take the top-decile-JS positions and report:
    - top-5 / top-10 Jaccard between B and RL top-K sets
    - rank of RL's top-1 token under B's distribution
    - B's probability assigned to RL's top-1

Implementation (fast path)
--------------------------
1. Load the precomputed JS parquet (`wave2_js/{base}__{rl}__{ds}.parquet`)
   to identify the top-10% JS rows by (prompt_id, sample_id, t).
2. Read JUST those rows from the WAVE 1 + WAVE 1.5 topk parquets via
   a pyarrow filter (or via a pandas merge), so we only materialise topk
   arrays for ~10% of positions.
3. Compute per-row Jaccard / rank / prob metrics on the high-JS subset.

This avoids the original keep_topk=True full-load, which was 20+ min for
the IFBench q3 pairs (2.3M rows × 64-element arrays each).

Plan §4 A.4 success criterion: high-JS positions still have top-K Jaccard
≥ 0.6 AND RL top-1 in B top-3 ≥ 60% → RL doesn't invent new shortcuts,
only re-ranks base candidates.

Outputs:
    analysis/data/d4_topk_overlap.parquet  — per high-JS row
    analysis/tables/d4_summary.csv         — per (base, rl, ds, scope) row
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.environ.get("PROJECT_ROOT", ".")
WAVE1 = f"{REPO}/analysis/data/wave1"
WAVE15 = f"{REPO}/analysis/data/wave1_5"
WAVE2_JS = f"{REPO}/analysis/data/wave2_js"

OUT_PARQUET = f"{REPO}/analysis/data/d4_topk_overlap.parquet"
OUT_TABLE = f"{REPO}/analysis/tables/d4_summary.csv"

PAIRS = [
    ("B_q3", "M_q3", "aime"),
    ("B_q3", "M_q3", "ifeval"),
    ("B_q3", "M_q3", "ifbench"),
    ("B_q3", "I_q3", "aime"),
    ("B_q3", "I_q3", "ifeval"),
    ("B_q3", "I_q3", "ifbench"),
    ("B_q25m", "M_q25m", "aime"),
    ("B_q25m", "M_q25m", "ifeval"),
    ("B_q25m", "I_q25m", "aime"),
    ("B_q25m", "I_q25m", "ifeval"),
]


def _softmax_within_topk(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    p = np.exp(z)
    return p / p.sum()


def per_row_metrics(ids_b: np.ndarray, lg_b: np.ndarray,
                    ids_r: np.ndarray, lg_r: np.ndarray) -> dict:
    set_b = set(ids_b.tolist())
    set_r = set(ids_r.tolist())
    set_b5 = set(ids_b[:5].tolist())
    set_r5 = set(ids_r[:5].tolist())
    set_b10 = set(ids_b[:10].tolist())
    set_r10 = set(ids_r[:10].tolist())

    def jacc(a, b):
        u = len(a | b)
        return float(len(a & b) / u) if u else 0.0

    rl_top1 = int(ids_r[0])
    if rl_top1 in set_b:
        rank_in_b = int(np.where(ids_b == rl_top1)[0][0]) + 1
        pB = _softmax_within_topk(lg_b)
        pB_on_rl_top1 = float(pB[np.where(ids_b == rl_top1)[0][0]])
    else:
        rank_in_b = len(ids_b) + 1
        pB_on_rl_top1 = 0.0

    return {
        "jaccard_top5": jacc(set_b5, set_r5),
        "jaccard_top10": jacc(set_b10, set_r10),
        "jaccard_topK": jacc(set_b, set_r),
        "rl_top1": rl_top1,
        "rank_rl_top1_in_B": rank_in_b,
        "B_prob_on_rl_top1": pB_on_rl_top1,
        "rl_top1_in_B_top3": int(rank_in_b <= 3),
        "rl_top1_in_B_topK": int(rl_top1 in set_b),
    }


def process_pair(base: str, rl: str, ds: str,
                 max_samples: int | None) -> tuple[pd.DataFrame, list[dict]]:
    # Step 1: read precomputed JS to identify high-JS rows.
    pjs = f"{WAVE2_JS}/{base}__{rl}__{ds}.parquet"
    if not os.path.exists(pjs):
        raise FileNotFoundError(pjs)
    js_df = pd.read_parquet(pjs, columns=[
        "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
        "js", "kl_B_RL", "kl_RL_B",
    ])
    if max_samples is not None:
        keys = (js_df[["prompt_id", "sample_id"]].drop_duplicates()
                  .sort_values(["prompt_id", "sample_id"]).head(max_samples))
        js_df = js_df.merge(keys, on=["prompt_id", "sample_id"])

    js_thresh = float(np.percentile(js_df["js"].values, 90))
    high_keys = js_df[js_df["js"] >= js_thresh][[
        "prompt_id", "sample_id", "t", "n_resp_tokens", "rel_pos",
        "js", "kl_B_RL", "kl_RL_B"]].copy()
    if high_keys.empty:
        return high_keys, []

    # Step 2: load WAVE 1 B parquet (only the topk + ids cols), then merge on keys.
    pB = f"{WAVE1}/{base}__{ds}.parquet"
    pR = f"{WAVE15}/{rl}_on_B__{ds}.parquet"
    t0 = time.time()
    dfB = pd.read_parquet(pB, columns=[
        "prompt_id", "sample_id", "t", "topk_ids", "topk_logits"
    ])
    dfR = pd.read_parquet(pR, columns=[
        "prompt_id", "sample_id", "t", "topk_ids", "topk_logits"
    ])
    print(f"[d4] ({base},{rl},{ds}) loaded topk parquets in {time.time()-t0:.1f}s")

    dfB = dfB.rename(columns={"topk_ids": "ids_B", "topk_logits": "lg_B"})
    dfR = dfR.rename(columns={"topk_ids": "ids_R", "topk_logits": "lg_R"})

    merged = (high_keys
              .merge(dfB, on=["prompt_id", "sample_id", "t"], how="left")
              .merge(dfR, on=["prompt_id", "sample_id", "t"], how="left"))
    # Drop rows where merge failed (shouldn't happen if WAVE 1 + 1.5 are
    # consistent with precompute output).
    merged = merged.dropna(subset=["ids_B", "ids_R"]).reset_index(drop=True)

    # Step 3: per-row metrics
    rows = []
    for r in merged.itertuples():
        ib = np.asarray(r.ids_B, dtype=np.int64)
        lb = np.asarray(r.lg_B, dtype=np.float32)
        ir = np.asarray(r.ids_R, dtype=np.int64)
        lr = np.asarray(r.lg_R, dtype=np.float32)
        m = per_row_metrics(ib, lb, ir, lr)
        m.update({
            "base_tag": base, "rl_tag": rl, "dataset": ds,
            "prompt_id": r.prompt_id, "sample_id": r.sample_id,
            "t": r.t, "rel_pos": r.rel_pos,
            "n_resp_tokens": r.n_resp_tokens,
            "js": float(r.js), "kl_B_RL": float(r.kl_B_RL),
            "kl_RL_B": float(r.kl_RL_B),
        })
        rows.append(m)
    out = pd.DataFrame(rows)

    summaries = []
    for scope_name, sub in [("high_js_top10pct", out),
                            ("high_js_pos1", out[out["t"] == 1])]:
        if sub.empty:
            continue
        summaries.append({
            "base_tag": base, "rl_tag": rl, "dataset": ds,
            "scope": scope_name, "n_rows": len(sub),
            "mean_jaccard_top5": float(sub.jaccard_top5.mean()),
            "mean_jaccard_top10": float(sub.jaccard_top10.mean()),
            "mean_jaccard_topK": float(sub.jaccard_topK.mean()),
            "median_rank_rl_top1_in_B": float(sub.rank_rl_top1_in_B.median()),
            "frac_rl_top1_in_B_top3": float(sub.rl_top1_in_B_top3.mean()),
            "frac_rl_top1_in_B_topK": float(sub.rl_top1_in_B_topK.mean()),
            "mean_B_prob_on_rl_top1": float(sub.B_prob_on_rl_top1.mean()),
            "js_thresh": js_thresh,
        })
    return out, summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples_per_pair", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    all_rows = []
    summary_rows = []
    t_total = time.time()
    for (base, rl, ds) in PAIRS:
        t0 = time.time()
        try:
            df, summ = process_pair(base, rl, ds, args.max_samples_per_pair)
        except FileNotFoundError as e:
            print(f"[d4] SKIP ({base},{rl},{ds}): {e}")
            continue
        if df.empty:
            continue
        all_rows.append(df)
        summary_rows.extend(summ)
        s = next((s for s in summ if s["scope"] == "high_js_top10pct"), None)
        if s:
            print(f"[d4] ({base:7s},{rl:7s},{ds:7s}): "
                  f"n_high={s['n_rows']:>7}  Jacc@10={s['mean_jaccard_top10']:.3f}  "
                  f"RL_top1∈B_top3={s['frac_rl_top1_in_B_top3']:.3f}  "
                  f"med_rank={s['median_rank_rl_top1_in_B']:.1f}  "
                  f"({time.time()-t0:.1f}s)")

    if not all_rows:
        print("[d4] nothing to do")
        return
    pd.concat(all_rows, ignore_index=True).to_parquet(OUT_PARQUET, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_TABLE, index=False)
    print(f"[d4] wrote {OUT_PARQUET} and {OUT_TABLE} "
          f"(total {time.time()-t_total:.1f}s)")


if __name__ == "__main__":
    main()
