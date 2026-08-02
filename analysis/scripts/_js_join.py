#!/usr/bin/env python
"""Shared loader: pair WAVE 1 (B on B-rollout) with WAVE 1.5 (RL on B-rollout)
parquets and compute per-token same-prefix JS.

The two parquets on each side store top-K logits at *every* response position
of B's own AIME / IFEval / IFBench rollout. Joining on
`(prompt_id, sample_id, t)` gives matched pairs of `(B_topk, RL_topk)` from
which `topk_union_js` produces the JS_t value.

Public API
----------
    load_js_for_pair(base_tag, rl_tag, dataset,
                     wave1_dir, wave1_5_dir,
                     keep_topk=False, max_samples=None) -> pd.DataFrame
        Columns: prompt_id, sample_id, t, n_resp_tokens (from B summary),
                 logp_B, logp_RL, js, kl_B_RL, kl_RL_B,
                 token_id_B (the B-rollout's actual token at t).
        If keep_topk=True, also includes topk_ids_B/topk_logits_B/topk_ids_RL/
        topk_logits_RL for downstream A.4 use.

    iter_pair_groups(...) -> generator yielding (prompt_id, sample_id, sub_df)
        Avoids loading the full join in memory if the parquets are huge.

Note on accuracy
----------------
The RL parquet was produced by `a1b_wave1_logprob.py` driven with the RL
model's path and the B-anchor rollout JSONL. So WAVE 1.5's `topk_ids` and
`topk_logits` for `M_q3_on_B` are M_q3's distribution evaluated at every
position of B_q3's response. Pairing with WAVE 1's `B_q3` parquet (which is
B_q3's distribution on the SAME response) gives the true same-prefix
top-K pair needed for `topk_union_js`.
"""
from __future__ import annotations

import os
import sys
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kl_utils import topk_union_js, topk_union_kl  # noqa: E402

REPO_ROOT = os.environ.get("PROJECT_ROOT", ".")
DEFAULT_WAVE1 = os.path.join(REPO_ROOT, "analysis/data/wave1")
DEFAULT_WAVE15 = os.path.join(REPO_ROOT, "analysis/data/wave1_5")
DEFAULT_WAVE2_JS = os.path.join(REPO_ROOT, "analysis/data/wave2_js")


# Map a "lineage" tag (B_q3, M_q3, I_q3, B_q25m, M_q25m, I_q25m) to which
# parquet directory and base name to read. The B side is always the
# self-rollout in WAVE 1; the RL side is the cross-forward in WAVE 1.5.
def _path_for(tag: str, dataset: str, *, wave1_dir: str, wave1_5_dir: str,
              kind: str) -> str:
    """kind: 'self' (WAVE 1) or 'on_B' (WAVE 1.5)."""
    if kind == "self":
        return os.path.join(wave1_dir, f"{tag}__{dataset}.parquet")
    if kind == "on_B":
        return os.path.join(wave1_5_dir, f"{tag}_on_B__{dataset}.parquet")
    raise ValueError(f"unknown kind={kind!r}")


def load_js_for_pair(
    base_tag: str,
    rl_tag: str,
    dataset: str,
    *,
    wave1_dir: str = DEFAULT_WAVE1,
    wave1_5_dir: str = DEFAULT_WAVE15,
    wave2_js_dir: str = DEFAULT_WAVE2_JS,
    keep_topk: bool = False,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Load the matched per-token JS dataframe for one (B, RL) pair on one
    dataset.

    Fast path
    ---------
    If `wave2_js_dir/{base}__{rl}__{ds}.parquet` exists (produced by
    `precompute_js.py`), load it directly — that file already has the
    scalar JS / KL columns and is ~50× smaller than the topk parquets.
    `keep_topk=True` falls back to the slow path because the precompute
    step drops the topk columns.

    Slow path (only used when keep_topk=True or precompute is missing)
    -----------------------------------------------------------------
    Re-join WAVE 1 + WAVE 1.5 topk parquets and compute JS row-wise.
    """
    fast = os.path.join(wave2_js_dir, f"{base_tag}__{rl_tag}__{dataset}.parquet")
    if os.path.exists(fast) and not keep_topk:
        df = pd.read_parquet(fast)
        if max_samples is not None:
            keys = (df[["prompt_id", "sample_id"]].drop_duplicates()
                      .sort_values(["prompt_id", "sample_id"]).head(max_samples))
            df = df.merge(keys, on=["prompt_id", "sample_id"])
        return df

    # Slow path (also used when caller needs topk arrays for d4)
    pB = _path_for(base_tag, dataset, wave1_dir=wave1_dir,
                   wave1_5_dir=wave1_5_dir, kind="self")
    pR = _path_for(rl_tag, dataset, wave1_dir=wave1_dir,
                   wave1_5_dir=wave1_5_dir, kind="on_B")
    if not os.path.exists(pB):
        raise FileNotFoundError(pB)
    if not os.path.exists(pR):
        raise FileNotFoundError(pR)

    dfB = pd.read_parquet(pB)
    dfR = pd.read_parquet(pR)

    if max_samples is not None:
        keys = (dfB[["prompt_id", "sample_id"]]
                .drop_duplicates()
                .sort_values(["prompt_id", "sample_id"])
                .head(max_samples))
        merge_keys = pd.merge(dfB, keys, on=["prompt_id", "sample_id"])
        dfB = merge_keys
        dfR = pd.merge(dfR, keys, on=["prompt_id", "sample_id"])

    # Inner join on (prompt_id, sample_id, t)
    dfB = dfB.rename(columns={
        "logp": "logp_B",
        "topk_ids": "topk_ids_B",
        "topk_logits": "topk_logits_B",
        "token_id": "token_id_B",
    })[["prompt_id", "sample_id", "t",
        "token_id_B", "logp_B", "topk_ids_B", "topk_logits_B"]]
    dfR = dfR.rename(columns={
        "logp": "logp_RL",
        "topk_ids": "topk_ids_RL",
        "topk_logits": "topk_logits_RL",
    })[["prompt_id", "sample_id", "t",
        "logp_RL", "topk_ids_RL", "topk_logits_RL"]]

    df = dfB.merge(dfR, on=["prompt_id", "sample_id", "t"], how="inner")
    if df.empty:
        return df

    # Compute JS, KL row-wise.
    # The arrays come back from parquet as numpy arrays (for stored list cols
    # pandas materialises object arrays). Convert per-row.
    js_list = np.empty(len(df), dtype=np.float32)
    kl_BR = np.empty(len(df), dtype=np.float32)
    kl_RB = np.empty(len(df), dtype=np.float32)
    for i, (ids_b, lg_b, ids_r, lg_r) in enumerate(zip(
            df.topk_ids_B.values, df.topk_logits_B.values,
            df.topk_ids_RL.values, df.topk_logits_RL.values)):
        ids_b = np.asarray(ids_b, dtype=np.int64)
        lg_b = np.asarray(lg_b, dtype=np.float32)
        ids_r = np.asarray(ids_r, dtype=np.int64)
        lg_r = np.asarray(lg_r, dtype=np.float32)
        kl = topk_union_kl((ids_b, lg_b), (ids_r, lg_r))
        js = topk_union_js((ids_b, lg_b), (ids_r, lg_r))
        js_list[i] = js
        kl_BR[i] = kl["kl_pq"]
        kl_RB[i] = kl["kl_qp"]
    df["js"] = js_list
    df["kl_B_RL"] = kl_BR
    df["kl_RL_B"] = kl_RB

    # Add per-row n_resp_tokens for relative-position bucketing.
    n_resp = (df.groupby(["prompt_id", "sample_id"])["t"]
                .max().rename("n_resp_tokens").reset_index())
    df = df.merge(n_resp, on=["prompt_id", "sample_id"], how="left")
    df["rel_pos"] = df["t"] / df["n_resp_tokens"]

    if not keep_topk:
        df = df.drop(columns=["topk_ids_B", "topk_logits_B",
                              "topk_ids_RL", "topk_logits_RL"])
    return df


def iter_pair_groups(
    base_tag: str,
    rl_tag: str,
    dataset: str,
    *,
    wave1_dir: str = DEFAULT_WAVE1,
    wave1_5_dir: str = DEFAULT_WAVE15,
    keep_topk: bool = False,
) -> Iterator[Tuple[str, int, pd.DataFrame]]:
    """Stream JS rows grouped by (prompt_id, sample_id). For Block A scripts
    that need to do per-rollout aggregations and won't tolerate the full join
    in memory."""
    df = load_js_for_pair(base_tag, rl_tag, dataset,
                          wave1_dir=wave1_dir, wave1_5_dir=wave1_5_dir,
                          keep_topk=keep_topk)
    for (pid, sid), sub in df.groupby(["prompt_id", "sample_id"]):
        yield pid, sid, sub.reset_index(drop=True)


# Position-bucket helper (used by d2 / d3 / d5).
def assign_position_bucket(t: int, n_resp: int) -> str:
    """Absolute position bucket {1, 2-4, 5-16, 17-63, interior, last 5%}.

    'interior' = position > 63 and not in the last 5%.
    'last 5%' = position > 0.95 * n_resp.
    """
    if t == 1:
        return "1"
    if 2 <= t <= 4:
        return "2-4"
    if 5 <= t <= 16:
        return "5-16"
    if 17 <= t <= 63:
        return "17-63"
    if t > 0.95 * n_resp:
        return "last 5%"
    return "interior"


POS_BUCKETS = ["1", "2-4", "5-16", "17-63", "interior", "last 5%"]


if __name__ == "__main__":
    # Smoke test on first 4 samples of (B_q3, M_q3) on aime.
    df = load_js_for_pair("B_q3", "M_q3", "aime", max_samples=4, keep_topk=False)
    print("smoke join shape:", df.shape)
    print("js stats:", df["js"].describe()[["mean", "max", "50%"]].to_dict())
    print(df.head(8).to_string())
