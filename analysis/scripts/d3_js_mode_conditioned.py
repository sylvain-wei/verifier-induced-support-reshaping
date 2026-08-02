#!/usr/bin/env python
"""D.3 — Mode-conditioned divergence.

Each B-anchor rollout response is labelled with `classify_opening_modes`
(DRI / DAI / CSI / Other). For IF datasets, CSI requires the prompt and
constraint metadata; we look those up from the eval-subset jsonl.

For each (B, RL) × dataset × mode × position-bucket cell we report:
    n_tokens, n_samples, mean_js, mean_kl_B_RL, mean_kl_RL_B,
    sample_acc_mean (when sample_acc available).

Plan §4 A.3 Table 5.1 schema is exactly:
    Model | RL_type | Mode | JS@pos1 | JS@first5% | JS@interior | AIME correct
We emit a parquet with the long form and a wide CSV mirroring Table 5.1.

Outputs:
    analysis/data/d3_js_mode_conditioned.parquet
    analysis/tables/d3_table_5_1.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _js_join import (load_js_for_pair, assign_position_bucket,  # noqa: E402
                      POS_BUCKETS)
from classify_opening_modes import classify_mode  # noqa: E402

REPO = os.environ.get("PROJECT_ROOT", ".")
WAVE1 = f"{REPO}/analysis/data/wave1"
EVAL_SUB = f"{REPO}/analysis/data/eval_subsets"

OUT_PARQUET = f"{REPO}/analysis/data/d3_js_mode_conditioned.parquet"
OUT_TABLE = f"{REPO}/analysis/tables/d3_table_5_1.csv"

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


def load_if_metadata(dataset: str) -> Optional[Dict[str, dict]]:
    """For IFEval / IFBench, load the eval subset jsonl and key by prompt_id
    so CSI classification can use inst_ids + prompt."""
    if dataset == "aime":
        return None
    path = os.path.join(EVAL_SUB, f"{dataset}_100.jsonl")
    if not os.path.exists(path):
        return None
    out: Dict[str, dict] = {}
    with open(path) as f:
        for ln in f:
            o = json.loads(ln)
            out[str(o["prompt_id"])] = {
                "prompt_text": o["prompt_text"],
                "instruction_id_list": o.get("instruction_id_list", []),
            }
    return out


def label_responses(base_tag: str, dataset: str,
                    if_meta: Optional[Dict[str, dict]]) -> pd.DataFrame:
    """Returns df with columns prompt_id, sample_id, mode, sample_acc."""
    p = os.path.join(WAVE1, f"{base_tag}__{dataset}__summary.parquet")
    s = pd.read_parquet(p)
    # We need the actual response text, which the summary parquet doesn't have.
    # Reload from per-token parquet's first-row response approximation? No —
    # the per-token parquet doesn't store response either. We reload from the
    # *original* rollout jsonl. Simpler: use sample_id ordinals which match
    # the rollout order in WAVE 1's loader.
    #
    # Trick: classify_mode only needs the response text. We reload by reading
    # the source rollout file and matching (prompt_id, sample_id). The path is
    # encoded in the run_wave1_all.sh queue. We hardcode the four anchor paths:
    rollout = _anchor_rollout(base_tag, dataset)
    responses = _load_responses_for_pids(
        rollout, dataset, set(s["prompt_id"].astype(str)),
        cap_per_prompt=int(s.groupby("prompt_id")["sample_id"].max().max()) + 1,
    )
    rows = []
    for pid, sid, resp in responses:
        meta = if_meta.get(str(pid)) if if_meta else None
        if meta:
            mode = classify_mode(
                resp,
                prompt=meta["prompt_text"],
                inst_ids=meta["instruction_id_list"],
            )
        else:
            mode = classify_mode(resp)
        rows.append({"prompt_id": str(pid), "sample_id": int(sid), "mode": mode})
    out = pd.DataFrame(rows)
    out = out.merge(s[["prompt_id", "sample_id", "sample_acc",
                       "n_resp_tokens", "response_chars"]],
                    on=["prompt_id", "sample_id"], how="left")
    return out


_ROLL_BASE = f"{REPO}/rollout/val_rollout/DAPO_sh_repro"
_ROLL_IFB = f"{REPO}/rollout/val_rollout_ifbench"


def _anchor_rollout(base_tag: str, dataset: str) -> str:
    if base_tag == "B_q3":
        if dataset == "ifbench":
            return f"{_ROLL_IFB}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl"
        return f"{_ROLL_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/0.jsonl"
    if base_tag == "B_q25m":
        return f"{_ROLL_BASE}/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/0.jsonl"
    raise ValueError(base_tag)


AIME_MARKER = "Solve the following math problem step by step"


def _strip_chat_envelope(s: str) -> str:
    """Same canonicalisation as a1b_wave1_logprob.py."""
    s2 = s
    if s2.startswith("system\n"):
        idx = s2.find("\nuser\n")
        if idx >= 0:
            s2 = s2[idx + len("\nuser\n"):]
    elif s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def _aime_pid(uc: str) -> str:
    import hashlib
    return hashlib.md5(uc.encode("utf-8")).hexdigest()[:16]


def _load_responses_for_pids(jsonl: str, dataset: str,
                              keep_pids: set, cap_per_prompt: int):
    """Yield (pid_str, sample_id, response) only for prompt_ids in keep_pids,
    matching the canonical pid scheme used by a1b_wave1_logprob.py."""
    if dataset in ("ifeval", "ifbench"):
        sub = {}
        sp = os.path.join(EVAL_SUB, f"{dataset}_100.jsonl")
        with open(sp) as f:
            for ln in f:
                o = json.loads(ln)
                sub[o["prompt_text"].strip()] = str(o["prompt_id"])

    seen: Dict[str, int] = {}
    out = []
    with open(jsonl) as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except Exception:
                continue
            inp = o.get("input", "")
            is_aime = AIME_MARKER in inp
            if dataset == "aime":
                if not is_aime:
                    continue
                uc = _strip_chat_envelope(inp)
                pid = _aime_pid(uc)
            else:
                if is_aime:
                    continue
                uc = _strip_chat_envelope(inp)
                pid = sub.get(uc)
                if pid is None:
                    continue
            pid = str(pid)
            if pid not in keep_pids:
                continue
            n = seen.get(pid, 0)
            if n >= cap_per_prompt:
                continue
            resp = o.get("output", "") or ""
            seen[pid] = n + 1
            out.append((pid, n, resp))
    return out


def aggregate_modes(df_join: pd.DataFrame, df_modes: pd.DataFrame) -> pd.DataFrame:
    """Per (mode × position-bucket) aggregates."""
    df = df_join.merge(df_modes[["prompt_id", "sample_id", "mode", "sample_acc"]],
                       on=["prompt_id", "sample_id"], how="left")
    df["pos_bucket"] = [
        assign_position_bucket(int(t), int(n)) for t, n in
        zip(df["t"].values, df["n_resp_tokens"].values)
    ]
    g = (df.groupby(["mode", "pos_bucket"])
           .agg(n_tokens=("js", "size"),
                n_samples=("sample_id", "nunique"),
                mean_js=("js", "mean"),
                mean_kl_B_RL=("kl_B_RL", "mean"),
                mean_kl_RL_B=("kl_RL_B", "mean"),
                sample_acc_mean=("sample_acc", "mean"))
           .reset_index())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples_per_pair", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    rows = []
    table_rows = []
    for (base, rl, ds) in PAIRS:
        try:
            joined = load_js_for_pair(base, rl, ds,
                                      max_samples=args.max_samples_per_pair,
                                      keep_topk=False)
        except FileNotFoundError as e:
            print(f"[d3] SKIP ({base},{rl},{ds}): {e}")
            continue
        if joined.empty:
            continue
        if_meta = load_if_metadata(ds)
        modes = label_responses(base, ds, if_meta)
        agg = aggregate_modes(joined, modes)
        agg["base_tag"] = base
        agg["rl_tag"] = rl
        agg["dataset"] = ds
        rows.append(agg)
        # Compact table 5.1 row per mode
        for mode in agg["mode"].dropna().unique():
            sub = agg[agg["mode"] == mode]
            row = {"base_tag": base, "rl_tag": rl, "dataset": ds, "mode": mode}
            for b in ["1", "2-4", "5-16", "17-63", "interior", "last 5%"]:
                rb = sub[sub.pos_bucket == b]
                row[f"js_{b}"] = float(rb["mean_js"].iloc[0]) if not rb.empty else np.nan
            row["sample_acc_mean"] = float(sub["sample_acc_mean"].mean()) if not sub.empty else np.nan
            row["n_samples"] = int(sub["n_samples"].max()) if not sub.empty else 0
            table_rows.append(row)
        print(f"[d3] ({base:7s},{rl:7s},{ds:7s}): "
              f"modes={dict(modes['mode'].value_counts())}")

    if not rows:
        print("[d3] nothing to write; exiting.")
        return
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_parquet(OUT_PARQUET, index=False)
    print(f"[d3] wrote {OUT_PARQUET}")

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUT_TABLE, index=False)
    print(f"[d3] wrote {OUT_TABLE}")
    print(table_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
