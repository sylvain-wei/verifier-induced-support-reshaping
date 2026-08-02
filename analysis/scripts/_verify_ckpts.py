#!/usr/bin/env python
"""WAVE 0.4 — verify every ckpt referenced by plan_paper.md.

For each ckpt path in CKPTS:
  1. exists check (config.json + tokenizer.json/merges.txt or tokenizer.model)
  2. AutoTokenizer.from_pretrained() loads
  3. AutoModelForCausalLM.from_pretrained() loads in bf16 (single-GPU only,
     so this works inside an 8-way shard)
  4. Tiny forward on a 32-token random tensor returns finite logits

Outputs:
  - analysis/data/ckpt_status.csv  : columns ckpt_role, path, exists, tokenizer_ok,
                                     model_ok, forward_ok, vocab_size, dtype, note,
                                     latency_s, gpu_id
  - prints a per-row PASS / FAIL line and a final summary.

8-GPU sharding
--------------
The full ckpt list is partitioned by `--n_shards` (default = number of CUDA
devices) and `--shard_id` (default = 0). Run 8 instances in parallel:

    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i \
        python _verify_ckpts.py --shard_id $i --n_shards 8 \
          --out_csv analysis/data/ckpt_status_shard${i}.csv &
    done
    wait
    # Merge shards manually (cat headers/data) or run --merge mode.

For convenience, `--merge` reads `ckpt_status_shard*.csv` from the data dir
and concatenates them into `ckpt_status.csv`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import List, Tuple

import pandas as pd


ROOT = os.environ.get("PROJECT_ROOT", ".")
CKPT_BASE = f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro"
OUT_DIR   = f"{ROOT}/analysis/data"
DEFAULT_OUT = f"{OUT_DIR}/ckpt_status.csv"


# ---------------------------------------------------------------------------
# Canonical ckpt list — keep in sync with plan_paper.md §4.
# ---------------------------------------------------------------------------
def _ckpt_list() -> List[Tuple[str, str]]:
    """Returns list of (ckpt_role, abs_path)."""
    out = []

    # Base models (raw HF dirs, not verl checkpoints).
    out.append(("B_q3",   f"{ROOT}/models/Qwen3-8B-Base"))
    out.append(("B_q25m", f"{ROOT}/models/Qwen2.5-Math-7B"))

    # Qwen3-8B-Base lineage — primary RL endpoints.
    out.append(("M_q3@220", f"{CKPT_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/"
                            f"global_step_220/actor/huggingface"))
    out.append(("I_q3@100", f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/"
                            f"global_step_100/actor/huggingface"))
    # Qwen3-8B-Base lineage — extra step trajectory ckpts (b2r1 sparse).
    for s in [20, 40, 60, 80, 100]:
        out.append((f"I_q3@{s}",
                    f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/"
                    f"global_step_{s}/actor/huggingface"))
    # Qwen3-8B-Base lineage — clip_high trajectory (14 ckpts spanning 20–260).
    for s in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]:
        out.append((f"I_q3_clip@{s}",
                    f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_clip_high_0.2_local_H20/"
                    f"global_step_{s}/actor/huggingface"))
    # Qwen3-8B-Base lineage — b1r1 trajectory (every 20 steps).
    for s in list(range(20, 521, 20)):
        out.append((f"M_q3@{s}",
                    f"{CKPT_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/"
                    f"global_step_{s}/actor/huggingface"))
    # b4r1 Qwen3 — only step 20 saved.
    out.append(("b4r1_q3@20",
                f"{CKPT_BASE}/b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20/"
                f"global_step_20/actor/huggingface"))

    # Qwen2.5-Math-7B lineage.
    # Plan says b1r1 step 470 / b2r1 step 390; on disk we have step 460/380.
    # Verify both endpoints + the broader trajectory.
    out.append(("M_q25m@460",
                f"{CKPT_BASE}/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/"
                f"global_step_460/actor/huggingface"))
    out.append(("I_q25m@380",
                f"{CKPT_BASE}/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/"
                f"global_step_380/actor/huggingface"))
    for s in list(range(20, 501, 20)):
        out.append((f"M_q25m@{s}",
                    f"{CKPT_BASE}/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/"
                    f"global_step_{s}/actor/huggingface"))
    for s in list(range(20, 381, 20)):
        out.append((f"I_q25m@{s}",
                    f"{CKPT_BASE}/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/"
                    f"global_step_{s}/actor/huggingface"))

    # Dedup while preserving order.
    seen = set()
    deduped = []
    for r, p in out:
        key = (r, p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((r, p))
    return deduped


# ---------------------------------------------------------------------------
# Per-ckpt verification.
# ---------------------------------------------------------------------------
def _check_files(path: str) -> Tuple[bool, str]:
    if not os.path.isdir(path):
        return False, "directory missing"
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False, "config.json missing"
    has_tok = any(
        os.path.isfile(os.path.join(path, fn))
        for fn in ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"]
    )
    if not has_tok:
        return False, "tokenizer files missing"
    return True, "files ok"


def verify_one(role: str, path: str, do_forward: bool = True) -> dict:
    t0 = time.time()
    row = {
        "ckpt_role": role,
        "path": path,
        "exists": False,
        "tokenizer_ok": False,
        "model_ok": False,
        "forward_ok": False,
        "vocab_size": -1,
        "dtype": "",
        "note": "",
        "latency_s": 0.0,
    }
    ok, why = _check_files(path)
    row["exists"] = ok
    row["note"] = why
    if not ok:
        row["latency_s"] = round(time.time() - t0, 2)
        return row

    if not do_forward:
        # quick mode — just file-existence check
        return row

    # Heavy: actually load tokenizer + model + run a forward.
    try:
        # Defer torch import so file-only check stays fast.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        row["note"] = f"torch/transformers import failed: {e}"
        row["latency_s"] = round(time.time() - t0, 2)
        return row

    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        row["tokenizer_ok"] = True
        row["vocab_size"] = int(getattr(tok, "vocab_size", -1))
    except Exception as e:
        row["note"] = f"tokenizer load failed: {type(e).__name__}: {e}"
        row["latency_s"] = round(time.time() - t0, 2)
        return row

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # whatever CUDA_VISIBLE_DEVICES exposes as cuda:0
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        model.eval()
        row["model_ok"] = True
        row["dtype"] = str(next(model.parameters()).dtype).replace("torch.", "")
    except Exception as e:
        row["note"] = f"model load failed: {type(e).__name__}: {e}"
        row["latency_s"] = round(time.time() - t0, 2)
        return row

    try:
        with torch.no_grad():
            # 32 random vocab tokens; rebuild on whichever device the model is.
            ids = torch.randint(0, max(1, row["vocab_size"] or 1), (1, 32),
                                device=next(model.parameters()).device)
            out = model(ids)
            logits = out.logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                row["note"] = "forward produced NaN/Inf logits"
            else:
                row["forward_ok"] = True
                row["note"] = "ok"
        del out, logits, ids, model
        torch.cuda.empty_cache()
    except Exception as e:
        row["note"] = f"forward failed: {type(e).__name__}: {e}"
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    row["latency_s"] = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Sharding + driver.
# ---------------------------------------------------------------------------
def shard(items, n_shards: int, shard_id: int):
    return [it for i, it in enumerate(items) if i % n_shards == shard_id]


def merge_shards(out_csv: str):
    pat = os.path.join(OUT_DIR, "ckpt_status_shard*.csv")
    import glob
    files = sorted(glob.glob(pat))
    if not files:
        print(f"[verify] no shards found at {pat}", file=sys.stderr)
        return
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # de-dup on (role, path) keeping last
    df = df.drop_duplicates(subset=["ckpt_role", "path"], keep="last")
    df = df.sort_values(["ckpt_role", "path"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[verify] merged {len(files)} shards → {out_csv}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="skip the model-load + forward step (only file check)")
    ap.add_argument("--out_csv", default=None,
                    help="output CSV; defaults to ckpt_status_shard{shard_id}.csv "
                         "if n_shards > 1, else ckpt_status.csv")
    ap.add_argument("--merge", action="store_true",
                    help="merge ckpt_status_shard*.csv into ckpt_status.csv "
                         "(no other work)")
    args = ap.parse_args()

    if args.merge:
        merge_shards(DEFAULT_OUT)
        return

    if args.out_csv is None:
        if args.n_shards > 1:
            args.out_csv = f"{OUT_DIR}/ckpt_status_shard{args.shard_id}.csv"
        else:
            args.out_csv = DEFAULT_OUT

    items = _ckpt_list()
    items = shard(items, args.n_shards, args.shard_id)
    print(f"[verify] shard {args.shard_id}/{args.n_shards}, "
          f"{len(items)} ckpts to check, "
          f"forward={'OFF' if args.quick else 'ON'}")

    rows = []
    for i, (role, path) in enumerate(items):
        try:
            r = verify_one(role, path, do_forward=not args.quick)
        except Exception:
            r = {
                "ckpt_role": role, "path": path, "exists": False,
                "tokenizer_ok": False, "model_ok": False, "forward_ok": False,
                "vocab_size": -1, "dtype": "", "latency_s": 0.0,
                "note": "uncaught: " + traceback.format_exc().splitlines()[-1],
            }
        r["gpu_id"] = args.shard_id
        rows.append(r)
        flag = ("PASS" if (r["forward_ok"] or (args.quick and r["exists"]))
                else ("FAIL" if not r["exists"] else "PARTIAL"))
        print(f"[verify] {flag:7s} {r['ckpt_role']:25s} "
              f"{r['note'][:80]:80s}  ({r['latency_s']:.1f}s)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[verify] wrote {args.out_csv}  ({len(df)} rows)")

    n_pass = int(df["forward_ok"].sum() if not args.quick else df["exists"].sum())
    print(f"[verify] summary: {n_pass}/{len(df)} ckpts passed "
          f"({'forward+load' if not args.quick else 'file check only'})")


if __name__ == "__main__":
    main()
