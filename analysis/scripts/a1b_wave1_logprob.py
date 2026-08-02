#!/usr/bin/env python
"""WAVE 1 — Generalised teacher-forced logprob + top-K logit cache.

Extends `a1_teacher_forced_logprob.py` to:

  - support arbitrary (model_tag, dataset) jobs via CLI;
  - cache top-K (ids, logits) at every response position when --save_topk_logits;
  - filter rollouts to a fixed evaluation subset (ifeval_100 / ifbench_100) and
    keep ≤ `--samples_per_prompt` samples per prompt to bound storage;
  - write one parquet per job under `analysis/data/wave1/{model_tag}__{dataset}.parquet`.

Output schema (one row per response token):
    model_tag (str), dataset (str), prompt_id (str), sample_id (int),
    t (int, 1-indexed), token_id (int), logp (float32),
    topk_ids (list[int64], len K) — only present if --save_topk_logits,
    topk_logits (list[float32], len K) — same.

Plus a per-(prompt_id, sample_id) summary parquet at
`analysis/data/wave1/{model_tag}__{dataset}__summary.parquet` with:
    prompt_id, sample_id, n_resp_tokens, mean_logp, sum_logp, prompt_chars,
    response_chars, sample_acc (from rollout, where present).

Usage
-----
    python a1b_wave1_logprob.py \\
        --model_path  "${PROJECT_ROOT}/models/Qwen3-8B-Base" \\
        --model_tag   B_q3 \\
        --rollout_jsonl "${PROJECT_ROOT}/rollout/val_rollout/<run>/0.jsonl" \\
        --dataset     aime \\
        --samples_per_prompt 32 \\
        --save_topk_logits --K 64 \\
        --gpu_id 0 \\
        --out_dir analysis/data/wave1

Filters (--dataset)
-------------------
    aime    : keep rows with the AIME marker in `input`.
    ifeval  : keep rows whose prompt is one of the 100 in
              analysis/data/eval_subsets/ifeval_100.jsonl. Rollout file is
              the same `val_rollout/.../{step}.jsonl` but we drop AIME rows.
    ifbench : keep rows whose prompt is one of the 100 in
              analysis/data/eval_subsets/ifbench_100.jsonl. Rollout file
              should be from `val_rollout_ifbench/...` (no AIME mixed in).

Convention: `prompt_id` for AIME = md5 prefix of user_content (matches
a1/a2 scripts). For IFEval / IFBench, `prompt_id` = the `key` field from the
eval-subset jsonl, kept as a string for consistency.

GPU & memory
------------
- One model is loaded on a single GPU (`device_map={"":gpu_id}`). bf16.
- Per-row forward truncates the (prompt + response) to `--max_ctx` (default
  16384) to bound activation memory.
- Top-K cache uses int64 + float32 numpy arrays per row; for AIME (~2k tok
  per resp × 960 rows × K=64) the parquet is ~700 MB, which pyarrow handles.

This script is idempotent: if `out_dir/<tag>__<ds>.parquet` already exists,
it skips (unless --force).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kl_utils import topk_logits_all_positions  # noqa: E402, used indirectly  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
AIME_MARKER = "Solve the following math problem step by step"
REPO_ROOT = os.environ.get("PROJECT_ROOT", ".")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis/data/wave1")
EVAL_SUBSET_DIR = os.path.join(REPO_ROOT, "analysis/data/eval_subsets")


# ----------------------------------------------------------------------
# Prompt utilities
# ----------------------------------------------------------------------
def _strip_chat_envelope(s: str) -> str:
    """Strip the chat-template envelope from val_rollout `input` strings.

    Two envelope styles encountered:
      1) Qwen3-8B-Base lineage (no system prompt baked in by reward eval):
            'user\\n<content>\\nassistant\\n'
      2) Qwen2.5-Math-7B lineage (default system prompt prepended):
            'system\\n<sys>\\nuser\\n<content>\\nassistant\\n'
         where <sys> is something like
            'Please reason step by step, and put your final answer within
             \\\\boxed{}.'

    We strip both forms so that `prompt_id = md5(user_content)` is canonical
    across base models — required for cross-base IFEval / IFBench prompt
    matching (eval_subsets/*.jsonl have no envelope).
    """
    s2 = s
    # Optional leading system\n<sys>\nuser\n
    if s2.startswith("system\n"):
        # Find the next 'user\n' and skip past it.
        marker = "\nuser\n"
        idx = s2.find(marker)
        if idx >= 0:
            s2 = s2[idx + len(marker):]
        # If no user\n marker, leave alone (unexpected format).
    elif s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def _aime_prompt_id(user_content: str) -> str:
    return hashlib.md5(user_content.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Rollout loader
# ----------------------------------------------------------------------
def load_eval_subset_prompts(dataset: str) -> Dict[str, str]:
    """Return {prompt_text -> prompt_id_string} for ifeval_100 / ifbench_100."""
    if dataset == "ifeval":
        path = os.path.join(EVAL_SUBSET_DIR, "ifeval_100.jsonl")
    elif dataset == "ifbench":
        path = os.path.join(EVAL_SUBSET_DIR, "ifbench_100.jsonl")
    else:
        raise ValueError(f"no eval subset for dataset={dataset!r}")
    out: Dict[str, str] = {}
    with open(path) as f:
        for ln in f:
            o = json.loads(ln)
            out[o["prompt_text"].strip()] = str(o["prompt_id"])
    if not out:
        raise RuntimeError(f"empty subset at {path}")
    return out


def load_rollout_records(
    jsonl_path: str,
    dataset: str,
    samples_per_prompt: int,
    max_total_rows: Optional[int] = None,
) -> List[dict]:
    """Stream one rollout jsonl, filter to `dataset`, and keep ≤ N samples
    per prompt. Returns list of dicts with keys:
        prompt_id, prompt_text (raw with envelope), user_content, response,
        response_chars, sample_id, sample_acc.

    For IFEval / IFBench, prompt_id comes from the eval subset jsonl. Rows
    whose prompt is *not* in the subset are dropped.
    """
    if dataset == "aime":
        subset_map = None
    elif dataset in ("ifeval", "ifbench"):
        subset_map = load_eval_subset_prompts(dataset)
    else:
        raise ValueError(f"unknown dataset={dataset!r}")

    out: List[dict] = []
    per_prompt_n: Dict[str, int] = {}

    with open(jsonl_path) as f:
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
                pid = _aime_prompt_id(uc)
            else:
                if is_aime:
                    continue
                uc = _strip_chat_envelope(inp)
                pid = subset_map.get(uc)  # type: ignore[union-attr]
                if pid is None:
                    continue

            n_seen = per_prompt_n.get(pid, 0)
            if n_seen >= samples_per_prompt:
                continue
            resp = o.get("output", "") or ""
            if not resp:
                continue

            acc_raw = o.get("acc")
            if isinstance(acc_raw, bool):
                acc = acc_raw
            else:
                try:
                    acc = float(o.get("score", 0.0)) >= 1.0
                except Exception:
                    acc = None  # type: ignore[assignment]

            out.append({
                "prompt_id": str(pid),
                "prompt_text": inp,
                "user_content": uc,
                "response": resp,
                "response_chars": len(resp),
                "sample_id": n_seen,
                "sample_acc": acc,
            })
            per_prompt_n[pid] = n_seen + 1
            if max_total_rows is not None and len(out) >= max_total_rows:
                break
    return out


# ----------------------------------------------------------------------
# Forward + topk extraction
# ----------------------------------------------------------------------
@torch.no_grad()
def score_one(
    model,
    tokenizer,
    prompt: str,
    response: str,
    save_topk: bool,
    K: int,
    device: str,
    max_ctx: int,
) -> Dict[str, np.ndarray]:
    """Returns dict with keys:
       n_resp (int)
       token_ids (n_resp,) int64
       logp      (n_resp,) float32 — log p(target_t | prompt + resp[:t])
       topk_ids  (n_resp, K) int64    — only if save_topk
       topk_logits (n_resp, K) float32 — only if save_topk
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + response, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    p_len = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= p_len:
        return {"n_resp": 0,
                "token_ids": np.zeros(0, dtype=np.int64),
                "logp": np.zeros(0, dtype=np.float32),
                "topk_ids": np.zeros((0, K), dtype=np.int64),
                "topk_logits": np.zeros((0, K), dtype=np.float32)}
    if full_ids.shape[1] > max_ctx:
        full_ids = full_ids[:, :max_ctx]
    n_resp = int(full_ids.shape[1] - p_len)

    logits = model(full_ids).logits[0]  # (T, V)
    # logits[t-1] predicts token t. Response tokens are at positions p_len..T-1
    # so corresponding logit rows are p_len-1 .. T-2.
    slab = logits[p_len - 1: -1, :].float()      # (n_resp, V)
    target = full_ids[0, p_len:]                  # (n_resp,)

    # per-token log-prob of the actual target
    log_probs = torch.log_softmax(slab, dim=-1)
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (n_resp,)

    res = {
        "n_resp": n_resp,
        "token_ids": target.cpu().numpy().astype(np.int64),
        "logp": token_lp.cpu().numpy().astype(np.float32),
    }

    if save_topk:
        tk = torch.topk(slab, K, dim=-1)
        res["topk_ids"] = tk.indices.cpu().numpy().astype(np.int64)
        res["topk_logits"] = tk.values.cpu().numpy().astype(np.float32)

    return res


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_tag", required=True,
                    help="Short tag, e.g. B_q3, M_q3, I_q3, B_q25m, M_q25m, I_q25m")
    ap.add_argument("--rollout_jsonl", required=True)
    ap.add_argument("--dataset", required=True, choices=["aime", "ifeval", "ifbench"])
    ap.add_argument("--samples_per_prompt", type=int, default=32,
                    help="Cap samples per prompt (AIME has 32; IF subsample 8).")
    ap.add_argument("--max_total_rows", type=int, default=None,
                    help="Optional global cap on rows for smoke tests.")
    ap.add_argument("--save_topk_logits", action="store_true")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--max_ctx", type=int, default=16384)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output parquet.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.model_tag}__{args.dataset}"
    out_pq = os.path.join(args.out_dir, f"{base}.parquet")
    out_summary = os.path.join(args.out_dir, f"{base}__summary.parquet")

    if os.path.exists(out_pq) and not args.force:
        print(f"[a1b] {out_pq} already exists; use --force to overwrite. Exiting.")
        return

    # GPU pinning. We honour CUDA_VISIBLE_DEVICES if set; otherwise pin via
    # device_map. The launcher script sets CUDA_VISIBLE_DEVICES=$gpu_id, in
    # which case the visible device index is always 0.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.gpu_id}"
    print(f"[a1b] gpu_id={args.gpu_id}  device={device}  CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Load rollouts.
    t0 = time.time()
    records = load_rollout_records(
        args.rollout_jsonl, args.dataset,
        samples_per_prompt=args.samples_per_prompt,
        max_total_rows=args.max_total_rows,
    )
    if not records:
        print(f"[a1b] FATAL: 0 records loaded from {args.rollout_jsonl} for "
              f"dataset={args.dataset}.")
        sys.exit(2)
    n_unique_prompts = len(set(r["prompt_id"] for r in records))
    print(f"[a1b] {args.model_tag} × {args.dataset}: loaded {len(records)} rows "
          f"({n_unique_prompts} unique prompts) in {time.time()-t0:.1f}s")

    # Load model.
    t0 = time.time()
    print(f"[a1b] loading model {args.model_path} on {device} ...")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[a1b] model loaded in {time.time()-t0:.1f}s")

    # Score each record.
    rows: List[dict] = []
    summary_rows: List[dict] = []
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            res = score_one(model, tokenizer,
                            rec["prompt_text"], rec["response"],
                            save_topk=args.save_topk_logits, K=args.K,
                            device=device, max_ctx=args.max_ctx)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[a1b] OOM on row {i} pid={rec['prompt_id']} sid={rec['sample_id']}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"[a1b] row {i} error pid={rec['prompt_id']} sid={rec['sample_id']}: {e}")
            continue

        n_resp = res["n_resp"]
        if n_resp == 0:
            continue

        token_ids = res["token_ids"]
        logp = res["logp"]
        if args.save_topk_logits:
            topk_ids = res["topk_ids"]      # (n_resp, K)
            topk_lg = res["topk_logits"]    # (n_resp, K)

        # per-token rows
        for t in range(n_resp):
            row = {
                "model_tag": args.model_tag,
                "dataset": args.dataset,
                "prompt_id": rec["prompt_id"],
                "sample_id": int(rec["sample_id"]),
                "t": t + 1,                 # 1-indexed response position
                "token_id": int(token_ids[t]),
                "logp": float(logp[t]),
            }
            if args.save_topk_logits:
                row["topk_ids"] = topk_ids[t].tolist()
                row["topk_logits"] = topk_lg[t].astype(np.float32).tolist()
            rows.append(row)

        summary_rows.append({
            "model_tag": args.model_tag,
            "dataset": args.dataset,
            "prompt_id": rec["prompt_id"],
            "sample_id": int(rec["sample_id"]),
            "n_resp_tokens": n_resp,
            "mean_logp": float(np.mean(logp)),
            "sum_logp": float(np.sum(logp)),
            "prompt_chars": len(rec["prompt_text"]),
            "response_chars": rec["response_chars"],
            "sample_acc": rec["sample_acc"],
        })

        if (i + 1) % 32 == 0 or i == len(records) - 1:
            elapsed = time.time() - t0
            rps = (i + 1) / max(elapsed, 1e-6)
            eta = (len(records) - i - 1) / max(rps, 1e-6)
            mem = torch.cuda.memory_allocated(device) / 1e9
            print(f"[a1b]   row {i+1}/{len(records)}  "
                  f"n_resp={n_resp:4d}  mean_lp={float(np.mean(logp)):+.3f}  "
                  f"GPU={mem:.1f}GB  rate={rps:.2f}/s  ETA={eta/60:.1f}m")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Write outputs.
    print(f"[a1b] writing {len(rows)} per-token rows  → {out_pq}")
    df = pd.DataFrame(rows)
    df.to_parquet(out_pq, index=False)
    sz = os.path.getsize(out_pq) / 1e6
    print(f"[a1b]   wrote {sz:.1f} MB")

    print(f"[a1b] writing {len(summary_rows)} per-sample rows  → {out_summary}")
    pd.DataFrame(summary_rows).to_parquet(out_summary, index=False)
    print(f"[a1b]   wrote {os.path.getsize(out_summary)/1e6:.1f} MB")

    print(f"[a1b] DONE  {args.model_tag} × {args.dataset}")


if __name__ == "__main__":
    main()
