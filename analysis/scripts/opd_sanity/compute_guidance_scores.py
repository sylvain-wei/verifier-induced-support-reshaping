#!/usr/bin/env python
"""Phase 3 — Token-averaged log-probabilities under base / IF teacher / math teacher.

Why this script
---------------
For each base-generated response on the 200 stratified IF-train prompts, we
compute three length-normalized log-probs:

    lp_base(x, y) = (1/T) Σ_t log π_base(y_t | x, y_<t)
    lp_if(x, y)   = (1/T) Σ_t log π_if-teacher  (y_t | x, y_<t)
    lp_math(x, y) = (1/T) Σ_t log π_math-teacher(y_t | x, y_<t)

From these we build the OPD sanity-check signals:

    S_T_if    = lp_if              (how natural the response feels to IF teacher)
    S_T_math  = lp_math
    S_ratio_if   = lp_if   - lp_base   (does IF teacher prefer this response
                                        more than base does — i.e. the OPD pull)
    S_ratio_math = lp_math - lp_base   (counter-experiment)

The PRIMARY metric is `S_ratio_if` aggregated by DeepSeek label
(contentful / shortcut / fail). If shortcut > contentful in S_ratio_if, the
IF teacher would actively pull an OPD student toward shortcut behavior.

Counter-experiment (a) — teacher self-rollouts: we ALSO score the teacher's
own n=32 rollouts on the same prompts under both base and IF teacher; this
shows what the teacher's already-shifted generation pool looks like.

Implementation notes
--------------------
- Uses HuggingFace Transformers (NOT vLLM) for exact teacher-forced log-probs.
  This is the approach in `a1_teacher_forced_logprob.py`; we re-use its
  `_score_one_pair` core unchanged.
- Loads one model at a time (sequential). Each pass is ~1 h on 1×H20 for
  ~6,400 responses; ~3 h total for the three models.
- We score the same `(input, output)` pairs that already exist in the
  rollout JSONLs, using `input` verbatim (it includes the
  `"user\\n…\\nassistant\\n"` template), so logp values are directly
  commensurable with the prompt format the policy actually saw.

Outputs
-------
analysis/data/opd_sanity/guidance_scores_{model_tag}.parquet
    one row per rollout, with columns:
      prompt_id, sample_id, family, response_chars, n_resp_tokens,
      lp_base, lp_if, lp_math, S_ratio_if, S_ratio_math,
      verifier_score, verifier_pass, label
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, f"{ROOT}/analysis/scripts")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


MODEL_PATHS = {
    "base":         f"{ROOT}/models/Qwen3-8B-Base",
    "teacher_if":   f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface",
    "teacher_math": f"{ROOT}/checkpoints/verl_exp/DAPO_sh_repro/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface",
    # OPD vanilla student ckpt at step 50 of the E1 OPD run (used for
    # plan §7 D.1: per-position teacher–student JS overlap). Path resolved
    # from analysis/scripts/opd_e1/run_e2_student_after_opd.sh defaults.
    # NB: 20260522 run is missing safetensors (incomplete); 20260519 run has
    # full weights and is the one whose rollouts are in
    # rollout/opd_sanity/student_after_opd/responses.jsonl
    # (file dated May 20 — matches the 20260519 ckpt).
    "student_after_opd": f"{ROOT}/opd-lab/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_20260519_222832/global_step_50/actor/huggingface",
}


def _score_one_pair(model, tokenizer, prompt: str, response: str,
                    device: str = "cuda", max_ctx: int = 16384) -> Tuple[float, int]:
    """Token-averaged log-prob (nats/tok) of `response` given `prompt`.

    Adapted from `a1_teacher_forced_logprob.py::_score_one_pair`.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    full = prompt + response
    full_ids = tokenizer(full, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= p_len:
        return float("nan"), 0
    cap = min(max_ctx, getattr(model.config, "max_position_embeddings", max_ctx) or max_ctx)
    if full_ids.shape[1] > cap:
        full_ids = full_ids[:, :cap]
    n_resp = full_ids.shape[1] - p_len

    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        log_probs = torch.log_softmax(logits[0, p_len - 1: -1, :].float(), dim=-1)
        target = full_ids[0, p_len:]
        token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mean_lp = token_lp.mean().item()
    return mean_lp, int(n_resp)


def score_all(model_path: str, items: List[dict],
              dtype=torch.bfloat16, max_ctx: int = 16384) -> List[Tuple[float, int]]:
    print(f"[guidance] loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="balanced",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    results: List[Tuple[float, int]] = []
    n = len(items)
    for i, p in enumerate(items):
        try:
            mean_lp, n_tok = _score_one_pair(
                model, tokenizer, p["input"], p["output"], max_ctx=max_ctx,
            )
        except Exception as e:
            print(f"[guidance] WARN row {i} failed: {repr(e)[:120]}")
            mean_lp, n_tok = float("nan"), 0
        results.append((mean_lp, n_tok))
        if (i + 1) % 200 == 0 or i == n - 1:
            valid = [r[0] for r in results if not (isinstance(r[0], float) and np.isnan(r[0]))]
            print(f"[guidance]   {i+1}/{n}  mean_lp_so_far={np.mean(valid):.4f}")
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True,
                    choices=["base", "teacher_if", "student_after_opd"],
                    help="Which rollout pool to score. Both base/teacher_if "
                         "get scored under all three reference models; "
                         "student_after_opd is the new D.1 pool.")
    ap.add_argument("--rollout_dir", default=f"{ROOT}/rollout/opd_sanity")
    ap.add_argument("--labelled_parquet", default=None,
                    help=f"Default: analysis/data/opd_sanity/responses_labelled_{{tag}}.parquet")
    ap.add_argument("--out_dir", default=f"{ROOT}/analysis/data/opd_sanity")
    ap.add_argument("--scorers", default="base,teacher_if,teacher_math",
                    help="Comma-separated subset of MODEL_PATHS to score "
                         "under. For D.1 student pool, recommend "
                         "'base,teacher_if,student_after_opd'.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_ctx", type=int, default=16384)
    # 8-way parallel sharding: shard rows by (prompt_id, sample_id) hash.
    # Each shard process loads ALL scorer models and scores its 1/N slice.
    # Final merge produces guidance_scores_{tag}.parquet from the per-shard
    # files. Naming: guidance_scores_{tag}__shard{i}of{N}.parquet.
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="Instead of computing, just concat all shard parquets "
                         "for this --model_tag into the final output.")
    args = ap.parse_args()

    # --- Merge mode (after all shards finished) ---
    if args.merge:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(
            args.out_dir, f"guidance_scores_{args.model_tag}.parquet")
        pieces = []
        import glob as _glob
        pattern = os.path.join(
            args.out_dir,
            f"guidance_scores_{args.model_tag}__shard*.parquet")
        for f in sorted(_glob.glob(pattern)):
            pieces.append(pd.read_parquet(f))
            print(f"[guidance][merge] loaded {f} ({len(pieces[-1])} rows)")
        if not pieces:
            print(f"[guidance][merge] no shards found at {pattern}")
            return
        out_df = pd.concat(pieces, ignore_index=True)
        out_df = (out_df.drop_duplicates(subset=["prompt_id", "sample_id"])
                          .sort_values(["prompt_id", "sample_id"])
                          .reset_index(drop=True))
        out_df.to_parquet(out_path, index=False)
        print(f"[guidance][merge] wrote {out_path}  ({len(out_df)} rows)")
        return

    rollout_jsonl = os.path.join(args.rollout_dir, args.model_tag, "responses.jsonl")
    print(f"[guidance] reading {rollout_jsonl}")
    rows: List[dict] = []
    with open(rollout_jsonl) as f:
        for ln in f:
            rows.append(json.loads(ln))
    if args.limit:
        rows = rows[:args.limit]
    if args.n_shards > 1:
        # Hash-based shard selection on (prompt_id, sample_id) so each
        # parallel worker handles a deterministic 1/N slice.
        import hashlib as _hl
        def _shard_of(rec):
            key = f"{rec.get('prompt_id','')}|{rec.get('sample_id',0)}"
            h = int(_hl.md5(key.encode()).hexdigest(), 16)
            return h % args.n_shards
        rows = [r for r in rows if _shard_of(r) == args.shard_id]
        print(f"[guidance]   shard {args.shard_id}/{args.n_shards}: "
              f"{len(rows)} rows")
    print(f"[guidance]   rows: {len(rows)}")

    # Optional join with labels (only available for the two pools we labelled).
    labelled_path = args.labelled_parquet or os.path.join(
        args.out_dir, f"responses_labelled_{args.model_tag}.parquet"
    )
    label_map = {}
    if os.path.exists(labelled_path):
        ldf = pd.read_parquet(labelled_path)
        for _, lr in ldf.iterrows():
            label_map[(lr["prompt_id"], int(lr["sample_id"]))] = lr["label"]
        print(f"[guidance] joined {len(label_map)} labels from {labelled_path}")
    else:
        print(f"[guidance] labels not found at {labelled_path}; label column will be NaN")

    scorer_tags = [s.strip() for s in args.scorers.split(",") if s.strip()]
    score_cols = {}
    nresp_cols = {}
    for tag in scorer_tags:
        path = MODEL_PATHS[tag]
        results = score_all(path, rows, max_ctx=args.max_ctx)
        score_cols[f"lp_{tag}"] = [r[0] for r in results]
        nresp_cols[f"n_resp_tokens_{tag}"] = [r[1] for r in results]

    out_df = pd.DataFrame({
        "prompt_id":      [r["prompt_id"] for r in rows],
        "sample_id":      [r["sample_id"] for r in rows],
        "family":         [r["family"] for r in rows],
        "response_chars": [r["response_chars"] for r in rows],
        "verifier_score": [r["score"] for r in rows],
        "verifier_pass":  [bool(r["acc"]) for r in rows],
        "label":          [label_map.get((r["prompt_id"], int(r["sample_id"])), None) for r in rows],
    })
    for k, v in score_cols.items():
        out_df[k] = v
    for k, v in nresp_cols.items():
        out_df[k] = v
    if "lp_base" in out_df.columns and "lp_teacher_if" in out_df.columns:
        out_df["S_ratio_if"] = out_df["lp_teacher_if"] - out_df["lp_base"]
    if "lp_base" in out_df.columns and "lp_teacher_math" in out_df.columns:
        out_df["S_ratio_math"] = out_df["lp_teacher_math"] - out_df["lp_base"]

    os.makedirs(args.out_dir, exist_ok=True)
    if args.n_shards > 1:
        out_path = os.path.join(
            args.out_dir,
            f"guidance_scores_{args.model_tag}__shard{args.shard_id}of{args.n_shards}.parquet"
        )
    else:
        out_path = os.path.join(args.out_dir, f"guidance_scores_{args.model_tag}.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"[guidance] wrote {out_path}")

    # Quick text summary
    print()
    print(out_df[[c for c in out_df.columns if c.startswith("lp_") or c.startswith("S_ratio_")]].describe(percentiles=[0.5]).to_string())


if __name__ == "__main__":
    main()
