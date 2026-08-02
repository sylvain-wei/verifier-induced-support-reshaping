#!/usr/bin/env python
"""F1 vLLM variant — fast batched implementation of single-token / forced-
prefix intervention conditions.

Plan §6 — single-token intervention reduces to "start the response with a
specific token". For Qwen3-family models with shared vocab the intervention's
top-1 token id, when decoded to its STRING form and prepended after
'assistant\\n', acts as a forced first-token prefix that vLLM can batch-
generate from. This is functionally equivalent to a HF LogitsProcessor that
masks out everything except the target id at step 1, but ~10× faster
because vLLM batches 32 samples in parallel.

This script is a drop-in replacement for `f1_single_token_intervention.py`:
- supports the same conditions
- same output schema (per-row parquet with prompt_id, sample_id, response,
  acc, mode, …)
- consumes the precomputed first-token-top-K JSON from
  `f1_precompute_topk.py` (so we don't need to load the intervention model
  here at all)

Usage
-----
    # 1) precompute top-K once per model (HF, ~1 min/model on 1 GPU)
    python f1_precompute_topk.py --model_tag B_q3 --gpu_id 0
    python f1_precompute_topk.py --model_tag I_q3 --gpu_id 0

    # 2) run each (condition, shard) on its own GPU via vLLM
    python f1_vllm_intervention.py \\
        --condition single_token_C1 \\
        --shard_id 0 --n_shards 8 --tp 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd

REPO = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, f"{REPO}/analysis/scripts")
from classify_opening_modes import classify_mode  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "verl"))
from verl.utils.reward_score.math_dapo import compute_score  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

OUT_DIR = f"{REPO}/analysis/data/f1_intervention"
AIME_PARQUET = f"{REPO}/data/aime24/test.parquet"
CKPT_BASE = f"{REPO}/checkpoints/verl_exp/DAPO_sh_repro"

MODEL_PATHS = {
    "B_q3": f"{REPO}/models/Qwen3-8B-Base",
    "I_q3": f"{CKPT_BASE}/b2r1_Qwen3-8B-Base_IFTrain_local_H20/global_step_100/actor/huggingface",
    # Math model not currently used in plan §6 conditions; included for symmetry
    "M_q3": f"{CKPT_BASE}/b1r1_Qwen3-8B-Base_math7.5k_local_H20/global_step_220/actor/huggingface",
    # Q2.5-Math lineage (WAVE 6 robustness)
    "B_q25m": f"{REPO}/models/Qwen2.5-Math-7B",
    "M_q25m": f"{CKPT_BASE}/b1r1_Qwen2.5-math-7B_math7.5k_local_H20/global_step_480/actor/huggingface",
    "I_q25m": f"{CKPT_BASE}/b2r1_Qwen2.5-math-7B_IFTrain_local_H20/global_step_380/actor/huggingface",
}


# (primary, intervention, force_prefix_string, force_strategy)
#   force_strategy ∈ {None, "intervention_top1", "primary_top2"}
CONDITIONS = {
    "free":             ("B_q3", None,    "", None),
    "free_I_q3":        ("I_q3", None,    "", None),
    "single_token_C1":  ("B_q3", "I_q3",  "", "intervention_top1"),
    "single_token_C2":  ("I_q3", "B_q3",  "", "intervention_top1"),
    "random_top2_B_q3": ("B_q3", None,    "", "primary_top2"),
    "random_top2_I_q3": ("I_q3", None,    "", "primary_top2"),
    "forced_DRI_B_q3":  ("B_q3", None,    "Let me solve this step by step.\n\n", None),
    "forced_DAI_B_q3":  ("B_q3", None,    "Answer: ", None),
    "forced_DRI_I_q3":  ("I_q3", None,    "Let me solve this step by step.\n\n", None),
    "forced_DAI_I_q3":  ("I_q3", None,    "Answer: ", None),
    # WAVE 6 — q25m mirror conditions (Plan §9 robustness)
    "free_q25m":             ("B_q25m", None,     "", None),
    "free_I_q25m":           ("I_q25m", None,     "", None),
    "single_token_C1_q25m":  ("B_q25m", "I_q25m", "", "intervention_top1"),
    "single_token_C2_q25m":  ("I_q25m", "B_q25m", "", "intervention_top1"),
    "random_top2_B_q25m":    ("B_q25m", None,     "", "primary_top2"),
    "random_top2_I_q25m":    ("I_q25m", None,     "", "primary_top2"),
    "forced_DRI_B_q25m":     ("B_q25m", None,     "Let me solve this step by step.\n\n", None),
    "forced_DAI_B_q25m":     ("B_q25m", None,     "Answer: ", None),
    "forced_DRI_I_q25m":     ("I_q25m", None,     "Let me solve this step by step.\n\n", None),
    "forced_DAI_I_q25m":     ("I_q25m", None,     "Answer: ", None),
}

# C.3 position sweep — auto-generate `{base}_pos{N}` conditions for N in
# {2, 3, 5, 20, 50, 100}. Strategy / primary / intervention / force_prefix
# all inherit from the base (pos=1) condition; only the `--forced_position`
# CLI arg differs at runtime.
_C3_BASES = ["single_token_C1", "single_token_C2",
             "random_top2_B_q3", "random_top2_I_q3"]
_C3_POSITIONS = [2, 3, 5, 20, 50, 100]
for _base in _C3_BASES:
    for _N in _C3_POSITIONS:
        CONDITIONS[f"{_base}_pos{_N}"] = CONDITIONS[_base]

# Extra density for the main Fig. 4b claim: only sweep Base-primary decoding
# with the IF-side routing token, avoiding new recovery/control runs.
_C3_C1_EXTRA_POSITIONS = [4, 8, 12, 16, 24, 32, 75]
for _N in _C3_C1_EXTRA_POSITIONS:
    CONDITIONS[f"single_token_C1_pos{_N}"] = CONDITIONS["single_token_C1"]


def parse_forced_position(condition_name: str) -> int:
    """Parse trailing `_pos{N}` from a condition name; default 1."""
    m = re.search(r"_pos(\d+)$", condition_name)
    return int(m.group(1)) if m else 1


def aime_prompt_id(prompt_text: str) -> str:
    s = prompt_text
    if s.startswith("user\n"):
        s = s[len("user\n"):]
    if s.endswith("\nassistant\n"):
        s = s[:-len("\nassistant\n")]
    elif s.endswith("assistant\n"):
        s = s[:-len("assistant\n")]
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()[:16]


def build_prompt(row, prefix: str = "") -> str:
    parts = []
    for m in row["prompt"]:
        parts.append(f"{m['role']}\n{m['content']}")
    parts.append("assistant\n")
    s = "\n".join(parts)
    if prefix:
        s = s + prefix
    return s


def load_topk(model_tag: str, out_dir: str = OUT_DIR) -> dict:
    p = os.path.join(out_dir, f"first_token_topk__{model_tag}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing — run f1_precompute_topk.py --model_tag {model_tag}")
    with open(p) as f:
        return json.load(f)["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default=None,
                    help="Single condition to run, OR pass --primary to run "
                         "all conditions sharing that primary in one process.")
    ap.add_argument("--conditions", default=None,
                    help="Comma-separated explicit conditions to run in one "
                         "process. All listed conditions must share the same "
                         "primary model.")
    ap.add_argument("--primary", default=None, choices=list(MODEL_PATHS.keys()),
                    help="Run all conditions with this primary in one process "
                         "(saves vLLM load time). Conflicts with --condition.")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--data_path", default=AIME_PARQUET)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--forced_position", type=int, default=None,
                    help="Position N at which to inject the forced token "
                         "(1-indexed; 1 = first response token). If unset, "
                         "the value is auto-parsed from the condition name's "
                         "trailing `_pos{N}` suffix (default 1 if absent).")
    args = ap.parse_args()

    if args.smoke:
        args.n_samples = 4
        args.max_new_tokens = 256

    mode_count = sum(x is not None for x in (args.condition, args.conditions, args.primary))
    if mode_count != 1:
        raise SystemExit("Pass exactly one of --condition, --conditions, or --primary.")

    if args.condition is not None:
        conditions_to_run = [args.condition]
        primary_tag = CONDITIONS[args.condition][0]
    elif args.conditions is not None:
        conditions_to_run = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in conditions_to_run if c not in CONDITIONS]
        if unknown:
            raise SystemExit(f"Unknown condition(s): {unknown}")
        primary_tags = {CONDITIONS[c][0] for c in conditions_to_run}
        if len(primary_tags) != 1:
            raise SystemExit(
                "--conditions entries must share one primary model; got "
                f"{sorted(primary_tags)}"
            )
        primary_tag = next(iter(primary_tags))
        print(f"[f1v] explicit conditions for primary={primary_tag}: {conditions_to_run}")
    else:
        primary_tag = args.primary
        conditions_to_run = [c for c, (p, _, _, _) in CONDITIONS.items()
                             if p == primary_tag]
        # Filter out already-done outputs
        conditions_to_run = [
            c for c in conditions_to_run
            if not os.path.exists(os.path.join(args.out_dir,
                f"{c}__shard{args.shard_id}of{args.n_shards}.parquet"))
        ]
        if not conditions_to_run:
            print(f"[f1v] all conditions for primary={primary_tag} "
                  f"shard {args.shard_id}/{args.n_shards} already done; exit.")
            return
        print(f"[f1v] primary={primary_tag} → conditions: {conditions_to_run}")

    primary_path = MODEL_PATHS[primary_tag]
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    if args.smoke:
        df = df.head(1).reset_index(drop=True)
    if args.n_shards > 1:
        df = df.iloc[args.shard_id::args.n_shards].reset_index(drop=True)
    print(f"[f1v] handling {len(df)} prompts (shard {args.shard_id}/{args.n_shards})")

    # Load vLLM once for the whole primary batch.
    print(f"[f1v] loading vLLM model {primary_path} (tp={args.tp})")
    t0 = time.time()
    llm = LLM(
        model=primary_path,
        tokenizer=primary_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        enforce_eager=False,
        enable_chunked_prefill=True,
        max_num_seqs=128,
        trust_remote_code=False,
        disable_log_stats=True,
        seed=args.seed,
    )
    print(f"[f1v] vLLM ready in {time.time()-t0:.1f}s")

    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for cond in conditions_to_run:
        out_path = os.path.join(args.out_dir,
            f"{cond}__shard{args.shard_id}of{args.n_shards}.parquet")
        if os.path.exists(out_path):
            print(f"[f1v]   {cond}: {out_path} exists; skipping")
            continue
        primary_tag2, intervention_tag, force_prefix, force_strategy = CONDITIONS[cond]
        assert primary_tag2 == primary_tag, f"primary mismatch in {cond}"

        # Resolve forced position: CLI override > condition-name suffix > 1.
        if args.forced_position is not None:
            forced_position = args.forced_position
        else:
            forced_position = parse_forced_position(cond)

        # Pick per-prompt forced token text.
        topk_lookup = None
        if force_strategy == "intervention_top1":
            topk_lookup = load_topk(intervention_tag, out_dir=args.out_dir)
        elif force_strategy == "primary_top2":
            topk_lookup = load_topk(primary_tag, out_dir=args.out_dir)

        prompts = []
        pids = []
        forced_token_ids = []
        forced_token_texts = []
        for idx, row in df.iterrows():
            base_prompt = build_prompt(row, prefix=force_prefix)
            pid = aime_prompt_id(base_prompt)
            ftok_text = None
            ftok_id = None
            if topk_lookup is not None:
                entry = topk_lookup.get(pid)
                if entry is None:
                    print(f"[f1v]   WARN no topk for pid={pid}, skipping force")
                else:
                    if force_strategy == "intervention_top1":
                        ftok_id = entry["top1_id"]
                        ftok_text = entry["top1_text"]
                    elif force_strategy == "primary_top2":
                        ftok_id = entry["top2_id"]
                        ftok_text = entry["top2_text"]
            pids.append(pid)
            forced_token_ids.append(ftok_id)
            forced_token_texts.append(ftok_text)

            # For pos==1 we keep the original behaviour: append forced token
            # text directly to the (force_prefix-augmented) base_prompt and
            # let vLLM batch-generate. For pos>1 we run a 2-stage call.
            if forced_position == 1 and ftok_text:
                prompts.append(base_prompt + ftok_text)
            else:
                prompts.append(base_prompt)

        print(f"[f1v]   running cond={cond}  primary={primary_tag}  "
              f"force_strategy={force_strategy}  force_prefix={force_prefix!r}  "
              f"forced_position={forced_position}")
        t1 = time.time()

        if forced_position == 1:
            # Original single-stage path (production-tested for pos=1).
            outputs = llm.generate(prompts, sp)
            print(f"[f1v]     gen done in {time.time()-t1:.1f}s")

            rows = []
            for i, out in enumerate(outputs):
                gt = str(df.iloc[i]["reward_model"]["ground_truth"])
                ftok_text = forced_token_texts[i]
                ftok_id = forced_token_ids[i]
                for sid, cand in enumerate(out.outputs):
                    text = cand.text
                    full = (force_prefix + (ftok_text or "")) + text
                    res = compute_score(full, gt)
                    rows.append({
                        "condition": cond,
                        "primary_tag": primary_tag,
                        "intervention_tag": intervention_tag,
                        "prompt_id": pids[i],
                        "prompt_idx": i,
                        "sample_id": sid,
                        "forced_token_id": ftok_id,
                        "forced_token_text": ftok_text,
                        "forced_position": forced_position,
                        "free_prefix_text": "",
                        "free_prefix_n_tokens": 0,
                        "stage1_truncated": False,
                        "force_prefix": force_prefix,
                        "response": text,
                        "response_chars": len(text),
                        "n_resp_tokens": len(cand.token_ids) if hasattr(cand, "token_ids") else len(text),
                        "score": float(res["score"]),
                        "acc": bool(res["acc"]),
                        "pred": str(res["pred"]),
                        "mode": classify_mode(text),
                    })
        else:
            # ---- Stage 1: free decode N-1 tokens (n=n_samples per prompt) ----
            N = forced_position
            sp_stage1 = SamplingParams(
                n=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=N - 1,
                seed=args.seed,
            )
            outs_stage1 = llm.generate(prompts, sp_stage1)
            print(f"[f1v]     stage1 done in {time.time()-t1:.1f}s")

            # ---- Stage 2: per (prompt, sample) build new prompt with forced token ----
            t2 = time.time()
            new_prompts = []
            stage1_token_id_lists = []  # to compare against `cand.token_ids`
            free_prefix_texts = []
            free_prefix_n_tokens_list = []
            stage1_truncated_flags = []
            stage2_meta = []  # parallel: (prompt_idx, sample_id, ftok_text, ftok_id)
            for i, out in enumerate(outs_stage1):
                ftok_text = forced_token_texts[i]
                ftok_id = forced_token_ids[i]
                base_prompt = prompts[i]
                for sid, cand in enumerate(out.outputs):
                    free_prefix_text = cand.text
                    n_free_tokens = (len(cand.token_ids)
                                     if hasattr(cand, "token_ids") else None)
                    # `cand.text` may include the model's natural EOS / stop
                    # before reaching N-1 tokens. We track this case but still
                    # try to inject the forced token next.
                    truncated = (n_free_tokens is not None
                                 and n_free_tokens < (N - 1))

                    # Build stage-2 prompt: base_prompt is the original prompt
                    # (no forced prefix) so we just append free_prefix_text +
                    # forced token text.
                    if ftok_text:
                        new_prompt = base_prompt + free_prefix_text + ftok_text
                    else:
                        # No forced token -> pure free continuation (free baseline
                        # at position N is well-defined by joining stage1 + stage2).
                        new_prompt = base_prompt + free_prefix_text

                    new_prompts.append(new_prompt)
                    free_prefix_texts.append(free_prefix_text)
                    free_prefix_n_tokens_list.append(n_free_tokens)
                    stage1_truncated_flags.append(truncated)
                    stage2_meta.append((i, sid, ftok_text, ftok_id))

            sp_stage2 = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=max(args.max_new_tokens - N, 1),
                seed=args.seed,
            )
            outs_stage2 = llm.generate(new_prompts, sp_stage2)
            print(f"[f1v]     stage2 done in {time.time()-t2:.1f}s")

            rows = []
            for j, out2 in enumerate(outs_stage2):
                i, sid, ftok_text, ftok_id = stage2_meta[j]
                gt = str(df.iloc[i]["reward_model"]["ground_truth"])
                free_prefix_text = free_prefix_texts[j]
                stage2_text = out2.outputs[0].text
                full_response = free_prefix_text + (ftok_text or "") + stage2_text
                full = force_prefix + full_response
                res = compute_score(full, gt)
                # Combined token count: stage1 tokens (≤ N-1) + 1 (forced)
                # + stage2 tokens. For schema parity with pos=1 we report the
                # total "response token count" if available.
                stage2_ntok = (len(out2.outputs[0].token_ids)
                               if hasattr(out2.outputs[0], "token_ids") else 0)
                stage1_ntok = free_prefix_n_tokens_list[j] or 0
                total_ntok = (stage1_ntok
                              + (1 if ftok_text else 0)
                              + stage2_ntok)
                rows.append({
                    "condition": cond,
                    "primary_tag": primary_tag,
                    "intervention_tag": intervention_tag,
                    "prompt_id": pids[i],
                    "prompt_idx": i,
                    "sample_id": sid,
                    "forced_token_id": ftok_id,
                    "forced_token_text": ftok_text,
                    "forced_position": forced_position,
                    "free_prefix_text": free_prefix_text,
                    "free_prefix_n_tokens": stage1_ntok,
                    "stage1_truncated": stage1_truncated_flags[j],
                    "force_prefix": force_prefix,
                    "response": full_response,
                    "response_chars": len(full_response),
                    "n_resp_tokens": total_ntok,
                    "score": float(res["score"]),
                    "acc": bool(res["acc"]),
                    "pred": str(res["pred"]),
                    "mode": classify_mode(full_response),
                })

        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[f1v]   wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
