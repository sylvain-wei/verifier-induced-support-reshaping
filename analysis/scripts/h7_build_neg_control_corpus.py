#!/usr/bin/env python
"""h7 — Build negative-control SFT corpora for WAVE 5.

Two corpora are built, both same size as the WAVE 5 dose=50 DRI corpus, both
correct-only, drawn from the *same* underlying math rollout pool but with
mode constraint differing:

  (1) c1_sft_corpus_DAI.parquet  — correct DAI-mode responses only
       Source: b4r1 step_20 (995 correct DAI samples, 63 unique prompts)
              + b2r1 step_100 (104 correct DAI, 16 unique prompts)
              + b2r1 step_60 (42 correct DAI, 5 unique prompts)
              union by prompt_id, take shortest correct DAI per prompt.

  (2) c1_sft_corpus_random.parquet — correct responses with NO mode filter
       (i.e. mixture of DRI / DAI / Other in roughly natural proportions)
       Source: same as DRI corpus (b1r1 step_0), all 2048 correct samples.
       Sampled to match dose size; deterministic seed=0.

The two control corpora and the original DRI corpus together form a 3-way
ablation: only the SFT *content* differs.

Outputs:
  analysis/data/c1_sft_corpus_DAI.parquet (messages format)
  analysis/data/c1_sft_corpus_DAI_flat.parquet
  analysis/data/c1_sft_corpus_random.parquet
  analysis/data/c1_sft_corpus_random_flat.parquet
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, f"{ROOT}/analysis/scripts")
from classify_opening_modes import classify_mode  # noqa: E402


# Pools where DAI samples are abundant.
DAI_SOURCES = [
    f"{ROOT}/rollout/math_support_probe/b4r1_Qwen3-8B-Base_math7.5k_if2math_local_H20/step_20.jsonl",
    f"{ROOT}/rollout/math_support_probe/b2r1_Qwen3-8B-Base_IFTrain_local_H20/step_100.jsonl",
    f"{ROOT}/rollout/math_support_probe/b2r1_Qwen3-8B-Base_IFTrain_local_H20/step_60.jsonl",
]

# Pool that the DRI corpus came from.
DRI_SOURCE = f"{ROOT}/rollout/math_support_probe/b1r1_Qwen3-8B-Base_math7.5k_local_H20/step_0.jsonl"


def load_correct_by_mode(path, mode_filter, by_pid):
    """Append (length, response, user_content) triples to by_pid[pid][mode]."""
    n_loaded = 0
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            resp = o.get("output", "")
            if not resp: continue
            acc = bool(o.get("acc", False)) or (float(o.get("score", 0) or 0) >= 1.0)
            if not acc: continue
            m = classify_mode(resp)
            if mode_filter is not None and m != mode_filter: continue
            pid = o.get("prompt_id", "")
            user_content = o.get("input", "")
            # `input` includes the bare-template prompt envelope; strip if present
            # so we get just the math problem prompt back.
            if "user\n" in user_content and "\nassistant" in user_content:
                # grab between "user\n" and "\nassistant"
                start = user_content.index("user\n") + len("user\n")
                end = user_content.index("\nassistant", start)
                user_content = user_content[start:end]
            by_pid[pid][m].append((len(resp), resp, user_content))
            n_loaded += 1
    return n_loaded


def build_dai_corpus(target_n: int, output_path: str):
    """Build DAI-only corpus from DAI_SOURCES; pick shortest correct DAI per prompt."""
    by_pid = defaultdict(lambda: defaultdict(list))
    total_loaded = 0
    for src in DAI_SOURCES:
        n = load_correct_by_mode(src, "DAI", by_pid)
        print(f"  loaded {n:5d} correct DAI from {os.path.basename(os.path.dirname(src))}/{os.path.basename(src)}")
        total_loaded += n

    # Per prompt, pick shortest correct DAI response.
    pool = []
    for pid, modes in by_pid.items():
        if "DAI" in modes and modes["DAI"]:
            modes["DAI"].sort()  # by length
            length, resp, user = modes["DAI"][0]
            pool.append({"prompt_id": pid, "user_content": user, "response": resp})
    print(f"  -> {len(pool)} unique prompts with correct DAI (target_n={target_n})")
    if len(pool) < target_n:
        print(f"  WARN: only {len(pool)} unique prompts available, less than target {target_n}")
        # We allow it; the SFT can still run on the smaller corpus.
    # Sort deterministically by prompt_id, take first target_n.
    pool.sort(key=lambda d: d["prompt_id"])
    pool = pool[:target_n]
    write_messages_parquet(pool, output_path)
    return len(pool)


def build_random_corpus(target_n: int, output_path: str, seed: int = 0):
    """Build random correct corpus from DRI_SOURCE — no mode filter, all correct."""
    by_pid = defaultdict(lambda: defaultdict(list))
    n = load_correct_by_mode(DRI_SOURCE, None, by_pid)
    print(f"  loaded {n} correct (any mode) from {os.path.basename(DRI_SOURCE)}")

    # Per prompt, pick *shortest correct response across any mode*.
    # This biases toward shorter responses but is mode-agnostic.
    pool = []
    mode_counts = {"DRI": 0, "DAI": 0, "Other": 0, "CSI": 0}
    for pid, modes in by_pid.items():
        all_resps = []
        for m, lst in modes.items():
            for length, resp, user in lst:
                all_resps.append((length, resp, user, m))
        if not all_resps: continue
        all_resps.sort()
        length, resp, user, m = all_resps[0]
        pool.append({"prompt_id": pid, "user_content": user, "response": resp, "mode": m})
        mode_counts[m] += 1
    print(f"  -> {len(pool)} unique prompts available")
    print(f"  shortest-response mode distribution: {mode_counts}")

    # Deterministic sample of target_n via seed-shuffle.
    import random
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:target_n]

    # Recount mode in chosen subset.
    chosen_modes = defaultdict(int)
    for d in pool: chosen_modes[d.get("mode", "?")] += 1
    print(f"  chosen-{target_n} mode distribution: {dict(chosen_modes)}")

    # Drop the 'mode' helper field before writing.
    for d in pool: d.pop("mode", None)
    write_messages_parquet(pool, output_path)
    return len(pool)


def write_messages_parquet(pool, out_path):
    rows = []
    for d in pool:
        rows.append({
            "messages": [
                {"role": "user", "content": d["user_content"]},
                {"role": "assistant", "content": d["response"]},
            ],
            "prompt_id": d["prompt_id"],
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} rows)")
    # Also write flat parquet (prompt_text / response_text).
    flat_rows = []
    for r in rows:
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        flat_rows.append({"prompt_text": u, "response_text": a})
    flat_df = pd.DataFrame(flat_rows)
    flat_path = out_path.replace(".parquet", "_flat.parquet")
    flat_df.to_parquet(flat_path, index=False)
    print(f"  wrote {flat_path} ({len(flat_df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_n", type=int, default=50,
                    help="Number of unique prompts per corpus (matches dose=50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=f"{ROOT}/analysis/data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n=== Building DAI corpus (target_n={args.target_n}) ===")
    dai_path = os.path.join(args.out_dir, "c1_sft_corpus_DAI.parquet")
    n_dai = build_dai_corpus(args.target_n, dai_path)

    print(f"\n=== Building random-correct corpus (target_n={args.target_n}, seed={args.seed}) ===")
    rnd_path = os.path.join(args.out_dir, "c1_sft_corpus_random.parquet")
    n_rnd = build_random_corpus(args.target_n, rnd_path, seed=args.seed)

    print()
    print(f"Summary: DAI={n_dai}, random={n_rnd}")


if __name__ == "__main__":
    main()
