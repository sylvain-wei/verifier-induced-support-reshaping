#!/usr/bin/env python
"""eval_if_offline.py — offline IFEval-test + IFBench evaluator for E3 ckpts.

Why offline
-----------
The E3 training-time val list was reduced to math-suite-only (AIME24/25 +
MATH-500-128) after the 5-parquet val pass triggered NCCL ALLGATHER timeouts
at step 25 across step_40/60/80 runs (see analysis/scripts/opd_e3/logs/
CRASH_DETECTED_*.txt). IFEval-test (541) and IFBench (300) are evaluated
here, after each E3 training run finishes, with the same val sampling
config (n=16, T=0.7, top_p=0.95, max_tokens=8192) so results are
commensurable with E3's training-time math val numbers.

What it does
------------
For one (run_label, ckpt_path) pair:
  1. Load ckpt with vLLM (TP=8 on the local 8×H20 96GB node).
  2. Generate n=16 responses for each IFEval-test prompt (541 × 16 = 8 656)
     and each IFBench prompt (300 × 16 = 4 800).
  3. Score every response via the E3 dispatcher
     (analysis/scripts/opd_e3/opd_e3_reward_func.py → IFEval verifier).
  4. Aggregate per (run, step, dataset) and append rows to
     analysis/tables/e3_val_metrics.csv with the exact same column schema
     used by watch_e3_progress.py — so existing watcher reads / findings
     docs pick up offline results without code changes.
  5. Also write per-prompt JSONL to rollout/opd_e3_offline_eval/<run>_step<N>/
     for downstream support analysis.

Usage
-----
  # one ckpt
  python3 analysis/scripts/opd_e3/eval_if_offline.py \\
      --run step20 --step 100 \\
      --ckpt opd-lab/outputs/baseline_opd_topk_reverse_kl_k16_tch_huggingface_20260522_141515/global_step_100/actor/huggingface

  # batch (one entry per run, evaluates all four ckpts step 25/50/75/100)
  bash analysis/scripts/opd_e3/eval_if_offline.sh
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys

# vLLM 'spawn' is mandatory after CUDA init.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pyarrow.parquet as pq  # noqa: E402

ROOT = os.environ.get("PROJECT_ROOT", ".")
CSV_PATH = f"{ROOT}/analysis/tables/e3_val_metrics.csv"
OUT_DIR_BASE = f"{ROOT}/rollout/opd_e3_offline_eval"

# Same val sampling kwargs used by training-time val, except max_tokens
# (8192 vs 31744): IF responses very rarely benefit from > 8k tokens, and
# the lower cap gives us a wider safety margin against generation-tail
# imbalance even in offline mode.
DEFAULT_N = 16
DEFAULT_T = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 8192

# Make the dispatcher importable.
sys.path.insert(0, f"{ROOT}/analysis/scripts/opd_e3")
import opd_e3_reward_func  # noqa: E402

CSV_HEADER = [
    "run", "global_step", "dataset",
    "acc_mean@16", "acc_best@16", "acc_maj@16", "acc_worst@16",
    "score_mean@16", "logged_at_utc",
]


def _utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _load_prompts(parquet_path: str) -> tuple[list[str], list]:
    """Return (prompt_strings, ground_truths) from an OPD-format val parquet."""
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    prompts: list[str] = []
    gts = []
    for r in rows:
        msgs = r["prompt"]
        # Mirror what verl's RLHFDataset does: build a chat-template-ish
        # `user\n<content>\nassistant\n` since our base / OPD students were
        # trained without a chat template.
        user_text = ""
        for m in msgs:
            if m["role"] == "user":
                user_text = m["content"]
                break
        prompts.append(f"user\n{user_text}\nassistant\n")
        gts.append(r["reward_model"]["ground_truth"])
    return prompts, gts


def _score(dataset_tag: str, response: str, ground_truth) -> dict:
    return opd_e3_reward_func.reward_func(
        dataset_tag, response, ground_truth, extra_info=None,
    )


def _stat(per_prompt: list[list[float]]):
    """For each list of n acc values per prompt, compute pool-level
    mean@n / best@n / worst@n / maj@n. Each prompt contributes one number
    per stat; we then average across prompts. Matches the verl _validate
    aggregation."""
    n_prompts = len(per_prompt)
    if n_prompts == 0:
        return {"mean": 0.0, "best": 0.0, "worst": 0.0, "maj": 0.0}
    means = [statistics.mean(xs) for xs in per_prompt]
    bests = [max(xs) for xs in per_prompt]
    worsts = [min(xs) for xs in per_prompt]
    majs = [1.0 if statistics.mean(xs) > 0.5 else 0.0 for xs in per_prompt]
    return {
        "mean": statistics.mean(means),
        "best": statistics.mean(bests),
        "worst": statistics.mean(worsts),
        "maj": statistics.mean(majs),
    }


def _ensure_csv():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)


def _append_csv_row(run: str, step: int, dataset: str, stats: dict, score_mean: float):
    """Idempotent: skip if (run, step, dataset) already present **with real data**.

    A row with an empty `acc_mean@16` field is a placeholder left by the
    in-line watcher when training-time val didn't cover this dataset (e.g.
    E3 reruns used math-only val_files; ifeval_test/ifbench_test rows for
    step40/60 were written by the watcher before the watcher patch and have
    empty values). Treating those as "already recorded" would prevent the
    offline eval from filling them in. Real data rows have a non-empty
    `acc_mean@16` field.
    """
    seen = set()
    placeholder_rows = []  # (run, step, dataset) keys with empty mean@16
    all_rows = []
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                all_rows.append(r)
                if r.get("acc_mean@16"):
                    seen.add((r["run"], r["global_step"], r["dataset"]))
                else:
                    placeholder_rows.append((r["run"], r["global_step"], r["dataset"]))

    key = (run, str(step), dataset)
    if key in seen:
        print(f"  [csv] skip (already recorded): {key}")
        return

    # If a placeholder row exists for this key, rewrite the CSV without it
    # before appending. This avoids ending up with both an empty row and a
    # populated row for the same (run, step, dataset).
    if key in placeholder_rows:
        kept = [r for r in all_rows if (r["run"], r["global_step"], r["dataset"]) != key
                or r.get("acc_mean@16")]
        fields = list(all_rows[0].keys()) if all_rows else CSV_HEADER
        with open(CSV_PATH, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print(f"  [csv] removed {len(all_rows) - len(kept)} placeholder row(s) for {key}")

    with open(CSV_PATH, "a", newline="") as fh:
        csv.writer(fh).writerow([
            run, step, dataset,
            f"{stats['mean']:.6f}",
            f"{stats['best']:.6f}",
            f"{stats['maj']:.6f}",
            f"{stats['worst']:.6f}",
            f"{score_mean:.6f}",
            _utcnow(),
        ])
    print(f"  [csv] appended {key}: mean@{DEFAULT_N}={stats['mean']:.4f}")


def evaluate_one(llm, dataset_tag: str, parquet: str, n: int, sp_kwargs: dict,
                 dump_path: str | None) -> tuple[dict, float]:
    """Generate + score for one dataset. Returns (acc stats dict, mean score)."""
    from vllm import SamplingParams
    prompts, gts = _load_prompts(parquet)
    print(f"  [{dataset_tag}] {len(prompts)} prompts × n={n}")

    params = SamplingParams(
        n=n,
        temperature=sp_kwargs["temperature"],
        top_p=sp_kwargs["top_p"],
        top_k=-1,
        max_tokens=sp_kwargs["max_tokens"],
        seed=1234,
    )
    outs = llm.generate(prompts, params)

    per_prompt_acc: list[list[float]] = []
    per_prompt_score: list[list[float]] = []
    if dump_path is not None:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        dump_fh = open(dump_path, "w")
    else:
        dump_fh = None

    for prompt_idx, out in enumerate(outs):
        accs: list[float] = []
        scores: list[float] = []
        for sample_idx, sample in enumerate(out.outputs):
            response = sample.text
            res = _score(dataset_tag, response, gts[prompt_idx])
            accs.append(1.0 if res["acc"] else 0.0)
            scores.append(float(res["score"]))
            if dump_fh is not None:
                dump_fh.write(json.dumps({
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "response": response,
                    "score": res["score"],
                    "acc": res["acc"],
                    "num_constraints": res.get("num_constraints"),
                    "num_satisfied": res.get("num_satisfied"),
                }, ensure_ascii=False) + "\n")
        per_prompt_acc.append(accs)
        per_prompt_score.append(scores)

    if dump_fh is not None:
        dump_fh.close()

    acc_stats = _stat(per_prompt_acc)
    mean_score = statistics.mean([statistics.mean(xs) for xs in per_prompt_score]) \
                 if per_prompt_score else 0.0
    print(
        f"  [{dataset_tag}] acc mean@{n}={acc_stats['mean']:.4f} "
        f"best@{n}={acc_stats['best']:.4f} maj@{n}={acc_stats['maj']:.4f}"
    )
    return acc_stats, mean_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Run label (e.g. step20, step40, step60, step80)")
    ap.add_argument("--step", type=int, required=True,
                    help="Global step number this ckpt corresponds to")
    ap.add_argument("--ckpt", required=True,
                    help="HF-format ckpt directory")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--temperature", type=float, default=DEFAULT_T)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--datasets",
                    default="ifeval_test,ifbench_test",
                    help="Comma-separated subset of {ifeval_test,ifbench_test}")
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt) or not os.path.isfile(f"{args.ckpt}/config.json"):
        sys.exit(f"ckpt missing or not HF-format: {args.ckpt}")

    parquet_map = {
        "ifeval_test": f"{ROOT}/data/ifeval/ifeval_test_opd_val.parquet",
        "ifbench_test": f"{ROOT}/data/ifbench/ifbench_test_opd_val.parquet",
    }
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in parquet_map:
            sys.exit(f"unknown dataset: {d}")
        if not os.path.isfile(parquet_map[d]):
            sys.exit(f"missing val parquet for {d}: {parquet_map[d]}")

    _ensure_csv()
    print(f"[eval_if_offline] run={args.run} step={args.step}")
    print(f"  ckpt={args.ckpt}")
    print(f"  datasets={datasets}")
    print(f"  sampling: n={args.n} T={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens} tp={args.tp}")

    from vllm import LLM
    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=args.max_tokens + 2048,  # max prompt slack
        seed=1234,
    )

    sp_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    out_run_dir = f"{OUT_DIR_BASE}/{args.run}_step{args.step}"
    for d in datasets:
        dump_path = f"{out_run_dir}/{d}_responses.jsonl"
        acc_stats, mean_score = evaluate_one(
            llm, d, parquet_map[d], args.n, sp_kwargs, dump_path,
        )
        _append_csv_row(args.run, args.step, d, acc_stats, mean_score)

    print(f"[eval_if_offline] done. Per-prompt dumps under {out_run_dir}")


if __name__ == "__main__":
    main()
