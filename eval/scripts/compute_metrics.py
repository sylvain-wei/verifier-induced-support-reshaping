#!/usr/bin/env python
"""
scripts/compute_metrics.py

Read responses/{run_id}/{model_id}/{benchmark}/{mode}.jsonl and compute:
  1. response-level metrics -> metrics/{run_id}/{model_id}/{benchmark}/{mode}_response_metrics.jsonl
  2. per-prompt (K samples folded) metrics -> metrics/{run_id}/{model_id}/{benchmark}/{mode}_perprompt_metrics.jsonl
  3. a single flat dict for the benchmark -> metrics/{run_id}/aggregate/{benchmark}_{mode}_{model}.json
     that aggregate_metrics.py reads.

Supports both math (answer extraction + K-level) and if (verifier) tasks. The
response jsonl already carries `benchmark`, `metadata.task_type` (if present),
and decoding params — but `task_type` we recover by looking up datasets.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.text_metrics import compute_text_metrics  # noqa: E402
from src.metrics.math_metrics import score_response as math_score, aggregate_over_samples as math_agg  # noqa: E402
from src.metrics.diversity_metrics import compute_diversity  # noqa: E402
from src.metrics.if_metrics import score_if_response, prefix_concentration, first_sentence_entropy  # noqa: E402
from src.metrics.aggregate import aggregate_math_run, aggregate_if_run  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.io import write_jsonl, iter_jsonl, atomic_write_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.tokenization import get_tokenizer  # noqa: E402


def _task_type_for(benchmark: str, datasets_cfg: Dict[str, Any]) -> str:
    return (datasets_cfg.get("benchmarks") or {}).get(benchmark, {}).get("task_type", "math")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--models-yaml", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--datasets-yaml", default=str(ROOT / "configs/datasets.yaml"))
    args = ap.parse_args()

    log = get_logger("compute_metrics", run_id=args.run_id)

    m_cfg = load_yaml(args.models_yaml)
    d_cfg = load_yaml(args.datasets_yaml)

    model_cfg = (m_cfg.get("models") or {})[args.model_id]
    tokenizer = None
    try:
        tokenizer = get_tokenizer(model_cfg.get("tokenizer_path") or model_cfg["path"])
    except Exception as e:
        log.warning(f"Tokenizer not loaded: {e}; falling back to whitespace counts")

    task_type = _task_type_for(args.benchmark, d_cfg)
    resp_path = ROOT / "responses" / args.run_id / args.model_id / args.benchmark / f"{args.mode}.jsonl"
    if not resp_path.exists():
        raise SystemExit(f"Missing responses jsonl: {resp_path}")

    out_dir = ROOT / "metrics" / args.run_id / args.model_id / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_metrics_path = out_dir / f"{args.mode}_response_metrics.jsonl"
    perprompt_path = out_dir / f"{args.mode}_perprompt_metrics.jsonl"

    # Group responses by example_id
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in iter_jsonl(resp_path):
        grouped[r["example_id"]].append(r)
    log.info(f"Loaded {sum(len(v) for v in grouped.values())} responses across {len(grouped)} prompts")

    resp_metrics_records: List[Dict[str, Any]] = []
    perprompt_records: List[Dict[str, Any]] = []

    # Preload IF metadata: we pick the first record's `metadata.verifier_metadata`
    # per example. For math we carry forward `metadata.answer_type`.
    for eid, samples in grouped.items():
        samples = sorted(samples, key=lambda x: int(x.get("sample_id", 0)))
        responses_text = [s.get("response") or "" for s in samples]
        gold = samples[0].get("gold")
        md = samples[0].get("metadata") or {}

        per_sample_records: List[Dict[str, Any]] = []
        for s in samples:
            text = s.get("response") or ""
            tm = compute_text_metrics(text, tokenizer=tokenizer)
            rec = {
                "run_id": s.get("run_id"),
                "model_id": s.get("model_id"),
                "benchmark": s.get("benchmark"),
                "example_id": s.get("example_id"),
                "sample_id": int(s.get("sample_id", 0)),
                "finish_reason": s.get("finish_reason"),
                "gen_tokens": s.get("gen_tokens"),
                **tm,
            }
            if task_type == "math":
                ms = math_score(text, gold=gold, answer_type=md.get("answer_type", ""))
                rec.update(ms)
            else:
                ifs = score_if_response(
                    text,
                    md.get("verifier_metadata") or {},
                    tokenizer=tokenizer,
                    benchmark=args.benchmark,
                )
                # Keep per_constraint nested
                rec.update(ifs)
            per_sample_records.append(rec)
        resp_metrics_records.extend(per_sample_records)

        # per-prompt rollup
        lengths = [r.get("response_length_tokens", 0) for r in per_sample_records]
        if task_type == "math":
            agg = math_agg(per_sample_records)
            pp = {
                "example_id": eid,
                "benchmark": args.benchmark,
                "model_id": args.model_id,
                "mode": args.mode,
                "gold": gold,
                "answer_type": md.get("answer_type", ""),
                **agg,
                "extraction_failure_rate": sum(int(r.get("extraction_failure", False)) for r in per_sample_records) / len(per_sample_records),
                "step_density_mean": _mean([r.get("step_density") for r in per_sample_records]),
                "equation_density_mean": _mean([r.get("equation_density") for r in per_sample_records]),
                "verification_marker_rate_mean": _mean([r.get("verification_marker_rate") for r in per_sample_records]),
                "over_reasoning_marker_rate_mean": _mean([r.get("over_reasoning_marker_rate") for r in per_sample_records]),
                "answer_position_ratio_mean": _mean([r.get("answer_position_ratio") for r in per_sample_records]),
                "response_length_tokens_mean": _mean(lengths),
                "response_length_tokens_std": _std(lengths),
            }
            # Diversity only if K > 1
            div = compute_diversity(responses_text, lengths)
            pp.update(div)
        else:
            # IF: for greedy (K=1) we keep a single sample; for sampling we average.
            def _m(key: str) -> float:
                return _mean([r.get(key) for r in per_sample_records])

            pp = {
                "example_id": eid,
                "benchmark": args.benchmark,
                "model_id": args.model_id,
                "mode": args.mode,
                "strict_prompt_pass": _m("strict_prompt_pass"),
                "instruction_level_pass_rate": _m("instruction_level_pass_rate"),
                "format_pass_rate": _m("format_pass_rate"),
                "compliance_efficiency": _m("compliance_efficiency"),
                "num_constraints_mean": _mean([r.get("num_constraints") for r in per_sample_records]),
                "response_length_tokens": _m("response_length_tokens"),
                "over_reasoning_marker_rate": _m("over_reasoning_marker_rate"),
                "prefix_8_tokens": per_sample_records[0].get("prefix_8_tokens", ""),
                "first_sentence": per_sample_records[0].get("first_sentence", ""),
            }

        perprompt_records.append(pp)

    write_jsonl(resp_metrics_path, resp_metrics_records)
    write_jsonl(perprompt_path, perprompt_records)

    # Roll-up for this (model, benchmark, mode)
    rollup_dir = ROOT / "metrics" / args.run_id / "aggregate"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    rollup_path = rollup_dir / f"{args.benchmark}__{args.mode}__{args.model_id}.json"
    if task_type == "math":
        rollup = aggregate_math_run(perprompt_records)
    else:
        rollup = aggregate_if_run(perprompt_records)
        # Additional corpus-level stats for IF
        prefixes = [r.get("prefix_8_tokens") for r in perprompt_records]
        fs = [r.get("first_sentence") for r in perprompt_records]
        rollup["prefix_concentration_top1"] = prefix_concentration(prefixes, 1)
        rollup["prefix_concentration_top3"] = prefix_concentration(prefixes, 3)
        rollup["first_sentence_entropy"] = first_sentence_entropy(fs)

    rollup.update({
        "run_id": args.run_id,
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "mode": args.mode,
        "task_type": task_type,
    })
    atomic_write_json(rollup_path, rollup)
    log.info(f"Wrote {resp_metrics_path.name}, {perprompt_path.name}, {rollup_path.name}")
    return 0


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _std(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(xs) < 2:
        return 0.0 if xs else float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
