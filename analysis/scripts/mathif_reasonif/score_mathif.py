#!/usr/bin/env python
"""Score MathIF inference outputs with math correctness + instruction following.

Input is the JSONL produced by eval/scripts/run_inference.py or any compatible
JSONL containing prompt/response/gold/metadata fields.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "eval"))

from utils import (  # noqa: E402
    coerce_dict,
    evaluate_following,
    example_id,
    extract_constraints,
    gold_answer,
    infer_checkpoint_step,
    opening_mode_for,
    prompt_text,
    read_records,
    response_text,
    sample_id,
    segment_trajectory,
    word_count,
    write_jsonl,
)
from src.metrics.math_metrics import score_response as math_score  # noqa: E402


def score_record(r: Dict[str, Any], idx: int) -> Dict[str, Any]:
    md = coerce_dict(r.get("metadata"))
    resp = response_text(r)
    answer_type = str(md.get("answer_type") or r.get("answer_type") or "")
    ms = math_score(resp, gold=gold_answer(r), answer_type=answer_type)
    constraints = extract_constraints(r)
    fs = evaluate_following(resp, constraints)
    acc = int(ms.get("correct", 0))
    follow_strict = fs.get("follow_strict")
    joint = int(acc == 1 and follow_strict == 1.0)
    segs = segment_trajectory(resp)
    out = {
        "run_id": r.get("run_id", ""),
        "model_id": r.get("model_id", ""),
        "model_path": r.get("model_path") or r.get("checkpoint") or "",
        "checkpoint_step": infer_checkpoint_step(r),
        "benchmark": r.get("benchmark", "mathif"),
        "example_id": example_id(r, idx),
        "sample_id": sample_id(r, idx),
        "prompt": prompt_text(r),
        "response": resp,
        "gold": gold_answer(r),
        "instruction_type": md.get("constraint_name") or md.get("instruction_type") or "",
        "constraint_desc": md.get("constraint_desc") or md.get("instruction") or "",
        "answer_type": answer_type,
        "response_length_chars": len(resp),
        "response_length_words": word_count(resp),
        "num_segments": len(segs),
        "opening_mode": opening_mode_for(r, benchmark="mathif"),
        "acc": acc,
        "extracted_answer": ms.get("extracted_answer"),
        "extraction_method": ms.get("extraction_method"),
        "follow_soft": fs.get("follow_soft"),
        "follow_strict": fs.get("follow_strict"),
        "joint_acc_follow": joint,
        "num_constraints": fs.get("num_constraints"),
        "num_supported_constraints": fs.get("num_supported_constraints"),
        "num_passed_constraints": fs.get("num_passed_constraints"),
        "per_constraint": fs.get("per_constraint"),
        "metadata": md,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Inference JSONL/JSON/Parquet")
    ap.add_argument("--output", required=True, help="Scored response-level JSONL")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    scored = [score_record(r, i) for i, r in enumerate(records)]
    n = write_jsonl(args.output, scored)
    print(json.dumps({"input": args.input, "output": args.output, "records": n}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
