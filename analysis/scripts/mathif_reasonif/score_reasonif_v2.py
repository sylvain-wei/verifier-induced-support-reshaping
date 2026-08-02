#!/usr/bin/env python
"""ReasonIF v2 scorer with reasoning-trace vs response-level following.

Headline question: did the model learn to *follow during reasoning*, or did it
shortcut the constraint by skipping reasoning altogether?

For each rollout we save:
  - opening_mode (DRI/DAI/CSI/Other)
  - wrapper_present  (did the model emit <answer>...</answer>?)
  - reasoning_strict / response_strict / reasoning_soft / response_soft
  - per-constraint pass for both reasoning and response halves
  - reasoning_word_count, response_word_count
  - answer correctness (when gold available, using <answer> block)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "eval"))

from utils import (  # noqa: E402
    coerce_dict,
    coerce_list,
    example_id,
    gold_answer,
    infer_checkpoint_step,
    opening_mode_for,
    prompt_text,
    read_records,
    response_text,
    sample_id,
    write_jsonl,
)
from reasonif_verifier import evaluate_record, split_reasoning_and_answer  # noqa: E402
from src.metrics.math_metrics import is_equiv, normalize_answer  # noqa: E402


def _extract_answer(answer_block: str) -> str:
    if not answer_block:
        return ""
    m = re.findall(r"-?\d+(?:\.\d+)?", answer_block)
    if m:
        return m[-1]
    return answer_block.strip()


def _load_raw_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the official ReasonIF JSONL and return id->record map.

    The dataset stores 300 reasoning prompts; we key them by their ordinal id
    (matches our processed `reasonif_{i:04d}` ids) and also by hf_id when
    available.
    """
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(rows):
        rid = f"reasonif_{i:04d}"
        out[rid] = r
        hf = r.get("hf_id")
        if hf:
            out[str(hf)] = r
    return out


def score_record(r: Dict[str, Any], idx: int, raw_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    md = coerce_dict(r.get("metadata"))
    eid = example_id(r, idx)
    raw = raw_map.get(eid) or {}
    if not raw:
        # fallback: maybe processed id has different prefix
        raw = raw_map.get(eid.replace("reasonif_", "")) or {}
    cn = coerce_list(raw.get("constraint_name") or md.get("constraint_name"))
    ca = coerce_list(raw.get("constraint_args") or md.get("constraint_args"))
    while len(ca) < len(cn):
        ca.append({})
    cn = [str(x) for x in cn]
    ca = [coerce_dict(x) if x is not None else {} for x in ca]

    resp = response_text(r)
    reasoning, answer_block, wrapper = split_reasoning_and_answer(resp)
    fs = evaluate_record(resp, cn, ca)
    gold = str(raw.get("answer") or gold_answer(r))
    answer_type = "numeric" if str(raw.get("source", "")).lower() in {"gsm8k", "amc"} else "latex_boxed"

    # Score answer correctness from <answer>...</answer> block if present, else fall back
    extracted = _extract_answer(answer_block) if wrapper else ""
    correct = None
    if gold:
        try:
            ok = is_equiv(extracted, str(gold), answer_type=answer_type)
            if not ok:
                # secondary: try last number anywhere in the response
                m = re.findall(r"-?\d+(?:\.\d+)?", resp)
                if m:
                    ok = is_equiv(m[-1], str(gold), answer_type=answer_type)
            correct = int(bool(ok))
        except Exception:
            correct = None

    om = opening_mode_for(r, benchmark="reasonif")
    rsn_strict = fs.get("reasoning_strict_pass")
    res_strict = fs.get("response_strict_pass")
    joint_reasoning = (correct == 1 and rsn_strict == 1.0)
    joint_response = (correct == 1 and res_strict == 1.0)

    return {
        "run_id": r.get("run_id", ""),
        "model_id": r.get("model_id", ""),
        "checkpoint_step": infer_checkpoint_step(r),
        "benchmark": "reasonif",
        "example_id": example_id(r, idx),
        "sample_id": sample_id(r, idx),
        "source": str(raw.get("source") or md.get("source") or ""),
        "constraint_names": cn,
        "constraint_categories": [c.split(":", 1)[0] for c in cn],
        "primary_constraint": cn[0] if cn else "",
        "primary_category": cn[0].split(":", 1)[0] if cn else "",
        "opening_mode": om,
        "wrapper_present": fs.get("wrapper_present"),
        "reasoning_word_count": fs.get("reasoning_word_count"),
        "response_word_count": fs.get("response_word_count"),
        "answer_word_count": fs.get("answer_word_count"),
        "reasoning_strict_pass": rsn_strict,
        "response_strict_pass": res_strict,
        "reasoning_soft_pass": fs.get("reasoning_soft_pass"),
        "response_soft_pass": fs.get("response_soft_pass"),
        "skip_reasoning": (not fs.get("wrapper_present")) or (fs.get("reasoning_word_count") < 5),
        "acc": correct,
        "extracted_answer": extracted,
        "joint_reasoning_pass": int(bool(joint_reasoning)) if correct is not None else None,
        "joint_response_pass": int(bool(joint_response)) if correct is not None else None,
        "per_constraint": fs.get("per_constraint"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-dataset", default=str(ROOT / "data" / "reasonif" / "reasonIF_dataset.json"))
    args = ap.parse_args()
    raw_map = _load_raw_dataset(Path(args.raw_dataset))
    records = read_records(args.input)
    out = [score_record(r, i, raw_map) for i, r in enumerate(records)]
    n = write_jsonl(args.output, out)
    print(json.dumps({"input": args.input, "output": args.output, "records": n, "raw_records": len(raw_map)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
