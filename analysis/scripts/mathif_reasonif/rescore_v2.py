#!/usr/bin/env python
"""Rescore MathIF/ReasonIF responses with a richer schema.

Outputs a v2 JSONL with the following extra columns per response:
  - reasoning_trace, final_answer, has_answer_tag
  - num_constraints, constraint_names, constraint_families
  - reasoning_follow_soft, reasoning_follow_strict, reasoning_per_constraint
  - response_follow_soft, response_follow_strict, response_per_constraint
  - reasoning_word_count, reasoning_char_count
  - bypass_reasoning (heuristic)
  - source (math source for MathIF; original source field for ReasonIF)
  - difficulty (single/double/triple for MathIF; "single" for ReasonIF)

The MathIF gold-answer matching also reads the inference jsonl directly so we
do not depend on the older score_mathif outputs.
"""
from __future__ import annotations
import os

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    word_count,
    write_jsonl,
)
from verify_official import (  # noqa: E402
    family_of,
    split_reasoning_and_answer,
    verify_constraints,
)
from src.metrics.math_metrics import score_response as math_score  # noqa: E402


def _decide_difficulty(meta: Dict[str, Any], example_id: str) -> str:
    src_file = str(meta.get("raw_metadata", {}).get("source_file") or "")
    if "single" in src_file:
        return "single"
    if "double" in src_file:
        return "double"
    if "triple" in src_file:
        return "triple"
    if "single" in (example_id or ""):
        return "single"
    if "double" in (example_id or ""):
        return "double"
    if "triple" in (example_id or ""):
        return "triple"
    return "unknown"


# ----------------------- raw dataset join -----------------------

_MATHIF_RAW: Optional[Dict[str, Dict[str, Any]]] = None
_REASONIF_RAW: Optional[Dict[str, Dict[str, Any]]] = None


def _load_mathif_raw() -> Dict[str, Dict[str, Any]]:
    global _MATHIF_RAW
    if _MATHIF_RAW is not None:
        return _MATHIF_RAW
    path = ROOT / "data" / "mathif" / "mathif.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rid = str(r.get("id"))
        out[f"mathif_{rid}"] = r
    _MATHIF_RAW = out
    return out


def _load_reasonif_raw() -> Dict[str, Dict[str, Any]]:
    global _REASONIF_RAW
    if _REASONIF_RAW is not None:
        return _REASONIF_RAW
    path = ROOT / "data" / "reasonif" / "reasonIF_dataset.json"
    text = path.read_text()
    out: Dict[str, Dict[str, Any]] = {}
    buf = ""
    depth = 0
    idx = 0
    for ch in text:
        buf += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = buf.strip()
                if s:
                    r = json.loads(s)
                    rid = f"reasonif_{idx:04d}"
                    out[rid] = r
                    idx += 1
                buf = ""
    _REASONIF_RAW = out
    return out


def _raw_constraints(dataset: str, ex_id: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    if dataset == "mathif":
        raw = _load_mathif_raw().get(ex_id) or {}
    else:
        raw = _load_reasonif_raw().get(ex_id) or {}
    names = raw.get("constraint_name") or []
    if isinstance(names, str):
        names = [names]
    args = raw.get("constraint_args") or []
    if isinstance(args, dict):
        args = [args]
    args_list: List[Dict[str, Any]] = []
    for a in args or []:
        args_list.append(a if isinstance(a, dict) else {})
    while len(args_list) < len(names):
        args_list.append({})
    return list(names), args_list, raw


def _extract_constraints(meta: Dict[str, Any], dataset: str) -> Tuple[List[str], List[Dict[str, Any]], str]:
    """Return (names, args_list, raw_source_field)."""
    raw = meta.get("raw_metadata") or {}
    if not isinstance(raw, dict):
        raw = {}
    names = raw.get("constraint_name") or meta.get("constraint_name") or []
    if isinstance(names, str):
        names = [names]
    args = raw.get("constraint_args") or meta.get("constraint_args") or []
    if isinstance(args, dict):
        args = [args]
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            try:
                parsed = ast.literal_eval(args)
                args = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                args = []
    args_list: List[Dict[str, Any]] = []
    for a in args or []:
        if isinstance(a, dict):
            args_list.append({k: v for k, v in a.items() if v is not None})
        else:
            args_list.append({})
    while len(args_list) < len(names):
        args_list.append({})
    raw_source = raw.get("source") or meta.get("source") or ""
    return list(names), args_list, str(raw_source)


def _bypass_signal(reasoning_trace: str, has_tag: bool) -> Dict[str, Any]:
    """Did the model bypass reasoning?

    Heuristics: trace shorter than 30 words OR no real reasoning markers.
    """
    wc = word_count(reasoning_trace)
    has_reasoning_words = bool(re.search(
        r"\b(step|first|second|third|then|next|therefore|thus|so|hence|because|consider|let|let's|compute|simplify|substitute)\b",
        reasoning_trace, flags=re.IGNORECASE,
    ))
    bypass = bool(has_tag and (wc < 30 or not has_reasoning_words))
    return {
        "bypass_reasoning": bypass,
        "reasoning_word_count": wc,
        "reasoning_char_count": len(reasoning_trace or ""),
        "has_reasoning_markers": has_reasoning_words,
    }


def rescore_record(r: Dict[str, Any], dataset: str, idx: int) -> Dict[str, Any]:
    md = coerce_dict(r.get("metadata"))
    resp = response_text(r)
    gold = gold_answer(r)
    answer_type = str(md.get("answer_type") or "")

    trace, ans_tag, has_tag = split_reasoning_and_answer(resp)
    body_for_correctness = ans_tag if has_tag and ans_tag else resp
    ms = math_score(body_for_correctness, gold=gold, answer_type=answer_type) if gold else {
        "correct": None, "extracted_answer": "", "extraction_method": "none"
    }

    ex_id = example_id(r, idx)
    names, args_list, raw = _raw_constraints(dataset, ex_id)
    raw_source = str(raw.get("source") or md.get("source") or "")
    families = sorted({family_of(n) for n in names})

    # ReasonIF: constraint targets the reasoning trace
    # MathIF: constraint targets the whole response (per MathIF paper)
    target_text = trace if dataset == "reasonif" else resp
    main = verify_constraints(target_text, names, args_list)
    other_text = resp if dataset == "reasonif" else trace
    other = verify_constraints(other_text, names, args_list)

    if dataset == "reasonif":
        reasoning_per = main["per"]
        response_per = other["per"]
        reasoning_soft = main["follow_soft"]
        reasoning_strict = main["follow_strict"]
        response_soft = other["follow_soft"]
        response_strict = other["follow_strict"]
        n_supp = main["n_supported"]
    else:
        reasoning_per = other["per"]
        response_per = main["per"]
        reasoning_soft = other["follow_soft"]
        reasoning_strict = other["follow_strict"]
        response_soft = main["follow_soft"]
        response_strict = main["follow_strict"]
        n_supp = main["n_supported"]

    bypass = _bypass_signal(trace, has_tag)

    acc = ms.get("correct")
    primary_strict = response_strict if dataset == "mathif" else reasoning_strict
    if acc is not None and not (primary_strict != primary_strict):  # not NaN
        joint = int(acc == 1 and primary_strict == 1.0)
    else:
        joint = None

    out = {
        "run_id": r.get("run_id", ""),
        "model_id": r.get("model_id", ""),
        "model_path": r.get("model_path") or r.get("checkpoint") or "",
        "checkpoint_step": infer_checkpoint_step(r),
        "benchmark": r.get("benchmark", dataset),
        "dataset": dataset,
        "example_id": example_id(r, idx),
        "sample_id": sample_id(r, idx),
        "prompt": prompt_text(r),
        "response": resp,
        "reasoning_trace": trace,
        "final_answer": ans_tag,
        "has_answer_tag": has_tag,
        "gold": gold,
        "extracted_answer": ms.get("extracted_answer"),
        "extraction_method": ms.get("extraction_method"),
        "acc": acc,
        "answer_type": answer_type,
        "source": raw_source,
        "difficulty": _decide_difficulty(md, example_id(r, idx)) if dataset == "mathif" else "single",
        "constraint_names": names,
        "constraint_families": families,
        "num_constraints": len(names),
        "n_supported_constraints": n_supp,
        "reasoning_follow_soft": reasoning_soft,
        "reasoning_follow_strict": reasoning_strict,
        "reasoning_per_constraint": reasoning_per,
        "response_follow_soft": response_soft,
        "response_follow_strict": response_strict,
        "response_per_constraint": response_per,
        "joint_acc_follow": joint,
        "response_length_chars": len(resp),
        "response_length_words": word_count(resp),
        "opening_mode": opening_mode_for(r, benchmark=dataset),
        **bypass,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", required=True, choices=["mathif", "reasonif"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    scored = [rescore_record(r, args.dataset, i) for i, r in enumerate(records)]
    n = write_jsonl(args.output, scored)
    print(json.dumps({"input": args.input, "output": args.output, "records": n}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
