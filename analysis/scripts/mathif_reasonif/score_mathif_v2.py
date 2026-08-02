#!/usr/bin/env python
"""MathIF v2 scorer: math correctness + per-constraint following + cross-tab.

Headline question: does IF-RLVR actually transfer to math-domain instruction
following, or does it stop at the surface tricks learned on IFEval/IFBench?

For every rollout we save:
  - acc, extracted_answer
  - per-constraint pass[k] using the official MathIF constraint_name routing,
    delegating to vendored IFEval verifier when possible (so the constraint
    classes that overlap with IFEval — keywords/forbidden_words, repeat_prompt,
    multiple_sections, … — are checked exactly the same way as in the paper's
    F1/F3 results).
  - n_constraints (1/2/3), source (gsm8k/math500/minerva/olympiad/aime),
    constraint_category buckets, length, opening_mode.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.metrics.math_metrics import score_response as math_score  # noqa: E402

# Try to use the vendored IFEval verifier for constraint checks. Some of MathIF's
# constraints (e.g. keywords:frequency, detectable_format:number_highlighted_sections,
# combination:repeat_prompt) reuse the IFEval taxonomy exactly, so we can route
# directly to the existing rule-based verifier. Anything not in the registry
# falls back to a conservative regex check.
try:
    from src.metrics.if_verifier.instructions_registry import INSTRUCTION_DICT  # noqa: E402
except Exception:  # pragma: no cover
    INSTRUCTION_DICT = {}


_FORBIDDEN_PATTERNS = ("forbidden_words", "no_comma", "english_lowercase", "english_capital")


def _coerce_args_dict(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    return {}


def _vendored_check(name: str, args: Dict[str, Any], response: str, prompt: str) -> Optional[bool]:
    cls = INSTRUCTION_DICT.get(name)
    if cls is None:
        return None
    try:
        inst = cls("mathif")
    except Exception:
        return None
    try:
        # Some checkers want explicit kwargs; pass whatever args carries plus a
        # reasonable default for prompt_to_repeat which often references the
        # original question.
        kw = dict(args or {})
        if name == "combination:repeat_prompt" and "prompt_to_repeat" not in kw:
            kw["prompt_to_repeat"] = prompt
        inst.build_description(**kw)
    except Exception:
        try:
            inst.build_description()
        except Exception:
            return None
    try:
        return bool(inst.check_following(response or ""))
    except Exception:
        return None


def _fallback_check(name: str, args: Dict[str, Any], response: str) -> Optional[bool]:
    text = response or ""
    low = text.lower()
    n = name.lower()
    args = args or {}
    if "no_comma" in n:
        return ("," not in text) and ("，" not in text)
    if "english_lowercase" in n:
        letters = re.findall(r"[A-Za-z]", text)
        return bool(letters) and all(c.islower() for c in letters)
    if "english_capital" in n:
        letters = re.findall(r"[A-Za-z]", text)
        return bool(letters) and all(c.isupper() for c in letters)
    if "forbidden_words" in n:
        words = args.get("forbidden_words") or []
        return all(w.lower() not in low for w in words)
    if "keywords:existence" in n:
        words = args.get("keywords") or []
        return all(w.lower() in low for w in words)
    if "keywords:frequency" in n:
        kw = args.get("keyword") or args.get("word")
        if not kw:
            return None
        try:
            relation = (args.get("relation") or "at least").lower()
            freq = int(args.get("frequency") or args.get("frequency_count") or 1)
            count = len(re.findall(rf"\b{re.escape(str(kw))}\b", text, re.IGNORECASE))
            if "at least" in relation or "no less" in relation:
                return count >= freq
            if "less than" in relation:
                return count < freq
            if "at most" in relation or "no more" in relation:
                return count <= freq
            if "exact" in relation:
                return count == freq
            return count >= freq
        except Exception:
            return None
    if "number_words" in n:
        wc = word_count(text)
        try:
            num = int(args.get("num_words") or args.get("N") or 200)
            relation = (args.get("relation") or "at most").lower()
            if "at most" in relation or "less" in relation:
                return wc <= num
            if "at least" in relation or "more" in relation:
                return wc >= num
            if "exact" in relation:
                return wc == num
            return wc <= num
        except Exception:
            return None
    if "number_highlighted_sections" in n:
        try:
            need = int(args.get("num_highlights") or args.get("N") or 1)
            count = len(re.findall(r"\*[^*\n]+\*", text)) + len(re.findall(r"_[^_\n]+_", text))
            return count >= need
        except Exception:
            return None
    if "number_bullet_lists" in n:
        try:
            need = int(args.get("num_bullets") or args.get("N") or 1)
            count = len(re.findall(r"(?:^|\n)\s*[\*\-]\s+", text))
            return count >= need
        except Exception:
            return None
    if "multiple_sections" in n:
        try:
            need = int(args.get("num_sections") or args.get("N") or 2)
            marker = args.get("section_spliter") or args.get("section_marker") or "Section"
            count = len(re.findall(rf"{re.escape(str(marker))}\s*\d", text))
            return count >= need
        except Exception:
            return None
    if "repeat_prompt" in n:
        target = (args.get("prompt_to_repeat") or "").strip()
        if not target:
            return None
        # prompt should appear at the very start
        return text.lstrip().startswith(target[: min(120, len(target))])
    if "quotation" in n:
        s = text.strip()
        return (s.startswith('"') and s.endswith('"')) or (s.startswith('\u201c') and s.endswith('\u201d'))
    if "end_checker" in n:
        target = (args.get("end_phrase") or args.get("ending") or "").strip()
        if not target:
            return None
        return text.rstrip().endswith(target)
    if "capital_word_frequency" in n:
        cap_words = re.findall(r"\b[A-Z]{2,}\b", text)
        try:
            need = int(args.get("capital_frequency") or args.get("N") or 1)
            relation = (args.get("capital_relation") or "at least").lower()
            if "at least" in relation:
                return len(cap_words) >= need
            if "at most" in relation:
                return len(cap_words) <= need
            return len(cap_words) >= need
        except Exception:
            return None
    if "response_language" in n:
        lang = (args.get("language") or "").lower()
        if not lang:
            return None
        cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        kana = bool(re.search(r"[\u3040-\u30ff]", text))
        cyr = bool(re.search(r"[\u0400-\u04ff]", text))
        if "chinese" in lang:
            return cjk
        if "japanese" in lang:
            return cjk or kana
        if "russian" in lang:
            return cyr
        if "english" in lang:
            return not (cjk or kana or cyr)
        return None
    return None


def evaluate_constraint(name: str, args: Dict[str, Any], response: str, prompt: str) -> Dict[str, Any]:
    args = _coerce_args_dict(args)
    rv = _vendored_check(name, args, response, prompt)
    if rv is not None:
        return {"name": name, "category": name.split(":", 1)[0], "supported": True, "passed": bool(rv), "source": "vendored"}
    fb = _fallback_check(name, args, response)
    if fb is None:
        return {"name": name, "category": name.split(":", 1)[0], "supported": False, "passed": None, "source": "unsupported"}
    return {"name": name, "category": name.split(":", 1)[0], "supported": True, "passed": bool(fb), "source": "fallback"}


def _load_raw_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the merged MathIF JSONL and return id->record map.

    Our processed records have ids of the form `mathif_<original-id>` (e.g.
    `mathif_aime-double-0`); we key the raw rows by both the original id and
    that prefixed form for safety.
    """
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        oid = str(r.get("id") or "")
        if oid:
            out[oid] = r
            out[f"mathif_{oid}"] = r
    return out


def score_record(r: Dict[str, Any], idx: int, raw_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    md = coerce_dict(r.get("metadata"))
    eid = example_id(r, idx)
    raw = raw_map.get(eid) or raw_map.get(eid.replace("mathif_", "")) or {}
    cn_raw = raw.get("constraint_name") or md.get("constraint_name")
    ca_raw = raw.get("constraint_args") or md.get("constraint_args")
    cn = coerce_list(cn_raw)
    ca = coerce_list(ca_raw)
    while len(ca) < len(cn):
        ca.append({})
    cn = [str(x) for x in cn]
    ca = [_coerce_args_dict(x) if x is not None else {} for x in ca]

    resp = response_text(r)
    pr = prompt_text(r)
    answer_type = str(md.get("answer_type") or "latex_boxed")
    gold = str(raw.get("answer") or gold_answer(r))
    ms = math_score(resp, gold=gold, answer_type=answer_type)
    per = [evaluate_constraint(n, a, resp, pr) for n, a in zip(cn, ca)]
    supported = [p for p in per if p.get("supported")]
    if supported:
        n_ok = sum(1 for p in supported if p.get("passed"))
        follow_strict = 1.0 if n_ok == len(supported) else 0.0
        follow_soft = n_ok / len(supported)
    else:
        follow_strict = follow_soft = float("nan")

    acc = int(ms.get("correct", 0))
    joint = int(acc == 1 and follow_strict == 1.0)
    om = opening_mode_for(r, benchmark="mathif")
    n_constraints = len(per)

    return {
        "run_id": r.get("run_id", ""),
        "model_id": r.get("model_id", ""),
        "checkpoint_step": infer_checkpoint_step(r),
        "benchmark": "mathif",
        "example_id": example_id(r, idx),
        "sample_id": sample_id(r, idx),
        "source": str(raw.get("source") or md.get("source") or ""),
        "source_file": str(raw.get("source_file") or ""),
        "n_constraints": n_constraints,
        "constraint_names": cn,
        "constraint_categories": [c.split(":", 1)[0] for c in cn],
        "opening_mode": om,
        "response_length_chars": len(resp),
        "response_length_words": word_count(resp),
        "acc": acc,
        "extracted_answer": ms.get("extracted_answer"),
        "extraction_method": ms.get("extraction_method"),
        "follow_soft": follow_soft,
        "follow_strict": follow_strict,
        "joint_acc_follow": joint,
        "num_constraints": len(per),
        "num_supported_constraints": len(supported),
        "num_passed_constraints": sum(1 for p in supported if p.get("passed")),
        "per_constraint": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-dataset", default=str(ROOT / "data" / "mathif" / "mathif.jsonl"))
    args = ap.parse_args()
    raw_map = _load_raw_dataset(Path(args.raw_dataset))
    records = read_records(args.input)
    out = [score_record(r, i, raw_map) for i, r in enumerate(records)]
    n = write_jsonl(args.output, out)
    print(json.dumps({"input": args.input, "output": args.output, "records": n, "raw_records": len(raw_map)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
