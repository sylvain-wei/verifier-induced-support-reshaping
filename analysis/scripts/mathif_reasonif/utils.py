#!/usr/bin/env python
"""Utilities for MathIF/ReasonIF checkpoint behavior analysis.

The functions here deliberately implement conservative, transparent rules. They
are meant to produce first-pass, auditable signals that can later be calibrated
against the official MathIF/ReasonIF scorers or an LLM judge.
"""
from __future__ import annotations
import os

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
ANALYSIS_SCRIPTS = ROOT / "analysis" / "scripts"
if str(ANALYSIS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPTS))

try:
    from classify_opening_modes import classify_mode
except Exception:  # pragma: no cover - smoke tests can still run without it
    classify_mode = None


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".jsonl":
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    if p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "examples", "records", "dataset"):
                if isinstance(obj.get(key), list):
                    return obj[key]
            return [obj]
    if p.suffix in (".parquet", ".csv", ".tsv"):
        import pandas as pd
        if p.suffix == ".parquet":
            return pd.read_parquet(p).to_dict("records")
        sep = "\t" if p.suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep).to_dict("records")
    raise ValueError(f"Unsupported input format: {p}")


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def coerce_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(x)
                return v if isinstance(v, dict) else {}
            except Exception:
                pass
    return {}


def coerce_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str) and x.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(x)
                return v if isinstance(v, list) else [v]
            except Exception:
                pass
        return [x]
    return [x]


def response_text(r: Dict[str, Any]) -> str:
    return str(r.get("response") or r.get("output") or r.get("prediction") or r.get("completion") or "")


def prompt_text(r: Dict[str, Any]) -> str:
    return str(r.get("prompt") or r.get("question") or r.get("problem") or r.get("input") or "")


def gold_answer(r: Dict[str, Any]) -> str:
    md = coerce_dict(r.get("metadata"))
    return str(r.get("gold") or r.get("answer") or r.get("final_answer") or md.get("answer") or "")


def example_id(r: Dict[str, Any], idx: int = 0) -> str:
    return str(r.get("example_id") or r.get("id") or r.get("prompt_id") or f"ex_{idx:05d}")


def sample_id(r: Dict[str, Any], idx: int = 0) -> int:
    try:
        return int(r.get("sample_id", idx))
    except Exception:
        return idx


def infer_checkpoint_step(r: Dict[str, Any]) -> int:
    for key in ("checkpoint_step", "step", "global_step"):
        if r.get(key) is not None:
            try:
                return int(r[key])
            except Exception:
                pass
    text = " ".join(str(r.get(k, "")) for k in ("checkpoint", "model_path", "model_id"))
    m = re.search(r"(?:global_)?step[_-]?(\d+)", text)
    return int(m.group(1)) if m else -1


def opening_mode_for(r: Dict[str, Any], *, benchmark: Optional[str] = None) -> str:
    if classify_mode is None:
        return "Other"
    md = coerce_dict(r.get("metadata"))
    inst_ids = md.get("instruction_id_list") or md.get("constraints") or md.get("instruction_ids") or []
    bench = benchmark or str(r.get("benchmark") or "")
    enable_csi = bench.lower() in {"ifeval", "ifbench", "mathif", "reasonif"}
    return classify_mode(response_text(r), prompt=prompt_text(r), inst_ids=inst_ids, enable_csi=enable_csi)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def segment_trajectory(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Prefer explicit numbered/step boundaries; otherwise paragraphs; otherwise sentences.
    pieces = re.split(r"(?=\n?\s*(?:Step\s*\d+|\d+[\.)])\s+)", text)
    pieces = [p.strip() for p in pieces if p.strip()]
    if len(pieces) >= 2:
        return pieces
    pieces = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(pieces) >= 2:
        return pieces
    pieces = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if p.strip()]
    return pieces or [text]


def _quoted_terms(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"['\"]([^'\"]{1,80})['\"]", text or "") if m.strip()]


def _numbers(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\b\d+\b", text or "")]


def _classify_constraint(name: str, desc: str, args: Dict[str, Any]) -> str:
    joined = f"{name} {desc} {' '.join(args.keys())}".lower()
    if any(k in joined for k in ("word", "length", "token", "character", "sentence", "paragraph")):
        return "length"
    if any(k in joined for k in ("start", "begin", "prefix", "end", "suffix", "affix")):
        return "affix"
    if any(k in joined for k in ("keyword", "include", "forbidden", "avoid", "do not use", "lexical")):
        return "lexical"
    if any(k in joined for k in ("json", "format", "boxed", "bullet", "markdown", "table", "list", "final answer")):
        return "format"
    if any(k in joined for k in ("language", "english", "chinese", "spanish", "multilingual")):
        return "language"
    return "unknown"


def _extract_limits(desc: str, args: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    key_map = {
        "max_words": "max_words", "maximum_words": "max_words", "word_limit": "max_words",
        "min_words": "min_words", "minimum_words": "min_words", "exact_words": "exact_words",
        "num_words": "exact_words", "n_words": "exact_words", "max_length": "max_chars",
        "min_length": "min_chars", "max_chars": "max_chars", "min_chars": "min_chars",
    }
    for k, v in args.items():
        lk = str(k).lower()
        if lk in key_map:
            try:
                out[key_map[lk]] = int(v)
            except Exception:
                pass
    dl = (desc or "").lower()
    pats = [
        ("max_words", r"(?:at most|no more than|under|less than|within)\s+(\d+)\s+words?"),
        ("min_words", r"(?:at least|more than)\s+(\d+)\s+words?"),
        ("exact_words", r"(?:exactly|in)\s+(\d+)\s+words?"),
        ("max_chars", r"(?:at most|no more than|under|less than|within)\s+(\d+)\s+characters?"),
    ]
    for key, pat in pats:
        m = re.search(pat, dl)
        if m and key not in out:
            out[key] = int(m.group(1))
    return out


def _extract_terms(args: Dict[str, Any], desc: str, *keys: str) -> List[str]:
    terms: List[str] = []
    for key in keys:
        for k, v in args.items():
            if key in str(k).lower():
                vals = coerce_list(v)
                terms.extend(str(x) for x in vals if str(x).strip())
    terms.extend(_quoted_terms(desc))
    # Preserve order but de-duplicate case-insensitively.
    seen = set()
    out = []
    for t in terms:
        s = t.strip()
        if not s:
            continue
        lk = s.lower()
        if lk not in seen:
            seen.add(lk)
            out.append(s)
    return out


def evaluate_single_constraint(response: str, name: str, desc: str, args: Dict[str, Any]) -> Dict[str, Any]:
    category = _classify_constraint(name, desc, args)
    text = response or ""
    low = text.lower()
    joined = f"{name} {desc}".lower()

    if category == "length":
        limits = _extract_limits(desc, args)
        wc = word_count(text)
        cc = len(text)
        checks: List[Tuple[str, bool, str]] = []
        if "max_words" in limits:
            checks.append(("max_words", wc <= limits["max_words"], f"words={wc}, limit={limits['max_words']}"))
        if "min_words" in limits:
            checks.append(("min_words", wc >= limits["min_words"], f"words={wc}, limit={limits['min_words']}"))
        if "exact_words" in limits:
            checks.append(("exact_words", wc == limits["exact_words"], f"words={wc}, target={limits['exact_words']}"))
        if "max_chars" in limits:
            checks.append(("max_chars", cc <= limits["max_chars"], f"chars={cc}, limit={limits['max_chars']}"))
        if "min_chars" in limits:
            checks.append(("min_chars", cc >= limits["min_chars"], f"chars={cc}, limit={limits['min_chars']}"))
        if checks:
            passed = all(x[1] for x in checks)
            return {"supported": True, "passed": passed, "category": category, "rule": "+".join(x[0] for x in checks), "evidence": "; ".join(x[2] for x in checks)}

    if category == "affix":
        prefix = args.get("prefix") or args.get("starts_with") or args.get("start")
        suffix = args.get("suffix") or args.get("ends_with") or args.get("end")
        if prefix is None and ("start" in joined or "begin" in joined):
            qs = _quoted_terms(desc)
            prefix = qs[0] if qs else None
        if suffix is None and "end" in joined:
            qs = _quoted_terms(desc)
            suffix = qs[-1] if qs else None
        checks = []
        if prefix:
            checks.append(("prefix", text.lstrip().startswith(str(prefix)), f"prefix={prefix!r}"))
        if suffix:
            checks.append(("suffix", text.rstrip().endswith(str(suffix)), f"suffix={suffix!r}"))
        if checks:
            return {"supported": True, "passed": all(x[1] for x in checks), "category": category, "rule": "+".join(x[0] for x in checks), "evidence": "; ".join(x[2] for x in checks)}

    if category == "lexical":
        forbidden_mode = any(k in joined for k in ("forbidden", "avoid", "do not use", "without"))
        if forbidden_mode:
            terms = _extract_terms(args, desc, "forbidden", "avoid", "banned", "excluded")
            if terms:
                hits = [t for t in terms if t.lower() in low]
                return {"supported": True, "passed": not hits, "category": category, "rule": "forbidden_terms", "evidence": f"hits={hits}"}
        terms = _extract_terms(args, desc, "keyword", "required", "include", "words")
        if terms:
            hits = [t for t in terms if t.lower() in low]
            return {"supported": True, "passed": len(hits) == len(terms), "category": category, "rule": "required_terms", "evidence": f"hits={hits}, required={terms}"}

    if category == "format":
        if "json" in joined:
            try:
                json.loads(text.strip())
                ok = True
            except Exception:
                ok = False
            return {"supported": True, "passed": ok, "category": category, "rule": "json_parse", "evidence": "json.loads"}
        if "boxed" in joined:
            ok = "\\boxed" in text
            return {"supported": True, "passed": ok, "category": category, "rule": "contains_boxed", "evidence": "\\boxed present"}
        if "bullet" in joined or "list" in joined:
            ok = bool(re.search(r"(^|\n)\s*(?:[-*+]\s+|\d+[\.)]\s+)", text))
            return {"supported": True, "passed": ok, "category": category, "rule": "bullet_or_numbered_list", "evidence": "list marker"}
        if "table" in joined:
            ok = "|" in text and "\n" in text
            return {"supported": True, "passed": ok, "category": category, "rule": "markdown_table", "evidence": "pipe table"}
        if "final answer" in joined or "answer:" in joined:
            ok = bool(re.search(r"(?:final\s+answer|answer\s*:)", text, flags=re.I))
            return {"supported": True, "passed": ok, "category": category, "rule": "answer_marker", "evidence": "answer marker"}

    if category == "language":
        if "chinese" in joined or "中文" in joined:
            ok = bool(re.search(r"[\u4e00-\u9fff]", text))
            return {"supported": True, "passed": ok, "category": category, "rule": "contains_cjk", "evidence": "CJK chars"}
        if "english" in joined:
            letters = re.findall(r"[A-Za-z]", text)
            non_ascii = re.findall(r"[^\x00-\x7f]", text)
            ok = len(letters) >= max(10, len(non_ascii) * 3)
            return {"supported": True, "passed": ok, "category": category, "rule": "english_ascii_ratio", "evidence": f"letters={len(letters)}, non_ascii={len(non_ascii)}"}

    return {"supported": False, "passed": None, "category": category, "rule": "unsupported", "evidence": f"name={name!r}; desc={desc[:120]!r}"}


def extract_constraints(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    md = coerce_dict(r.get("metadata"))
    raw = md.get("constraint_list") or md.get("constraints_detail") or r.get("constraints")
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw

    names = coerce_list(md.get("constraint_name") or r.get("constraint_name") or md.get("instruction_type") or r.get("instruction_type"))
    descs = coerce_list(md.get("constraint_desc") or r.get("constraint_desc") or md.get("instruction") or r.get("instruction") or r.get("instructions"))
    args_raw = md.get("constraint_args") or r.get("constraint_args") or md.get("instruction_args") or r.get("instruction_args") or {}
    args_list = args_raw if isinstance(args_raw, list) else [args_raw]
    n = max(len(names), len(descs), len(args_list), 1)
    out = []
    for i in range(n):
        out.append({
            "name": str(names[i] if i < len(names) else (names[0] if names else "")),
            "desc": str(descs[i] if i < len(descs) else (descs[0] if descs else "")),
            "args": coerce_dict(args_list[i] if i < len(args_list) else (args_list[0] if args_list else {})),
        })
    return out


def evaluate_following(response: str, constraints: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per = []
    for c in constraints:
        per.append(evaluate_single_constraint(response, str(c.get("name", "")), str(c.get("desc", "")), coerce_dict(c.get("args"))))
    supported = [x for x in per if x.get("supported")]
    if not supported:
        return {
            "follow_soft": float("nan"),
            "follow_strict": float("nan"),
            "num_constraints": len(per),
            "num_supported_constraints": 0,
            "num_passed_constraints": 0,
            "per_constraint": per,
        }
    passed = [x for x in supported if x.get("passed") is True]
    return {
        "follow_soft": len(passed) / len(supported),
        "follow_strict": 1.0 if len(passed) == len(supported) else 0.0,
        "num_constraints": len(per),
        "num_supported_constraints": len(supported),
        "num_passed_constraints": len(passed),
        "per_constraint": per,
    }


def first_violation(response: str, constraints: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    segs = segment_trajectory(response)
    full = evaluate_following(response, constraints)
    if full.get("follow_strict") == 1.0:
        return {"first_violation_segment": None, "first_violation_phase": None, "num_segments": len(segs)}
    if not segs:
        return {"first_violation_segment": 0, "first_violation_phase": "empty", "num_segments": 0}
    cumulative = ""
    for i, seg in enumerate(segs):
        cumulative = (cumulative + "\n" + seg).strip()
        ev = evaluate_following(cumulative, constraints)
        # Only trust early violation for constraints where prefix failure is meaningful.
        failed = [pc for pc in ev.get("per_constraint", []) if pc.get("supported") and pc.get("passed") is False and pc.get("category") in {"length", "lexical", "affix"}]
        if failed:
            phase = "early" if i < len(segs) / 3 else "middle" if i < 2 * len(segs) / 3 else "late"
            return {"first_violation_segment": i, "first_violation_phase": phase, "num_segments": len(segs)}
    return {"first_violation_segment": len(segs) - 1, "first_violation_phase": "late", "num_segments": len(segs)}
