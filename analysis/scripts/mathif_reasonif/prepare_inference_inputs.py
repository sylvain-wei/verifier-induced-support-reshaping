#!/usr/bin/env python
"""Prepare MathIF/ReasonIF raw data into the local eval unified JSONL schema.

Examples:
  python prepare_inference_inputs.py --benchmark mathif \
    --input data/mathif/mathif.jsonl \
    --output eval/cache/processed_data/mathif.jsonl

  python prepare_inference_inputs.py --benchmark reasonif \
    --input data/reasonif/reasonIF_dataset.json \
    --output eval/cache/processed_data/reasonif.jsonl
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import coerce_dict, coerce_list, read_records, write_jsonl  # noqa: E402

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))


def _first(r: Dict[str, Any], *keys: str, default: str = "") -> Any:
    for k in keys:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return default


def _mathif_prompt(question: str, desc: str) -> str:
    if desc:
        return f"{question.strip()}\n\nInstruction: {desc.strip()}"
    return question.strip()


def convert_mathif(records: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(records[:limit] if limit else records):
        question = str(_first(r, "question", "problem", "prompt", "input"))
        answer = str(_first(r, "answer", "gold", "final_answer"))
        cname = _first(r, "constraint_name", "category", "instruction_type", default="")
        cdesc = str(_first(r, "constraint_desc", "instruction", "instructions", default=""))
        cargs = coerce_dict(_first(r, "constraint_args", "args", "kwargs", default={}))
        source = str(_first(r, "source", "dataset", default="MathIF"))
        answer_type = "numeric" if source.lower() == "gsm8k" else "latex_boxed"
        ex_id = str(_first(r, "id", "example_id", default=f"mathif_{i:04d}"))
        out.append({
            "id": f"mathif_{ex_id}",
            "benchmark": "mathif",
            "task_type": "mathif_reasoning",
            "prompt": _mathif_prompt(question, cdesc),
            "raw_problem": question,
            "gold": answer,
            "metadata": {
                "source": source,
                "answer_type": answer_type,
                "constraint_name": cname,
                "constraint_desc": cdesc,
                "constraint_args": cargs,
                "constraint_types": [str(cname).split(":", 1)[0]] if cname else [],
                "prompt_template_id": "mathif_passthrough",
            },
        })
    return out


def _reasonif_prompt(r: Dict[str, Any]) -> str:
    prompt = _first(r, "prompt", "input", default="")
    if prompt:
        return str(prompt).strip()
    q = str(_first(r, "question", "problem", "query", default="")).strip()
    inst = _first(r, "instruction", "instructions", "constraint", "constraint_desc", default="")
    if isinstance(inst, list):
        inst = " ".join(str(x) for x in inst)
    inst = str(inst).strip()
    if q and inst:
        return f"{q}\n\nInstruction: {inst}"
    return q or inst


def convert_reasonif(records: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(records[:limit] if limit else records):
        answer = str(_first(r, "answer", "gold", "final_answer", "target", default=""))
        ex_id = str(_first(r, "id", "example_id", default=f"reasonif_{i:04d}"))
        category = _first(r, "category", "instruction_category", "instruction_type", "type", default="")
        instruction = _first(r, "instruction", "instructions", "constraint", "constraint_desc", default="")
        if isinstance(instruction, list):
            instruction = " ".join(str(x) for x in instruction)
        args = coerce_dict(_first(r, "constraint_args", "instruction_args", "args", "kwargs", default={}))
        domain = _first(r, "domain", "task", "subject", "source", default="")
        out.append({
            "id": f"reasonif_{ex_id}",
            "benchmark": "reasonif",
            "task_type": "reasonif_reasoning",
            "prompt": _reasonif_prompt(r),
            "raw_problem": str(_first(r, "question", "problem", "query", default="")),
            "gold": answer,
            "metadata": {
                "source": str(domain or "ReasonIF"),
                "answer_type": str(_first(r, "answer_type", default="numeric")),
                "instruction_type": str(category),
                "instruction": str(instruction),
                "constraint_name": str(category),
                "constraint_desc": str(instruction),
                "constraint_args": args,
                "dynamic_instruction_type": str(category),
                "prompt_template_id": "reasonif_passthrough",
                "raw_metadata": {k: v for k, v in r.items() if k not in {"prompt", "question", "problem"}},
            },
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["mathif", "reasonif"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = read_records(args.input)
    if args.benchmark == "mathif":
        out = convert_mathif(records, args.limit)
    else:
        out = convert_reasonif(records, args.limit)
    n = write_jsonl(args.output, out)
    print(json.dumps({"benchmark": args.benchmark, "input": args.input, "output": args.output, "records": n}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
