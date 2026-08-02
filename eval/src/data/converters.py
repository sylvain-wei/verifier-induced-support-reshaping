"""
Converters: turn heterogeneous raw datasets into the unified eval schema.

All converters take (raw_records, benchmark_cfg, prompts_cfg) and return a
list of unified records. Prompt templating is applied here so that the
inference engine only sees text prompts (no special tokens).
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional


# ----------------------------- helpers -----------------------------

def _apply_prompt(template: str, **kwargs) -> str:
    """Safe template application using str.format with a small safeguard."""
    # Escape raw braces that are NOT our placeholders. Our templates use
    # single-brace placeholders like {problem}/{question}/{prompt}. We replace
    # placeholders by simple string substitution to avoid LaTeX brace issues.
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _load_prompt_template(prompts_cfg: Dict[str, Any], template_id: str) -> str:
    templates = prompts_cfg.get("templates", {})
    if template_id not in templates:
        raise KeyError(f"Prompt template {template_id!r} not found in configs/prompts.yaml")
    return templates[template_id]


# ----------------------------- math500 -----------------------------

def math500_from_parquet(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each record is from data/math500/test.parquet (DigitalLearningGmbH/MATH-lighteval schema)."""
    tpl = _load_prompt_template(prompts_cfg, bench_cfg["prompt_template_id"])
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        # Raw problem is inside prompt[0].content in the shipped parquet; strip the
        # canned "Let's think step by step..." tail so the problem is clean.
        prompt_list = r.get("prompt") or []
        if prompt_list and isinstance(prompt_list, list) and isinstance(prompt_list[0], dict):
            content = prompt_list[0].get("content") or ""
        else:
            content = r.get("problem") or ""
        # Remove the canonical suffix if present
        m = re.search(r"^(?P<body>.*?)(?:\s*Let's think step by step.*)?$", content.strip(), flags=re.S)
        raw_problem = (m.group("body") if m else content).strip()

        gold = ""
        rm = r.get("reward_model")
        if isinstance(rm, dict):
            gold = rm.get("ground_truth", "")
        extra = r.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}

        prompt_text = _apply_prompt(tpl, problem=raw_problem)
        out.append({
            "id": f"math500_{i:04d}",
            "benchmark": "math500",
            "task_type": "math",
            "prompt": prompt_text,
            "raw_problem": raw_problem,
            "gold": gold,
            "metadata": {
                "source": r.get("data_source", "DigitalLearningGmbH/MATH-lighteval"),
                "level": extra.get("level"),
                "subject": extra.get("type"),
                "answer_type": "latex_boxed",
                "index": extra.get("index", i),
                "prompt_template_id": bench_cfg["prompt_template_id"],
            },
        })
    return out


# ----------------------------- aime24 / aime25 -----------------------------

def aime_from_parquet(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AIME records already have `problem` + `answer` + `prompt` (MATH_v2 style)."""
    tpl = _load_prompt_template(prompts_cfg, bench_cfg["prompt_template_id"])
    benchmark = None
    for key, val in prompts_cfg.get("_benchmark_hint", {}).items():  # no-op
        pass
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        raw_problem = r.get("problem") or ""
        extra = r.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        if "raw_problem" in extra and extra["raw_problem"]:
            raw_problem = extra["raw_problem"]

        gold = r.get("answer")
        if gold is None:
            rm = r.get("reward_model") or {}
            if isinstance(rm, dict):
                gold = rm.get("ground_truth")
        gold = str(gold) if gold is not None else ""

        origin = (extra.get("origin") or r.get("data_source") or "aime24").lower()
        if "25" in origin:
            bench_name = "aime25"
        else:
            bench_name = "aime24"

        prompt_text = _apply_prompt(tpl, problem=raw_problem)
        out.append({
            "id": f"{bench_name}_{i:04d}",
            "benchmark": bench_name,
            "task_type": "math",
            "prompt": prompt_text,
            "raw_problem": raw_problem,
            "gold": gold,
            "metadata": {
                "source": r.get("data_source", bench_name),
                "answer_type": "integer_0_999",
                "index": extra.get("index", i),
                "prompt_template_id": bench_cfg["prompt_template_id"],
            },
        })
    return out


# ----------------------------- gsm8k -----------------------------

_GSM8K_GOLD_RE = re.compile(r"####\s*([^\n]+)")


def _parse_gsm8k_gold(full_answer: str) -> str:
    m = _GSM8K_GOLD_RE.search(full_answer or "")
    if m:
        return m.group(1).strip()
    # fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", full_answer or "")
    return nums[-1] if nums else (full_answer or "").strip()


def gsm8k_from_hf(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GSM8K test records: {question, answer_full} or upstream {question, answer}."""
    tpl = _load_prompt_template(prompts_cfg, bench_cfg["prompt_template_id"])
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        q = r.get("question") or ""
        full_answer = r.get("answer_full") or r.get("answer") or ""
        gold = _parse_gsm8k_gold(full_answer)
        # Pass both `problem` and `question` so that either the GSM8K-specific
        # template (`gsm8k_default` uses {question}) or the unified math
        # template (`default_math` uses {problem}) works without edits.
        prompt_text = _apply_prompt(tpl, problem=q, question=q)
        out.append({
            "id": f"gsm8k_{i:04d}",
            "benchmark": "gsm8k",
            "task_type": "math",
            "prompt": prompt_text,
            "raw_problem": q,
            "gold": gold,
            "metadata": {
                "source": "openai/gsm8k",
                "answer_type": "numeric",
                "answer_full": full_answer,
                "index": i,
                "prompt_template_id": bench_cfg["prompt_template_id"],
            },
        })
    return out


# ----------------------------- ifeval -----------------------------

def _parse_ifeval_ground_truth(rm_gt: Any) -> Dict[str, Any]:
    """Local IFEval parquet stores reward_model.ground_truth as a Python-literal string
    holding a list[dict] like: [{'instruction_id': [...], 'kwargs': [...]}]."""
    if isinstance(rm_gt, dict):
        return rm_gt
    if isinstance(rm_gt, str):
        try:
            obj = ast.literal_eval(rm_gt)
        except Exception:
            try:
                obj = json.loads(rm_gt)
            except Exception:
                return {}
        if isinstance(obj, list) and obj:
            return obj[0] if isinstance(obj[0], dict) else {}
        if isinstance(obj, dict):
            return obj
    return {}


def ifeval_from_parquet(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """IFEval local parquet schema: key, prompt(list[dict]), instruction_id_list, kwargs, reward_model.ground_truth."""
    tpl = _load_prompt_template(prompts_cfg, bench_cfg["prompt_template_id"])
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        prompt_list = r.get("prompt") or []
        if prompt_list and isinstance(prompt_list, list) and isinstance(prompt_list[0], dict):
            prompt_text_raw = prompt_list[0].get("content") or ""
        else:
            prompt_text_raw = r.get("prompt_text") or ""

        instruction_ids: List[str] = list(r.get("instruction_id_list") or [])
        kwargs_list: List[Dict[str, Any]] = list(r.get("kwargs") or [])

        # Also parse reward_model.ground_truth to cross-check / fallback
        rm = r.get("reward_model")
        gt: Dict[str, Any] = {}
        if isinstance(rm, dict):
            gt = _parse_ifeval_ground_truth(rm.get("ground_truth"))
        if not instruction_ids and gt.get("instruction_id"):
            instruction_ids = list(gt.get("instruction_id") or [])
        if not kwargs_list and gt.get("kwargs"):
            kwargs_list = list(gt.get("kwargs") or [])

        # Normalise kwargs: drop None-valued keys
        cleaned_kwargs: List[Dict[str, Any]] = []
        for kw in kwargs_list:
            if not isinstance(kw, dict):
                cleaned_kwargs.append({})
                continue
            cleaned_kwargs.append({k: v for k, v in kw.items() if v is not None})

        prompt_text = _apply_prompt(tpl, prompt=prompt_text_raw)
        out.append({
            "id": f"ifeval_{i:04d}",
            "benchmark": "ifeval",
            "task_type": "if",
            "prompt": prompt_text,
            "raw_problem": prompt_text_raw,
            "gold": None,
            "metadata": {
                "source": r.get("data_source", "google/IFEval"),
                "key": r.get("key", i),
                "constraints": instruction_ids,
                "constraint_types": [cid.split(":", 1)[0] for cid in instruction_ids],
                "verifier_metadata": {
                    "instruction_ids": instruction_ids,
                    "kwargs": cleaned_kwargs,
                },
                "prompt_template_id": bench_cfg["prompt_template_id"],
            },
        })
    return out


# ----------------------------- ifbench (format-only) -----------------------------

def ifbench_format_only(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """IFBench records: {key, prompt, instruction_id_list, kwargs}. Filter to
    format-only instruction IDs supported by the bundled verifier. Subset to
    bench_cfg['format_only_max'] with a fixed seed."""
    # Lazy import to avoid circulars
    from ..metrics.if_verifier.ifbench_supported_ids import (
        FORMAT_ONLY_SUPPORTED_IDS,
        is_format_only_example,
    )

    tpl = _load_prompt_template(prompts_cfg, bench_cfg["prompt_template_id"])
    tmp: List[Dict[str, Any]] = []
    dropped = 0
    for r in records:
        instruction_ids = list(r.get("instruction_id_list") or [])
        kwargs_list = list(r.get("kwargs") or [])
        if not instruction_ids:
            dropped += 1
            continue
        if not is_format_only_example(instruction_ids):
            dropped += 1
            continue

        # Prompt field in IFBench HF sometimes is `prompt` (str) and sometimes
        # a list[dict] similar to IFEval. Handle both.
        prompt_raw = r.get("prompt")
        if isinstance(prompt_raw, list) and prompt_raw and isinstance(prompt_raw[0], dict):
            prompt_text_raw = prompt_raw[0].get("content") or ""
        elif isinstance(prompt_raw, str):
            prompt_text_raw = prompt_raw
        else:
            prompt_text_raw = r.get("text") or r.get("instruction") or ""

        cleaned_kwargs: List[Dict[str, Any]] = []
        for kw in kwargs_list:
            if not isinstance(kw, dict):
                cleaned_kwargs.append({})
            else:
                cleaned_kwargs.append({k: v for k, v in kw.items() if v is not None})

        tmp.append({
            "benchmark": "ifbench",
            "task_type": "if",
            "raw_problem": prompt_text_raw,
            "prompt": _apply_prompt(tpl, prompt=prompt_text_raw),
            "gold": None,
            "metadata": {
                "source": r.get("data_source", "allenai/IFBench"),
                "key": r.get("key", r.get("id")),
                "constraints": instruction_ids,
                "constraint_types": [cid.split(":", 1)[0] for cid in instruction_ids],
                "verifier_metadata": {
                    "instruction_ids": instruction_ids,
                    "kwargs": cleaned_kwargs,
                },
                "prompt_template_id": bench_cfg["prompt_template_id"],
            },
        })

    # Deterministic subset
    import random as _r
    rng = _r.Random(int(bench_cfg.get("format_only_seed", 42)))
    rng.shuffle(tmp)
    max_n = int(bench_cfg.get("format_only_max", 100))
    chosen = sorted(tmp[:max_n], key=lambda x: str(x["metadata"].get("key")))
    # Assign sequential ids
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(chosen):
        rec["id"] = f"ifbench_{i:04d}"
        out.append(rec)
    return out


# ----------------------------- MathIF / ReasonIF -----------------------------

def _first_nonempty(r: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return default


def mathif_from_json(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MathIF JSONL schema from TingchenFu/MathIF.

    Expected fields: source, id, question, answer, constraint_desc,
    constraint_name, constraint_args. The function is permissive so locally
    downloaded HF variants with slightly different field names still convert.
    """
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        question = str(_first_nonempty(r, "question", "problem", "prompt", "input"))
        answer = str(_first_nonempty(r, "answer", "gold", "final_answer"))
        cname = _first_nonempty(r, "constraint_name", "category", "instruction_type")
        cdesc = str(_first_nonempty(r, "constraint_desc", "instruction", "instructions"))
        cargs = r.get("constraint_args") or r.get("args") or r.get("kwargs") or {}
        if not isinstance(cargs, dict):
            cargs = {}
        source = str(_first_nonempty(r, "source", "dataset", default="MathIF"))
        answer_type = "numeric" if source.lower() == "gsm8k" else "latex_boxed"
        prompt_text = f"{question.strip()}\n\nInstruction: {cdesc.strip()}" if cdesc else question.strip()
        out.append({
            "id": f"mathif_{_first_nonempty(r, 'id', 'example_id', default=f'{i:04d}')}",
            "benchmark": "mathif",
            "task_type": "mathif_reasoning",
            "prompt": prompt_text,
            "raw_problem": question,
            "gold": answer,
            "metadata": {
                "source": source,
                "answer_type": answer_type,
                "constraint_name": cname,
                "constraint_desc": cdesc,
                "constraint_args": cargs,
                "constraint_types": [str(cname).split(":", 1)[0]] if cname else [],
                "prompt_template_id": bench_cfg.get("prompt_template_id", "mathif_passthrough"),
            },
        })
    return out


def reasonif_from_json(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ReasonIF JSON schema from ykwon0407/reasonIF / HF mirror.

    The official dataset contains 300 reasoning prompts spanning multilingual,
    formatting, and length-control instructions. This converter keeps original
    prompt text if present and stores the instruction metadata for downstream
    trajectory analysis.
    """
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        prompt = _first_nonempty(r, "prompt", "input")
        question = str(_first_nonempty(r, "question", "problem", "query"))
        inst = _first_nonempty(r, "instruction", "instructions", "constraint", "constraint_desc")
        if isinstance(inst, list):
            inst = " ".join(str(x) for x in inst)
        if not prompt:
            prompt = f"{question.strip()}\n\nInstruction: {str(inst).strip()}" if inst else question.strip()
        args = r.get("constraint_args") or r.get("instruction_args") or r.get("args") or r.get("kwargs") or {}
        if not isinstance(args, dict):
            args = {}
        category = _first_nonempty(r, "category", "instruction_category", "instruction_type", "type")
        out.append({
            "id": f"reasonif_{_first_nonempty(r, 'id', 'example_id', default=f'{i:04d}')}",
            "benchmark": "reasonif",
            "task_type": "reasonif_reasoning",
            "prompt": str(prompt).strip(),
            "raw_problem": question,
            "gold": str(_first_nonempty(r, "answer", "gold", "final_answer", "target")),
            "metadata": {
                "source": str(_first_nonempty(r, "domain", "task", "subject", "source", default="ReasonIF")),
                "answer_type": str(_first_nonempty(r, "answer_type", default="numeric")),
                "instruction_type": str(category),
                "dynamic_instruction_type": str(category),
                "instruction": str(inst),
                "constraint_name": str(category),
                "constraint_desc": str(inst),
                "constraint_args": args,
                "prompt_template_id": bench_cfg.get("prompt_template_id", "reasonif_passthrough"),
            },
        })
    return out
