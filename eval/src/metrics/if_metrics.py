"""
IF-specific metrics: delegates constraint checking to the unified verifier
and adds a few response-level derived metrics (compliance efficiency,
over-reasoning marker rate).

The unified verifier routes to either IFEval (vendored verl.bak) or IFBench
(vendored allenai/IFBench) based on the `benchmark` argument.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .if_verifier_unified import verify as verify_unified
from .text_metrics import compute_text_metrics


def score_if_response(
    response: str,
    verifier_metadata: Dict[str, Any],
    tokenizer=None,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    instruction_ids = list(verifier_metadata.get("instruction_ids") or [])
    kwargs_list = list(verifier_metadata.get("kwargs") or [])

    result = verify_unified(response or "", instruction_ids, kwargs_list, benchmark=benchmark)
    tm = compute_text_metrics(response or "", tokenizer=tokenizer)

    gen_tokens = max(1, int(tm.get("response_length_tokens") or 1))
    compliance_efficiency = (result.num_satisfied / gen_tokens) * 100.0

    return {
        "strict_prompt_pass": int(result.strict_prompt_pass),
        "num_constraints": result.num_constraints,
        "num_supported": result.num_supported,
        "num_satisfied": result.num_satisfied,
        "instruction_level_pass_rate": result.instruction_level_pass_rate,
        "format_pass_rate": result.format_pass_rate,
        "num_format_constraints": result.num_format_constraints,
        "num_format_passed": result.num_format_passed,
        "compliance_efficiency": compliance_efficiency,
        "per_constraint": [
            {
                "instruction_id": pc.instruction_id,
                "supported": pc.supported,
                "passed": pc.passed,
                "error": pc.error,
            }
            for pc in result.per_constraint
        ],
        "response_length_tokens": tm["response_length_tokens"],
        "over_reasoning_marker_rate": tm["over_reasoning_marker_rate"],
        "prefix_8_tokens": tm["prefix_8_tokens"],
        "first_sentence": tm["first_sentence"],
    }


def prefix_concentration(prefixes: List[str], top_k: int = 1) -> float:
    if not prefixes:
        return float("nan")
    from collections import Counter
    cnt = Counter(prefixes)
    top = cnt.most_common(top_k)
    return sum(c for _, c in top) / len(prefixes)


def first_sentence_entropy(sentences: List[str]) -> float:
    if not sentences:
        return float("nan")
    from collections import Counter
    from math import log
    cnt = Counter(sentences)
    total = sum(cnt.values())
    ent = 0.0
    for _, c in cnt.items():
        p = c / total
        if p > 0:
            ent -= p * log(p, 2)
    return ent
