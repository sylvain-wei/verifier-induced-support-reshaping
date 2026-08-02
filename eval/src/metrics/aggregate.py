"""
Aggregation across (model, benchmark, mode).

Two-stage:
  1. compute_metrics.py produces per-response and per-prompt metrics.
  2. aggregate.py rolls them up into a single flat record per (model,
     benchmark, mode) for the main summary CSV, and plot_data.py computes
     z-scored pattern indexes.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import math


def _safe_mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and not math.isnan(x)]
    return float(mean(xs)) if xs else float("nan")


def _safe_std(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and not math.isnan(x)]
    return float(pstdev(xs)) if len(xs) > 1 else (0.0 if xs else float("nan"))


def aggregate_math_run(
    per_prompt_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate math metrics over prompts (each prompt has K samples already folded)."""
    out: Dict[str, Any] = {}
    n = len(per_prompt_records)
    out["num_prompts"] = n

    def col(key: str) -> List[float]:
        return [r.get(key) for r in per_prompt_records]

    for key in [
        "pass@1", "best@k", "maj@k",
        "best_minus_pass", "maj_minus_pass",
        "answer_entropy", "normalized_answer_entropy", "num_unique_answers",
        "response_length_mean", "response_length_std",
        "response_length_tokens_mean", "response_length_tokens_std",
        "distinct_1", "distinct_2", "distinct_3",
        "self_bleu_2", "self_bleu_4",
        "step_density_mean", "equation_density_mean",
        "verification_marker_rate_mean", "over_reasoning_marker_rate_mean",
        "answer_position_ratio_mean",
        "extraction_failure_rate",
    ]:
        out[key] = _safe_mean(col(key))
    return out


def aggregate_if_run(
    per_prompt_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(per_prompt_records)
    out["num_prompts"] = n

    def col(key: str) -> List[float]:
        return [r.get(key) for r in per_prompt_records]

    for key in [
        "strict_prompt_pass",
        "instruction_level_pass_rate",
        "format_pass_rate",
        "compliance_efficiency",
        "response_length_tokens",
        "over_reasoning_marker_rate",
        "num_constraints_mean",
    ]:
        out[key] = _safe_mean(col(key))

    # prefix_concentration + first_sentence_entropy are computed on the side
    return out


def compute_pattern_indexes(
    by_model: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Given {model: {metric: value}}, z-score each metric across models and
    return {model: {math_style_index, if_style_index}}."""
    metrics = set()
    for m in by_model.values():
        metrics.update(m.keys())
    z_by_metric: Dict[str, Dict[str, float]] = {}
    for mk in metrics:
        vals = [by_model[m].get(mk) for m in by_model]
        vals_f = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        if len(vals_f) < 2:
            continue
        mu = mean(vals_f)
        sd = pstdev(vals_f) or 1.0
        z_by_metric[mk] = {m: ((by_model[m].get(mk) - mu) / sd
                                if isinstance(by_model[m].get(mk), (int, float)) and not math.isnan(by_model[m].get(mk))
                                else float("nan"))
                            for m in by_model}

    out: Dict[str, Dict[str, float]] = {m: {} for m in by_model}

    # Math-Style Index: long, decomposed, formulaic, verifying, delayed, diverse
    math_keys_positive = [
        "response_length_tokens_mean",
        "step_density_mean",
        "equation_density_mean",
        "verification_marker_rate_mean",
        "answer_position_ratio_mean",
        "distinct_2_mean",
        "best_minus_pass_mean",   # "search benefit"
    ]
    if_keys_positive = [
        "strict_prompt_pass_mean",
        "format_pass_rate_mean",
        "compliance_efficiency_mean",
        "prefix_concentration_scalar",
    ]
    if_keys_negative = [
        "response_length_tokens_mean",
        "over_reasoning_marker_rate_mean",
    ]

    for model in by_model:
        math_zs = [z_by_metric.get(k, {}).get(model) for k in math_keys_positive]
        math_zs = [z for z in math_zs if isinstance(z, (int, float)) and not math.isnan(z)]
        out[model]["math_style_index"] = _safe_mean(math_zs)

        if_zs_pos = [z_by_metric.get(k, {}).get(model) for k in if_keys_positive]
        if_zs_neg = [-z for z in [z_by_metric.get(k, {}).get(model) for k in if_keys_negative]
                     if isinstance(z, (int, float)) and not math.isnan(z)]
        if_zs = [z for z in if_zs_pos if isinstance(z, (int, float)) and not math.isnan(z)] + if_zs_neg
        out[model]["if_style_index"] = _safe_mean(if_zs)
    return out
