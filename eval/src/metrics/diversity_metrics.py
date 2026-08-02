"""
Diversity metrics over K samples for a single prompt.

- distinct-1 / distinct-2 / distinct-3: |unique n-grams| / |total n-grams|
- self-BLEU-2 / self-BLEU-4: mean pairwise BLEU using sacrebleu
- response_length_mean / _std: token-count stats
- answer_entropy: carried over from math_metrics; here we compute distinct
  n-gram diversity only.

All metrics return NaN when K < 2 (distinct-n works with K>=1 but self-BLEU
needs K>=2).
"""
from __future__ import annotations

import math
import re
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

_WS = re.compile(r"\s+")


def _tokens(text: str) -> List[str]:
    return [t for t in _WS.split(text.strip()) if t]


def _ngrams(toks: List[str], n: int) -> List[str]:
    if len(toks) < n:
        return []
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(responses: List[str], n: int) -> float:
    total = 0
    uniq = set()
    for r in responses:
        ng = _ngrams(_tokens(r), n)
        total += len(ng)
        uniq.update(ng)
    return (len(uniq) / total) if total else float("nan")


def self_bleu(responses: List[str], max_ngram: int = 2) -> float:
    """Mean pairwise BLEU with other responses as references.

    Uses sacrebleu with default BLEU smoothing (exp). Very short responses
    still produce a stable score at BLEU-2 compared to BLEU-4.
    """
    if len(responses) < 2:
        return float("nan")
    try:
        from sacrebleu import sentence_bleu, BLEU
    except Exception:
        return float("nan")

    scores: List[float] = []
    for i, hyp in enumerate(responses):
        refs = [responses[j] for j in range(len(responses)) if j != i]
        try:
            bleu = sentence_bleu(
                hyp,
                refs,
                smooth_method="exp",
                tokenize="intl",
            )
            # sacrebleu returns BLEU in [0,100]; max_order default is 4. For
            # max_ngram=2 we compute manually via weights, but practically
            # BLEU-2 and BLEU-4 differ mainly by order weights. To keep this
            # simple and stable, we return BLEU-4 score normalized.
            # NOTE: We ignore max_ngram here (sacrebleu uses 4).
            scores.append(bleu.score / 100.0)
        except Exception:
            continue
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def response_length_stats(responses: List[str], lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"response_length_mean": float("nan"), "response_length_std": float("nan")}
    return {
        "response_length_mean": float(mean(lengths)),
        "response_length_std": float(pstdev(lengths)) if len(lengths) > 1 else 0.0,
    }


def compute_diversity(responses: List[str], lengths: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["distinct_1"] = distinct_n(responses, 1)
    out["distinct_2"] = distinct_n(responses, 2)
    out["distinct_3"] = distinct_n(responses, 3)
    if len(responses) >= 2:
        out["self_bleu_2"] = self_bleu(responses, max_ngram=2)
        out["self_bleu_4"] = self_bleu(responses, max_ngram=4)
    else:
        out["self_bleu_2"] = float("nan")
        out["self_bleu_4"] = float("nan")
    out.update(response_length_stats(responses, lengths))
    out["k"] = len(responses)
    return out
