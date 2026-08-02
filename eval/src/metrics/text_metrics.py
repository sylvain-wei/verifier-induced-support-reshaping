"""
Post-hoc text metrics computed per response.

Tunable via `TEXT_METRIC_MARKERS` but defaults reflect the marker sets listed
in the RQ1 spec (English-primary, Chinese-supplementary).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------
# Marker regexes: precompiled for speed.
# Each group matches a "reasoning step signal" etc. Case-insensitive unless
# a Chinese literal.
# -----------------------------------------------------------------------

STEP_PATTERNS = [
    re.compile(r"\bstep\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?|finally)\b", re.IGNORECASE),
    re.compile(r"\b(therefore|thus|hence|so that|so,)\b", re.IGNORECASE),
    re.compile(r"\b(we need to|we have to|we should|let me)\b", re.IGNORECASE),
    re.compile(r"\b(let's|let us)\b", re.IGNORECASE),
    re.compile(r"(首先|然后|其次|接着|最后|因此|所以)"),
]

EQUATION_PATTERNS = [
    re.compile(r"\\frac\s*\{"),
    re.compile(r"\\sqrt\s*\{"),
    re.compile(r"\\boxed\s*\{"),
    re.compile(r"[A-Za-z0-9)\]]\s*\^\s*[{A-Za-z0-9(]"),
    re.compile(r"\d+\s*[+\-*/=×÷]\s*\d+"),
    re.compile(r"[=≤≥<>]"),
    re.compile(r"\$[^$]{1,200}\$"),
]

VERIFICATION_PATTERNS = [
    re.compile(r"\b(check|checking|verify|verified|substitute|plug back|sanity check|double[- ]?check|re[- ]?check)\b", re.IGNORECASE),
    re.compile(r"\b(let's check|let's verify|let us check)\b", re.IGNORECASE),
    re.compile(r"(验证|检查|代入|回代|复核)"),
]

OVER_REASONING_PATTERNS = [
    re.compile(r"\blet's think\b", re.IGNORECASE),
    re.compile(r"\bstep by step\b", re.IGNORECASE),
    re.compile(r"\bwe need to\b", re.IGNORECASE),
    re.compile(r"\b(reasoning|analysis|deliberately)\b", re.IGNORECASE),
    re.compile(r"\b(therefore|hence|thus)\b", re.IGNORECASE),
    re.compile(r"(首先|然后|因此|所以|推理|思考)"),
]


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE_SPLIT = re.compile(r"\s+")


def _safe_count(patterns: List[re.Pattern], text: str) -> int:
    total = 0
    for p in patterns:
        total += len(p.findall(text))
    return total


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = _SENT_SPLIT_RE.split(text, maxsplit=1)
    return parts[0].strip()[:500]


def _answer_position_ratio(text: str) -> Optional[float]:
    """Fraction of the response before the FINAL answer marker.

    Heuristic: prefers the last `\\boxed{...}` position, falling back to an
    "Answer:" line, then the very last numeric-looking token. Returns None
    when no candidate is found — *not* 0, so we don't conflate with "answered
    instantly".
    """
    if not text:
        return None
    best_end: Optional[int] = None
    matches = list(_BOXED_RE.finditer(text))
    if matches:
        best_end = matches[-1].end()
    else:
        m = list(_ANSWER_LINE_RE.finditer(text))
        if m:
            best_end = m[-1].end()
        else:
            nums = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
            if nums:
                best_end = nums[-1].end()
    if best_end is None or best_end <= 0:
        return None
    return min(1.0, max(0.0, best_end / max(1, len(text))))


def _prefix_tokens(text: str, n: int = 8) -> str:
    toks = _WHITESPACE_SPLIT.split(text.strip())
    return " ".join(toks[:n])


def compute_text_metrics(text: str, tokenizer=None) -> Dict[str, Any]:
    """Compute response-level text metrics. `tokenizer` is optional; when
    provided, `response_length_tokens` uses its vocab; otherwise whitespace
    tokens are used as a fallback."""
    if text is None:
        text = ""
    length_chars = len(text)
    if tokenizer is not None:
        try:
            length_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            length_tokens = len(text.split())
    else:
        length_tokens = len(text.split())

    step_count = _safe_count(STEP_PATTERNS, text)
    equation_count = _safe_count(EQUATION_PATTERNS, text)
    verification_count = _safe_count(VERIFICATION_PATTERNS, text)
    over_reasoning_count = _safe_count(OVER_REASONING_PATTERNS, text)

    # Densities: per-100-tokens to be robust to short outputs. Use length_tokens.
    scale = 100.0 / max(1, length_tokens)

    return {
        "response_length_chars": length_chars,
        "response_length_tokens": length_tokens,
        "first_sentence": _first_sentence(text),
        "prefix_8_tokens": _prefix_tokens(text, 8),
        "answer_position_ratio": _answer_position_ratio(text),
        "step_count": step_count,
        "step_density": step_count * scale,
        "equation_count": equation_count,
        "equation_density": equation_count * scale,
        "verification_marker_count": verification_count,
        "verification_marker_rate": verification_count * scale,
        "over_reasoning_marker_count": over_reasoning_count,
        "over_reasoning_marker_rate": over_reasoning_count * scale,
    }
