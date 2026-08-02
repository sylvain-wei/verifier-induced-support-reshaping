"""
Math answer extraction, normalization, and correctness.

Extraction priority (matches RQ1 spec):
  1. last `\\boxed{...}`          (MATH-500, GSM8K default_math)
  2. "Answer: X" on its own line (AIME)
  3. "The answer is X"
  4. last latex/numeric token
  5. extraction_failure=True

Comparison:
  1. Canonical string equality (`normalize_answer`)
  2. sympy `simplify(a - b) == 0` with 3s timeout
  3. Integer match (for AIME 0-999)
"""
from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from typing import Any, Counter as CounterT, Dict, List, Optional, Tuple

try:
    import sympy as _sp
    from sympy.parsing.latex import parse_latex as _parse_latex
    _HAS_SYMPY = True
except Exception:
    _sp = None
    _parse_latex = None
    _HAS_SYMPY = False

# ---------------- extraction ----------------

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_THE_ANSWER_IS_RE = re.compile(r"\b(?:the\s+answer\s+is|final\s+answer\s*[:\-]?)\s*([\-+]?\s*[\d\\{][^\n\r]{0,120})", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
# last_latex fallback: only match TRUE latex math constructs. The old
# "$...$" matcher kept eating GSM8K currency pairs like "$2 each = $18".
# We now only recognise \frac{..}{..}, \sqrt{..}, and \boxed{..} as latex
# fallback.
_LATEX_FRAGMENT_RE = re.compile(
    r"\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}|\\sqrt\s*\{[^{}]*\}|\\boxed\s*\{[^{}]*\}"
)


def extract_final_answer(text: str) -> Tuple[str, str]:
    """Returns (extracted, method). method in:
       boxed / answer_line / the_answer_is / last_number / last_latex / none."""
    if not text:
        return "", "none"

    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip(), "boxed"

    ans = list(_ANSWER_LINE_RE.finditer(text))
    if ans:
        val = ans[-1].group(1).strip().rstrip(".")
        return val, "answer_line"

    the = list(_THE_ANSWER_IS_RE.finditer(text))
    if the:
        val = the[-1].group(1).strip().rstrip(".")
        return val, "the_answer_is"

    # Look within the last 200 chars for a numeric fallback
    tail = text[-400:]
    frags = list(_LATEX_FRAGMENT_RE.finditer(tail))
    if frags:
        return frags[-1].group(0).strip("$").strip(), "last_latex"
    nums = list(_NUM_RE.finditer(tail))
    if nums:
        return nums[-1].group(0), "last_number"

    return "", "none"


# ---------------- normalization ----------------

_SPACE_RE = re.compile(r"\s+")
_LEFT_RE = re.compile(r"\\left\s*")
_RIGHT_RE = re.compile(r"\\right\s*")
_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\s*\{([^{}]*)\}")
_DOLLAR_RE = re.compile(r"\\\$")
_PERCENT_RE = re.compile(r"\\%")


def _strip_latex_decor(s: str) -> str:
    s = _LEFT_RE.sub("", s)
    s = _RIGHT_RE.sub("", s)
    s = _TEXT_RE.sub(lambda m: m.group(1), s)
    s = _MBOX_RE.sub(lambda m: m.group(1), s)
    s = _DOLLAR_RE.sub("$", s)
    s = _PERCENT_RE.sub("%", s)
    s = s.replace("\\!", "").replace("\\;", "").replace("\\,", "")
    s = s.replace("\\$", "$")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    return s


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = _strip_latex_decor(s)
    # outer braces
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    # dollar wrap
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # remove trailing punctuation / units
    s = s.rstrip(".;,")
    s = _SPACE_RE.sub("", s)
    # common integer normalisation: remove leading +, "-0" -> "0"
    if s.startswith("+"):
        s = s[1:]
    s = s.lower()
    return s


# ---------------- numeric-aware normalization ----------------
# GSM8K, arithmetic problems, etc. have numeric answers where the model will
# frequently write "$18", "70,000", "18%", "18 dollars", "\boxed{\$70,000}".
# normalize_numeric strips these decorations without losing the number itself.

_NUMERIC_TRAILING_UNITS_RE = re.compile(
    r"\s*(dollars?|usd|cents?|pounds?|kg|kgs|km|m|ft|feet|inches|meters|degrees?|hours?|minutes?|seconds?|days?|years?|months?|%)\s*$",
    re.IGNORECASE,
)


def normalize_numeric(s: str) -> str:
    """Normalize a numeric answer: strip $, commas, units, % — return a
    canonical form that `float(...)` can parse.
    Examples:
      '$70,000'           -> '70000'
      '\\$18'              -> '18'
      '18%'               -> '18'
      '18 dollars'        -> '18'
      '\\boxed{18}'        -> '18'
      '\\boxed{$70,000}'   -> '70000'
    """
    if s is None:
        return ""
    # If caller passes raw response text containing \boxed{...}, unwrap it.
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1)
    s = normalize_answer(s)
    # strip currency symbols (after latex-decor) and trailing units
    s = s.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    s = _NUMERIC_TRAILING_UNITS_RE.sub("", s)
    # strip thousands commas between digits: 70,000 -> 70000
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # strip stray whitespace left over
    s = s.strip()
    # final rstrip of punctuation
    s = s.rstrip(".;,")
    return s


def _parse_float(s: str) -> Optional[float]:
    """Try to parse s as a float after numeric normalization. Returns None on failure."""
    try:
        return float(normalize_numeric(s))
    except Exception:
        return None


def _as_int_in_range(s: str, lo: int = 0, hi: int = 999) -> Optional[int]:
    try:
        v = int(str(s).strip())
    except Exception:
        return None
    if lo <= v <= hi:
        return v
    return None


# -------------- sympy equivalence with hard timeout --------------

class _TimeoutErr(Exception):
    pass


@contextmanager
def _time_limit(sec: float):
    # Use SIGALRM if available (POSIX). Fallback: no timeout.
    handler_installed = False
    try:
        def handler(signum, frame):
            raise _TimeoutErr()
        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, sec)
        handler_installed = True
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except (AttributeError, ValueError):
        # signal.SIGALRM not available (e.g., threads) — no-op.
        yield


def _sympy_equiv(a: str, b: str, timeout: float = 3.0) -> bool:
    if not _HAS_SYMPY or not a or not b:
        return False
    try:
        with _time_limit(timeout):
            try:
                pa = _parse_latex(a) if "\\" in a else _sp.sympify(a)
                pb = _parse_latex(b) if "\\" in b else _sp.sympify(b)
            except Exception:
                try:
                    pa = _sp.sympify(a)
                    pb = _sp.sympify(b)
                except Exception:
                    return False
            try:
                diff = _sp.simplify(pa - pb)
            except Exception:
                return False
            return diff == 0
    except _TimeoutErr:
        return False
    except Exception:
        return False


def is_equiv(a: str, b: str, answer_type: str = "") -> bool:
    if a is None or b is None:
        return False
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if na == nb and na != "":
        return True

    # Numeric tasks: normalize $, commas, units before comparing.
    if answer_type in ("numeric", "integer_0_999"):
        fa = _parse_float(a)
        fb = _parse_float(b)
        if fa is not None and fb is not None:
            if fa == fb:
                return True
            # tolerate tiny floating-point discrepancies for decimal answers
            if abs(fa - fb) < 1e-6:
                return True

    if answer_type == "integer_0_999":
        ia = _as_int_in_range(a)
        ib = _as_int_in_range(b)
        if ia is not None and ib is not None:
            return ia == ib

    # fallback numeric match (less strict about types)
    try:
        if float(na) == float(nb):
            return True
    except Exception:
        pass
    return _sympy_equiv(a, b)


# -------------- per-response math metrics --------------

def score_response(text: str, gold: str, answer_type: str = "") -> Dict[str, Any]:
    extracted, method = extract_final_answer(text)
    correct = False
    if extracted != "" and gold is not None:
        correct = is_equiv(extracted, str(gold), answer_type=answer_type)
    return {
        "extracted_answer": extracted,
        "extraction_method": method,
        "extraction_failure": method == "none",
        "correct": int(bool(correct)),
        "normalized_extracted": normalize_answer(extracted),
        "normalized_gold": normalize_answer(str(gold) if gold is not None else ""),
    }


# -------------- per-prompt K-sample metrics --------------

def aggregate_over_samples(sample_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    from math import log

    if not sample_scores:
        return {
            "k": 0, "pass@1": float("nan"), "best@k": float("nan"), "maj@k": float("nan"),
            "best_minus_pass": float("nan"), "maj_minus_pass": float("nan"),
            "answer_entropy": float("nan"), "normalized_answer_entropy": float("nan"),
            "num_unique_answers": 0,
        }
    k = len(sample_scores)
    correct_list = [int(s.get("correct", 0)) for s in sample_scores]
    pass_at_1 = sum(correct_list) / k
    best_at_k = 1.0 if any(correct_list) else 0.0

    # majority-vote over normalized answers
    answers = [s.get("normalized_extracted") or "" for s in sample_scores]
    cnt = Counter([a for a in answers if a])
    if cnt:
        top, _ = cnt.most_common(1)[0]
        gold_norm = sample_scores[0].get("normalized_gold", "")
        maj_at_k = 1.0 if (top == gold_norm and gold_norm != "") else 0.0
    else:
        maj_at_k = 0.0

    # entropy over non-empty answers
    total = sum(cnt.values())
    ent = 0.0
    for _, c in cnt.items():
        p = c / total if total else 0.0
        if p > 0:
            ent -= p * log(p, 2)
    norm_ent = ent / log(max(2, len(cnt)), 2) if len(cnt) > 1 else 0.0

    return {
        "k": k,
        "pass@1": pass_at_1,
        "best@k": best_at_k,
        "maj@k": maj_at_k,
        "best_minus_pass": best_at_k - pass_at_1,
        "maj_minus_pass": maj_at_k - pass_at_1,
        "answer_entropy": ent,
        "normalized_answer_entropy": norm_ent,
        "num_unique_answers": len(cnt),
    }
