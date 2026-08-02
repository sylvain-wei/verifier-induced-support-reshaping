#!/usr/bin/env python
"""ReasonIF-specific verifier.

Critical design: ReasonIF instructions apply to the *reasoning trace*, NOT to
the final answer. The model is told to wrap its final answer in
`<answer>...</answer>` and to follow the constraint within everything before
the `<answer>` block.

This module implements one checker per of the 6 official ReasonIF constraint
classes:

  - punctuation:no_comma
  - length_constraint_checkers:number_words
  - language:reasoning_language
  - startend:end_checker
  - detectable_format:json_format
  - change_case:english_capital

Each checker returns:
  {
    "supported": True/False,
    "reasoning_passed": True/False/None,   # constraint applied to reasoning trace
    "response_passed": True/False/None,    # same constraint applied to full response
    "wrapper_present": True/False,         # whether <answer>...</answer> wrapper exists
    "answer_correct": True/False/None,     # when gold provided
    "evidence": str,
  }

The split between reasoning-trace IF and response-level IF is what lets us
answer the headline question: did the model learn to follow during reasoning,
or did it skip reasoning altogether so the constraint can't be violated?
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

ANSWER_OPEN_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"<\s*/\s*answer\s*>", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def split_reasoning_and_answer(response: str) -> Tuple[str, Optional[str], bool]:
    """Return (reasoning_text, answer_text_or_None, wrapper_present).

    reasoning_text is everything BEFORE the first <answer> open tag (lstripped).
    If no wrapper found, reasoning_text = full response and answer_text = None.
    """
    if not response:
        return "", None, False
    m_open = ANSWER_OPEN_RE.search(response)
    if not m_open:
        return response, None, False
    reasoning = response[: m_open.start()]
    rest = response[m_open.end():]
    m_close = ANSWER_CLOSE_RE.search(rest)
    if m_close:
        answer = rest[: m_close.start()]
    else:
        answer = rest
    return reasoning, answer, True


# ---------------- per-constraint checkers ----------------

def _check_no_comma(text: str) -> bool:
    """Punctuation: no comma anywhere in the text. Both ASCII and full-width."""
    return ("," not in text) and ("，" not in text)


def _check_number_words(text: str, args: Dict[str, Any]) -> bool:
    """Length: number of words satisfies the lower/upper bound from args.

    The MathIF/ReasonIF args may use any of these keys:
      num_words / N / max_words / min_words / target / relation
    Defaults to 'less than 200 words' when ambiguous.
    """
    wc = _word_count(text)
    if not args:
        return wc <= 200
    if "num_words" in args and "relation" in args:
        n = int(args["num_words"])
        rel = str(args["relation"]).lower().strip()
        if "less" in rel:
            return wc < n
        if "more" in rel or "great" in rel:
            return wc > n
        if "least" in rel:
            return wc >= n
        if "most" in rel:
            return wc <= n
        if "exact" in rel or rel == "equal" or rel == "==":
            return wc == n
        return wc <= n
    for key, op in [
        ("max_words", lambda v: wc <= v),
        ("maximum_words", lambda v: wc <= v),
        ("min_words", lambda v: wc >= v),
        ("minimum_words", lambda v: wc >= v),
        ("exact_words", lambda v: wc == v),
        ("num_words", lambda v: wc <= v),
        ("N", lambda v: wc <= v),
    ]:
        if key in args:
            try:
                return op(int(args[key]))
            except Exception:
                pass
    return wc <= 200


def _check_reasoning_language(text: str, args: Dict[str, Any]) -> bool:
    """Language: text body must be in the requested language.

    args typically has language / reasoning_language. Common values:
      'chinese', 'english', 'spanish', 'french', 'german', 'japanese', 'korean'.
    """
    lang = ""
    if args:
        for key in ("language", "reasoning_language", "target_language"):
            if key in args and args[key]:
                lang = str(args[key]).lower()
                break
    if not lang:
        return True  # cannot evaluate
    s = text or ""
    n_letters = len(re.findall(r"[A-Za-z]", s))
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    n_kana = len(re.findall(r"[\u3040-\u30ff]", s))
    n_hangul = len(re.findall(r"[\uac00-\ud7af]", s))
    n_cyr = len(re.findall(r"[\u0400-\u04ff]", s))
    n_arabic = len(re.findall(r"[\u0600-\u06ff]", s))
    n_total_alphabetic = n_letters + n_cjk + n_kana + n_hangul + n_cyr + n_arabic
    if n_total_alphabetic == 0:
        return False
    if "chinese" in lang or "中文" in lang:
        return n_cjk >= max(20, 0.4 * n_total_alphabetic)
    if "japanese" in lang:
        return (n_kana + n_cjk) >= max(20, 0.4 * n_total_alphabetic)
    if "korean" in lang:
        return n_hangul >= max(20, 0.4 * n_total_alphabetic)
    if "russian" in lang or "cyrillic" in lang:
        return n_cyr >= max(20, 0.4 * n_total_alphabetic)
    if "arabic" in lang:
        return n_arabic >= max(20, 0.4 * n_total_alphabetic)
    if "english" in lang:
        return n_letters >= max(20, 0.6 * n_total_alphabetic)
    if any(x in lang for x in ("spanish", "french", "german", "italian", "portuguese", "dutch")):
        return n_letters >= max(20, 0.6 * n_total_alphabetic)
    return True


def _check_end_checker(text: str, args: Dict[str, Any]) -> bool:
    """Startend: the text must end with a specific literal string (typically a phrase)."""
    target = ""
    if args:
        for key in ("end_phrase", "ending", "suffix", "end", "phrase"):
            if key in args and args[key]:
                target = str(args[key]).strip()
                break
    if not target:
        return True
    return text.rstrip().endswith(target)


def _check_json_format(text: str) -> bool:
    """Detectable format: text must parse as JSON (object or array).

    For reasoning-trace evaluation we let any single top-level JSON in the
    text (possibly wrapped in ```json fences) qualify.
    """
    s = (text or "").strip()
    if not s:
        return False
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        json.loads(s)
        return True
    except Exception:
        # Try to grab the first {...} or [...] block
        for opener, closer in [("{", "}"), ("[", "]")]:
            i = s.find(opener)
            j = s.rfind(closer)
            if i != -1 and j > i:
                try:
                    json.loads(s[i:j+1])
                    return True
                except Exception:
                    pass
    return False


def _check_english_capital(text: str) -> bool:
    """change_case:english_capital — every English letter must be uppercase."""
    letters = re.findall(r"[A-Za-z]", text or "")
    if not letters:
        return False
    return all(c.isupper() for c in letters)


# ---------------- dispatcher ----------------

CHECKERS = {
    "punctuation:no_comma": lambda text, args: _check_no_comma(text),
    "length_constraint_checkers:number_words": _check_number_words,
    "language:reasoning_language": _check_reasoning_language,
    "startend:end_checker": _check_end_checker,
    "detectable_format:json_format": lambda text, args: _check_json_format(text),
    "change_case:english_capital": lambda text, args: _check_english_capital(text),
}


def evaluate(constraint_name: str, args: Dict[str, Any], reasoning_text: str, response_text: str) -> Dict[str, Any]:
    fn = CHECKERS.get(constraint_name)
    cat = constraint_name.split(":", 1)[0] if ":" in constraint_name else constraint_name
    if fn is None:
        return {"supported": False, "category": cat, "reasoning_passed": None, "response_passed": None, "evidence": f"unsupported:{constraint_name}"}
    try:
        rp = bool(fn(reasoning_text, args)) if fn.__code__.co_argcount == 2 else bool(fn(reasoning_text))
        wp = bool(fn(response_text, args)) if fn.__code__.co_argcount == 2 else bool(fn(response_text))
    except Exception as e:
        return {"supported": True, "category": cat, "reasoning_passed": None, "response_passed": None, "evidence": f"err:{e}"}
    return {
        "supported": True,
        "category": cat,
        "reasoning_passed": rp,
        "response_passed": wp,
        "evidence": f"reasoning_words={_word_count(reasoning_text)}; response_words={_word_count(response_text)}",
    }


def evaluate_record(response: str, constraint_names: List[str], constraint_args: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Evaluate one ReasonIF record. Returns aggregate + per-constraint detail."""
    reasoning, answer_block, wrapper = split_reasoning_and_answer(response)
    per = []
    for name, args in zip(constraint_names, constraint_args):
        per.append({"name": name, **evaluate(name, args or {}, reasoning, response)})
    supported = [p for p in per if p.get("supported")]
    if supported:
        n_r = sum(1 for p in supported if p.get("reasoning_passed"))
        n_w = sum(1 for p in supported if p.get("response_passed"))
        reasoning_strict = 1.0 if n_r == len(supported) else 0.0
        response_strict = 1.0 if n_w == len(supported) else 0.0
        reasoning_soft = n_r / len(supported)
        response_soft = n_w / len(supported)
    else:
        reasoning_strict = response_strict = reasoning_soft = response_soft = float("nan")
    return {
        "wrapper_present": wrapper,
        "reasoning_word_count": _word_count(reasoning),
        "response_word_count": _word_count(response),
        "answer_word_count": _word_count(answer_block or ""),
        "reasoning_strict_pass": reasoning_strict,
        "response_strict_pass": response_strict,
        "reasoning_soft_pass": reasoning_soft,
        "response_soft_pass": response_soft,
        "per_constraint": per,
    }
