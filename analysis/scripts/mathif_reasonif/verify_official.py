#!/usr/bin/env python
"""Strict verifiers for MathIF and ReasonIF instruction types.

The functions here implement the constraint families used by both datasets:
  - punctuation:no_comma
  - length_constraint_checkers:number_words
  - language:reasoning_language / language:response_language
  - startend:end_checker / startend:quotation
  - detectable_format:json_format / number_highlighted_sections /
    number_bullet_lists / multiple_sections
  - change_case:english_capital / english_lowercase / capital_word_frequency
  - keywords:frequency / existence / forbidden_words
  - combination:repeat_prompt
  - punctuation:* (no_comma)

For ReasonIF the verifier targets the *reasoning trace* (the text outside the
final `<answer>` tag) because the dataset prompts explicitly attach the
constraint to reasoning content.

For MathIF we apply the constraint to the *whole response*, matching the
official MathIF scorer convention.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

ANSWER_BLOCK_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)


def split_reasoning_and_answer(text: str) -> Tuple[str, str, bool]:
    """Return (reasoning_trace, final_answer, has_tag).

    reasoning_trace = text before the FIRST <answer> tag.
    final_answer    = first content inside <answer>...</answer>.
    has_tag         = whether an <answer> tag is present at all.
    """
    if not text:
        return "", "", False
    m = ANSWER_BLOCK_RE.search(text)
    if m:
        trace = text[: m.start()]
        ans = m.group(1)
        return trace.strip(), ans.strip(), True
    m2 = ANSWER_OPEN_RE.search(text)
    if m2:
        return text[: m2.start()].strip(), text[m2.end():].strip(), True
    return text.strip(), "", False


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w\-']+\b", text or ""))


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _strip_md_decor(text: str) -> str:
    """Strip markdown emphasis/code fences when checking content rules."""
    if not text:
        return ""
    out = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    out = re.sub(r"`[^`\n]+`", " ", out)
    out = re.sub(r"\*+|_+|~+", " ", out)
    return out


# ---------------------------------------------------------------------------
# individual constraint verifiers
# ---------------------------------------------------------------------------

def check_punctuation_no_comma(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    body = text or ""
    has_comma = bool(re.search(r",", body))
    return {"supported": True, "passed": not has_comma,
            "evidence": f"comma_present={has_comma}, len={len(body)}"}


def check_length_number_words(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    n_min = args.get("num_words_min") or args.get("min_words") or args.get("num_words_lower")
    n_max = args.get("num_words_max") or args.get("max_words") or args.get("num_words_upper")
    target = args.get("num_words") or args.get("n_words")
    rel = (args.get("relation") or args.get("comparator") or args.get("relation_type") or "").lower()

    wc = _word_count(text)
    ok = True
    why = [f"words={wc}"]
    if target is not None:
        try:
            target_i = int(target)
        except Exception:
            target_i = None
        if target_i is not None:
            if rel in {"at most", "less than or equal to", "<=", "le"}:
                ok = ok and (wc <= target_i)
                why.append(f"<= {target_i}")
            elif rel in {"at least", "greater than or equal to", ">=", "ge"}:
                ok = ok and (wc >= target_i)
                why.append(f">= {target_i}")
            elif rel in {"less than", "<", "lt"}:
                ok = ok and (wc < target_i)
                why.append(f"< {target_i}")
            elif rel in {"more than", ">", "gt"}:
                ok = ok and (wc > target_i)
                why.append(f"> {target_i}")
            elif rel in {"equal to", "=", "eq", "exactly"}:
                ok = ok and (wc == target_i)
                why.append(f"== {target_i}")
            else:
                ok = ok and (wc == target_i)
                why.append(f"~= {target_i}")
    if n_min is not None:
        try:
            ok = ok and (wc >= int(n_min))
            why.append(f">= {int(n_min)}")
        except Exception:
            pass
    if n_max is not None:
        try:
            ok = ok and (wc <= int(n_max))
            why.append(f"<= {int(n_max)}")
        except Exception:
            pass
    if target is None and n_min is None and n_max is None:
        # No interpretable bound => not supported
        return {"supported": False, "passed": None, "evidence": f"words={wc}, args={args}"}
    return {"supported": True, "passed": ok, "evidence": "; ".join(why)}


def check_language_reasoning(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    lang = (args.get("language") or args.get("reasoning_language") or "").lower()
    if not lang:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    body = text or ""
    if lang in {"en", "english"}:
        # English: predominantly latin chars, very few CJK
        latin = len(re.findall(r"[A-Za-z]", body))
        non_ascii = len(re.findall(r"[^\x00-\x7f]", body))
        ok = latin >= max(20, non_ascii * 5)
        return {"supported": True, "passed": ok,
                "evidence": f"latin={latin}, non_ascii={non_ascii}"}
    if lang in {"zh", "chinese", "zh-cn"}:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
        latin = len(re.findall(r"[A-Za-z]", body))
        ok = cjk >= max(20, latin // 2)
        return {"supported": True, "passed": ok,
                "evidence": f"cjk={cjk}, latin={latin}"}
    if lang in {"es", "spanish", "fr", "french", "de", "german", "ru", "russian"}:
        # Very loose: require >50 latin chars and presence of language-specific letters
        latin = len(re.findall(r"[A-Za-z]", body))
        markers = {
            "es": r"[ñáéíóúü¿¡]", "spanish": r"[ñáéíóúü¿¡]",
            "fr": r"[àâçéèêëîïôûùü]", "french": r"[àâçéèêëîïôûùü]",
            "de": r"[äöüß]", "german": r"[äöüß]",
            "ru": r"[а-яё]", "russian": r"[а-яё]",
        }
        pat = markers.get(lang)
        ok = latin >= 50 and (not pat or bool(re.search(pat, body, flags=re.IGNORECASE)))
        return {"supported": True, "passed": ok,
                "evidence": f"latin={latin}, lang={lang}"}
    return {"supported": False, "passed": None, "evidence": f"lang={lang}"}


def check_startend_end(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    phrase = (args.get("end_phrase") or args.get("ending") or args.get("end")
              or args.get("phrase"))
    if not phrase:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    norm = re.sub(r"\s+", " ", (text or "").strip()).rstrip('."\')]')
    target = re.sub(r"\s+", " ", str(phrase).strip()).rstrip('."\')]')
    ok = norm.lower().endswith(target.lower())
    return {"supported": True, "passed": ok,
            "evidence": f"target={phrase!r}, tail={norm[-len(target)-5:][:80]!r}"}


def check_startend_quotation(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {"supported": True, "passed": False, "evidence": "empty"}
    starts = body[0] in '"\'\u201c\u2018'
    ends = body[-1] in '"\'\u201d\u2019'
    return {"supported": True, "passed": starts and ends,
            "evidence": f"start={body[:1]!r}, end={body[-1:]!r}"}


def check_format_json(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    body = (text or "").strip()
    candidates = [body]
    m = re.search(r"```(?:json)?\s*(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidates.insert(0, m.group(1).strip())
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, (dict, list)):
                return {"supported": True, "passed": True, "evidence": "json.loads ok"}
        except Exception:
            continue
    return {"supported": True, "passed": False, "evidence": "json.loads failed"}


def check_format_highlighted(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    n = args.get("num_highlights") or args.get("num_highlighted_sections") or args.get("count") or 1
    try:
        n = int(n)
    except Exception:
        n = 1
    body = text or ""
    bold = len(re.findall(r"\*\*[^*\n]+\*\*", body))
    italic = len(re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", body))
    high = bold + italic
    return {"supported": True, "passed": high >= n,
            "evidence": f"bold={bold}, italic={italic}, target>={n}"}


def check_format_bullets(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    n = args.get("num_bullets") or args.get("num_bullet_lists") or args.get("count") or 1
    try:
        n = int(n)
    except Exception:
        n = 1
    body = text or ""
    bullets = len(re.findall(r"(?m)^\s*(?:[-*+]\s+|\d+[\.)]\s+)", body))
    return {"supported": True, "passed": bullets >= n,
            "evidence": f"bullets={bullets}, target>={n}"}


def check_format_sections(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    n = args.get("num_sections") or args.get("count") or 2
    try:
        n = int(n)
    except Exception:
        n = 2
    body = text or ""
    headers = len(re.findall(r"(?m)^\s*(?:#+\s+|Section\s+\d+|SECTION\s+\d+|---+)", body))
    return {"supported": True, "passed": headers >= n,
            "evidence": f"headers={headers}, target>={n}"}


def check_case_capital(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    body = text or ""
    letters = re.findall(r"[A-Za-z]", body)
    if not letters:
        return {"supported": True, "passed": False, "evidence": "no letters"}
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return {"supported": True, "passed": upper_ratio >= 0.95,
            "evidence": f"upper_ratio={upper_ratio:.2f}"}


def check_case_lowercase(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    body = text or ""
    letters = re.findall(r"[A-Za-z]", body)
    if not letters:
        return {"supported": True, "passed": False, "evidence": "no letters"}
    lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)
    return {"supported": True, "passed": lower_ratio >= 0.95,
            "evidence": f"lower_ratio={lower_ratio:.2f}"}


def check_case_capital_word_frequency(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    n = args.get("capital_frequency") or args.get("frequency") or args.get("num") or 1
    rel = (args.get("relation") or "at least").lower()
    try:
        n = int(n)
    except Exception:
        n = 1
    body = text or ""
    cap_words = sum(1 for w in re.findall(r"\b[\w\-']+\b", body) if w.isupper() and len(w) > 1)
    if "at most" in rel or "less" in rel:
        ok = cap_words <= n
    else:
        ok = cap_words >= n
    return {"supported": True, "passed": ok,
            "evidence": f"capword={cap_words}, target {rel} {n}"}


def check_keywords_frequency(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    keyword = args.get("keyword") or args.get("word") or args.get("token")
    n = args.get("frequency") or args.get("num") or args.get("count") or 1
    rel = (args.get("relation") or "at least").lower()
    if not keyword:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    body = (text or "").lower()
    cnt = len(re.findall(r"\b" + re.escape(str(keyword).lower()) + r"\b", body))
    try:
        n = int(n)
    except Exception:
        n = 1
    if "at most" in rel:
        ok = cnt <= n
    elif "less" in rel:
        ok = cnt < n
    elif "exact" in rel or rel.strip() in {"=", "==", "eq"}:
        ok = cnt == n
    else:
        ok = cnt >= n
    return {"supported": True, "passed": ok,
            "evidence": f"keyword={keyword!r}, count={cnt}, target {rel} {n}"}


def check_keywords_existence(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    kws = args.get("keywords") or args.get("required") or []
    if isinstance(kws, str):
        kws = [kws]
    if not kws:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    body = (text or "").lower()
    missing = [k for k in kws if k.lower() not in body]
    return {"supported": True, "passed": not missing,
            "evidence": f"missing={missing}"}


def check_keywords_forbidden(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    kws = args.get("forbidden_words") or args.get("forbidden") or args.get("avoid") or []
    if isinstance(kws, str):
        kws = [kws]
    if not kws:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    body = (text or "").lower()
    hits = [k for k in kws if k.lower() in body]
    return {"supported": True, "passed": not hits, "evidence": f"hits={hits}"}


def check_combination_repeat_prompt(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    target = args.get("prompt_to_repeat") or args.get("prompt") or args.get("request")
    if not target:
        return {"supported": False, "passed": None, "evidence": f"args={args}"}
    body = (text or "").strip()
    needle = re.sub(r"\s+", " ", str(target).strip())
    hay = re.sub(r"\s+", " ", body)
    return {"supported": True, "passed": needle in hay,
            "evidence": f"prefix_hit={hay.startswith(needle)}, contains={needle in hay}"}


def check_response_language(text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return check_language_reasoning(text, args)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

CHECKERS = {
    "punctuation:no_comma": check_punctuation_no_comma,
    "length_constraint_checkers:number_words": check_length_number_words,
    "language:reasoning_language": check_language_reasoning,
    "language:response_language": check_response_language,
    "startend:end_checker": check_startend_end,
    "startend:quotation": check_startend_quotation,
    "detectable_format:json_format": check_format_json,
    "detectable_format:number_highlighted_sections": check_format_highlighted,
    "detectable_format:number_bullet_lists": check_format_bullets,
    "detectable_format:multiple_sections": check_format_sections,
    "change_case:english_capital": check_case_capital,
    "change_case:english_lowercase": check_case_lowercase,
    "change_case:capital_word_frequency": check_case_capital_word_frequency,
    "keywords:frequency": check_keywords_frequency,
    "keywords:existence": check_keywords_existence,
    "keywords:forbidden_words": check_keywords_forbidden,
    "combination:repeat_prompt": check_combination_repeat_prompt,
}


def family_of(name: str) -> str:
    return (name or "").split(":", 1)[0]


def verify_one(text: str, name: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    args = args or {}
    fn = CHECKERS.get(name)
    if fn is None:
        return {"supported": False, "passed": None, "evidence": f"no_checker_for {name!r}"}
    out = fn(text or "", args)
    out["instruction_name"] = name
    out["family"] = family_of(name)
    return out


def verify_constraints(text: str, names: List[str], args_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    per: List[Dict[str, Any]] = []
    for i, name in enumerate(names or []):
        a = args_list[i] if i < len(args_list or []) else {}
        if not isinstance(a, dict):
            a = {}
        per.append(verify_one(text, name, a))
    supported = [p for p in per if p.get("supported")]
    if not supported:
        return {"per": per, "follow_soft": float("nan"),
                "follow_strict": float("nan"),
                "n_supported": 0, "n_passed": 0}
    passed = [p for p in supported if p.get("passed") is True]
    return {"per": per, "follow_soft": len(passed) / len(supported),
            "follow_strict": 1.0 if len(passed) == len(supported) else 0.0,
            "n_supported": len(supported), "n_passed": len(passed)}
