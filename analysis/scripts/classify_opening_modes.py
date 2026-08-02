#!/usr/bin/env python
"""High-precision opening-mode classifier for AIME/IFEval/IFBench rollouts.

4-class taxonomy (paper terminology):

  - DRI : Deliberative Reasoning Initiation
          Begins with explicit step / planning / problem-restatement language.
          Examples: "Step 1:", "To solve …", "Let's solve …", "First, …",
                    "We need to …", "We are asked …", "I'll solve …"

  - DAI : Direct Answer Initiation
          Begins with answer-style declaration with no reasoning preamble.
          Examples: "Answer:", "The answer is …", "Final answer …",
                    "Therefore the answer …", \\boxed{…}, plain numeric prefix.

  - CSI : Constraint-Surface Initiation (IF tasks only — needs prompt context)
          Begins by directly emitting the surface tokens that the verifier checks
          (placeholder-front, keyword-dump, language-marker dump). Requires the
          IF prompt's constraint metadata, so this category is only assigned
          when `inst_ids` / `prompt` are passed.

  - Other : everything else (filler, off-topic, refusal, partial reasoning).

Pattern rules use ASCII-only tokens and are case-sensitive where it matters,
case-insensitive where it doesn't (logged in detail in
`appendix_pattern_rules.md`). All patterns operate on the response *after*
`lstrip()` and look only at the first ~120 chars.

Usage
-----
  from classify_opening_modes import classify_mode
  m = classify_mode(response, prompt=None, inst_ids=None)   # AIME default
  m = classify_mode(response, prompt=ifeval_prompt, inst_ids=ifeval_inst_ids)
"""
from __future__ import annotations

import re
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# DRI patterns: deliberative reasoning initiation.
# ---------------------------------------------------------------------------
_DRI_CASE_SENSITIVE = (
    "Step 1:", "Step 1.", "Step 1 ", "Step1:", "Step1.",
    "To solve", "To find", "To compute", "To determine",
    "First,", "First we ", "First, we", "First let",
    "We first", "We need", "We are asked", "We have", "We want",
    "I'll solve", "I will solve", "I need to",
    "Let's solve", "Let's break", "Let's tackle", "Let's compute",
    "Let's find", "Let's denote", "Let's compute", "Let's see",
    "Let us solve", "Let us break", "Let us tackle", "Let us denote",
    "Let me solve", "Let me find",
    # Qwen3-style reasoning preambles (very common after Math-RLVR)
    "Alright,",   "Okay,",     "OK,",      "Right,",
    "Alright ",   "Okay ",     "OK ",
    "So,",        "So ",       "Now,",     "Now ",
    # generic problem-restatement openers also count as DRI
    "Given ",     "Given,",    "Suppose",  "Consider",  "Define",
    "Note that",  "Recall",    "Observe",
)

_DRI_LOWER = (
    # case-insensitive variants (already covered by lower comparison)
    "to solve", "to find", "to compute", "to determine",
    "let's solve", "let's break", "let's tackle", "let's compute",
    "let's denote", "let's set", "let's find", "let's see",
    "let us solve", "let us break", "let us tackle", "let us denote",
    "we need to", "we have to", "we want to", "we are asked", "we first",
    "first,", "first we ", "first, we",
    "i'll solve", "i will solve", "i need to",
    "step 1:", "step 1.", "step1:", "step1.",
    "alright,", "alright ", "okay,", "okay ", "ok,",
    "so,", "now,", "given ", "suppose", "consider", "define",
    "note that", "recall", "observe",
)

# ---------------------------------------------------------------------------
# DAI patterns: direct answer initiation.
# ---------------------------------------------------------------------------
_DAI_CASE_SENSITIVE = (
    "Answer:",
    "Final answer:", "Final Answer:",
    "The answer is", "The final answer is",
    "Therefore, the answer", "Therefore the answer",
    "Thus, the answer", "Thus the answer",
    "So, the answer", "So the answer",
    "Hence, the answer", "Hence the answer",
    "ANSWER:",
)

_DAI_LOWER = tuple(s.lower() for s in _DAI_CASE_SENSITIVE) + (
    "answer is",  # safety net
)

# Bare numeric / boxed openings (treated as DAI).
_BARE_NUMERIC_RE = re.compile(r"^\s*\\?boxed\{|^\s*\$?-?\d+(\.\d+)?[\s\.\,\$]")
_LATEX_BOX_RE   = re.compile(r"^\s*\\boxed\s*\{")


# ---------------------------------------------------------------------------
# CSI: constraint-surface initiation (IF only).
# ---------------------------------------------------------------------------
def _matches_csi(response: str, prompt: str, inst_ids: Sequence[str]) -> bool:
    """Heuristic: response begins by dumping verifier-surface tokens.

    Two triggers (any of):
      (a) Placeholder-front: the prompt requires `[placeholder]` (instruction
          ids containing 'placeholder') AND the response's first 30 chars
          contains a literal "[placeholder]" or "[<word>]" token.
      (b) Keyword-front-dump: the prompt requires keyword inclusion
          (instruction id contains 'keywords') AND the first 80 chars contain
          ≥2 of those keywords (extracted via regex on the prompt) AND the
          response is short (<=300 chars total).
    """
    if not inst_ids:
        return False
    head = response.lstrip()[:80]

    has_placeholder = any("placeholder" in i for i in inst_ids)
    if has_placeholder:
        if re.search(r"\[\s*placeholder\s*\]", head, flags=re.IGNORECASE):
            return True
        # generic [word] token in head — a common surface pattern
        if re.search(r"^\[[a-zA-Z][a-zA-Z\s_-]{1,30}\]", head):
            return True

    has_keywords = any("keywords" in i for i in inst_ids)
    if has_keywords:
        # extract keyword candidates from prompt: tokens in single quotes
        # like 'foo' or "foo". This is heuristic but high-precision.
        kws = re.findall(r"['\"]([A-Za-z][\w\s-]{1,40}?)['\"]", prompt or "")
        if kws:
            head_lower = head.lower()
            hits = sum(1 for k in kws if k.lower() in head_lower)
            if hits >= 2 and len(response) <= 300:
                return True
    return False


def classify_mode(
    response: str,
    prompt: Optional[str] = None,
    inst_ids: Optional[Sequence[str]] = None,
    enable_csi: bool = False,
) -> str:
    """Return 'DRI' / 'DAI' / 'CSI' / 'Other'.

    Args:
      response: raw rollout text.
      prompt:   IF prompt text (only consulted for CSI).
      inst_ids: list of instruction id strings (only for CSI).
      enable_csi: must be True to enable CSI (avoid spurious CSI on AIME rows).
    """
    s = (response or "").lstrip()

    # CSI takes precedence in IF mode (often "[placeholder] X" looks like DAI).
    if enable_csi and prompt is not None and inst_ids:
        if _matches_csi(response, prompt, inst_ids):
            return "CSI"

    # DAI: case-sensitive prefix
    for p in _DAI_CASE_SENSITIVE:
        if s.startswith(p):
            return "DAI"
    # DAI: bare number / \boxed{...}
    if _LATEX_BOX_RE.match(s) or _BARE_NUMERIC_RE.match(s):
        return "DAI"

    # DRI: case-sensitive prefix
    for p in _DRI_CASE_SENSITIVE:
        if s.startswith(p):
            return "DRI"

    # Lowercase fallback — catches sentence-case variants.
    sl = s.lower()
    for p in _DAI_LOWER:
        if sl.startswith(p):
            return "DAI"
    for p in _DRI_LOWER:
        if sl.startswith(p):
            return "DRI"

    return "Other"


def classify_dataframe(df, *, response_col="response", prompt_col="prompt",
                      inst_ids_col="instruction_id_list", benchmark_col="benchmark"):
    """Vectorized application onto a consolidated rollout DataFrame.

    CSI is auto-enabled for IF benchmarks (ifeval, ifbench).
    """
    modes = []
    for _, r in df.iterrows():
        bench = r.get(benchmark_col, "")
        is_if = bench in ("ifeval", "ifbench")
        modes.append(classify_mode(
            r[response_col],
            prompt=r.get(prompt_col) if is_if else None,
            inst_ids=r.get(inst_ids_col) if is_if else None,
            enable_csi=is_if,
        ))
    df = df.copy()
    df["opening_mode"] = modes
    return df


def main():
    """CLI: load consolidated rollouts, classify, write back with opening_mode col."""
    import argparse
    import os
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="parquet path")
    ap.add_argument("--output", default=None,
                    help="parquet path (default: in-place overwrite)")
    args = ap.parse_args()
    df = pd.read_parquet(args.input)
    df = classify_dataframe(df)
    out = args.output or args.input
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")
    print(df.groupby(["benchmark", "opening_mode"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
