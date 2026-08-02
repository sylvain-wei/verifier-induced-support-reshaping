"""
Unified IFEval / IFBench verifier entrypoint.

IFEval uses the rule-based verifier vendored from verl.bak (under
`src/metrics/if_verifier/`). IFBench uses the official rule-based verifier
vendored from github.com/allenai/IFBench (under
`src/metrics/if_verifier_ifbench/`). Both return the same IFResult shape.

Given a response string and a list of (instruction_id, kwargs), returns
per-constraint pass/fail plus aggregates needed by `if_metrics.py`:

  - strict_prompt_pass: all constraints satisfied
  - instruction_level_pass_rate: fraction of constraints satisfied
  - per-constraint records, including constraint_type (prefix)

Unsupported instruction_id is reported explicitly: pass=False + supported=False.
The caller can aggregate `supported` to report coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConstraintResult:
    instruction_id: str
    supported: bool
    passed: bool
    error: Optional[str] = None


@dataclass
class IFResult:
    num_constraints: int
    num_supported: int
    num_satisfied: int
    strict_prompt_pass: bool
    instruction_level_pass_rate: float
    format_pass_rate: float
    per_constraint: List[ConstraintResult] = field(default_factory=list)
    num_format_constraints: int = 0
    num_format_passed: int = 0


def remove_thinking_section(prediction: str) -> str:
    s = (prediction or "").replace("<|assistant|>", "").strip()
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = s.replace("<answer>", "").replace("</answer>", "")
    return s.strip()


def _is_format_type_ifeval(instruction_id: str) -> bool:
    """IFEval-style: format prefixes we count towards `format_pass_rate`."""
    prefix = instruction_id.split(":", 1)[0]
    return prefix in {
        "detectable_format",
        "detectable_content",
        "paragraphs",
        "length_constraints",
        "punctuation",
        "startend",
        "change_case",
        "combination",
        "copy",
        "new",
    }


def _is_format_type_ifbench(instruction_id: str) -> bool:
    """IFBench uses a different prefix scheme; all its constraints are pure
    format/content rules (no LLM judge). We count the `format:*` family as
    `format_pass_rate` and leave the broader `instruction_level_pass_rate`
    for everything else (count/ratio/words/sentence/repeat/custom)."""
    return instruction_id.startswith("format:")


def _registry_for(benchmark: Optional[str]) -> Dict[str, Any]:
    if benchmark and benchmark.lower().startswith("ifbench"):
        from .if_verifier_ifbench.instructions_registry import INSTRUCTION_DICT as D
    else:
        from .if_verifier.instructions_registry import INSTRUCTION_DICT as D
    return D


def _is_format_type_for(benchmark: Optional[str], instruction_id: str) -> bool:
    if benchmark and benchmark.lower().startswith("ifbench"):
        return _is_format_type_ifbench(instruction_id)
    return _is_format_type_ifeval(instruction_id)


def verify(
    prediction: str,
    instruction_ids: List[str],
    kwargs_list: List[Dict[str, Any]],
    benchmark: Optional[str] = None,
) -> IFResult:
    """Top-level entrypoint. Picks IFEval or IFBench registry based on
    `benchmark` (case-insensitive prefix match)."""
    answer = remove_thinking_section(prediction)
    registry = _registry_for(benchmark)

    # Pad kwargs so callers don't have to.
    if len(kwargs_list) < len(instruction_ids):
        kwargs_list = list(kwargs_list) + [{}] * (len(instruction_ids) - len(kwargs_list))

    per: List[ConstraintResult] = []
    num_satisfied = 0
    num_supported = 0
    num_format_constraints = 0
    num_format_passed = 0

    for instruction_id, args in zip(instruction_ids, kwargs_list):
        checker_cls = registry.get(instruction_id)
        is_format = _is_format_type_for(benchmark, instruction_id)

        if checker_cls is None:
            per.append(ConstraintResult(instruction_id=instruction_id, supported=False, passed=False, error="unsupported"))
            continue

        num_supported += 1
        if is_format:
            num_format_constraints += 1

        if not answer:
            per.append(ConstraintResult(instruction_id=instruction_id, supported=True, passed=False, error=None))
            continue

        try:
            non_none_args = {} if args is None else {k: v for k, v in args.items() if v is not None}
            checker = checker_cls(instruction_id)
            checker.build_description(**non_none_args)
            ok = bool(checker.check_following(answer))
        except Exception as e:
            per.append(ConstraintResult(instruction_id=instruction_id, supported=True, passed=False, error=f"{type(e).__name__}: {e}"))
            continue

        per.append(ConstraintResult(instruction_id=instruction_id, supported=True, passed=ok, error=None))
        if ok:
            num_satisfied += 1
            if is_format:
                num_format_passed += 1

    num_constraints = len(instruction_ids)
    strict_prompt_pass = (num_constraints > 0) and (num_satisfied == num_constraints)
    instruction_level_pass_rate = (num_satisfied / num_constraints) if num_constraints else 0.0
    format_pass_rate = (num_format_passed / num_format_constraints) if num_format_constraints else float("nan")

    return IFResult(
        num_constraints=num_constraints,
        num_supported=num_supported,
        num_satisfied=num_satisfied,
        strict_prompt_pass=strict_prompt_pass,
        instruction_level_pass_rate=instruction_level_pass_rate,
        format_pass_rate=format_pass_rate,
        per_constraint=per,
        num_format_constraints=num_format_constraints,
        num_format_passed=num_format_passed,
    )


# Backwards-compat helpers that keep the original API used elsewhere.
def verify_ifeval(
    prediction: str,
    instruction_ids: List[str],
    kwargs_list: List[Dict[str, Any]],
) -> IFResult:
    return verify(prediction, instruction_ids, kwargs_list, benchmark="ifeval")


def verify_ifbench(
    prediction: str,
    instruction_ids: List[str],
    kwargs_list: List[Dict[str, Any]],
) -> IFResult:
    return verify(prediction, instruction_ids, kwargs_list, benchmark="ifbench")
