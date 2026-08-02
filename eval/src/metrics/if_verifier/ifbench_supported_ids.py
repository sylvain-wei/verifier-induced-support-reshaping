"""
Which instruction IDs are rule-based ("format-only", no LLM judge)?

We maintain TWO registries:

1. IFEval (vendored from verl.bak / google IFEval) — 54 IDs.
2. IFBench (vendored from allenai/IFBench) — 58 IDs.

Both are pure rule-based verifiers. An example is "format-only" iff every
constraint in the example is in at least one registry, i.e. every constraint
can be mechanically verified without an LLM judge.

The IFBench loader (`src/data/converters.py::ifbench_format_only`) uses
`is_format_only_example` to drop anything that references an unsupported ID
(rather than silently scoring it zero).
"""
from __future__ import annotations

from typing import Iterable

from .instructions_registry import INSTRUCTION_DICT as IFEVAL_DICT

try:
    from ..if_verifier_ifbench.instructions_registry import INSTRUCTION_DICT as IFBENCH_DICT
except Exception:
    IFBENCH_DICT = {}

FORMAT_ONLY_SUPPORTED_IDS = frozenset(set(IFEVAL_DICT.keys()) | set(IFBENCH_DICT.keys()))


def is_format_only_example(instruction_ids: Iterable[str]) -> bool:
    """True iff every constraint id in the example is covered by at least
    one of the bundled rule-based registries."""
    ids = list(instruction_ids)
    if not ids:
        return False
    return all(i in FORMAT_ONLY_SUPPORTED_IDS for i in ids)
