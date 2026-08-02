import ast
import json
from dataclasses import dataclass
from typing import Any

from langdetect import DetectorFactory

from .instructions_registry import INSTRUCTION_DICT

# Fallback registry for IFBench OOD constraints (58 ids not present in
# the in-distribution INSTRUCTION_DICT). Lazy-imported so missing optional
# deps (emoji/syllapy) only fail when an OOD id is actually evaluated.
try:
    from ..ifbench.instructions_registry import INSTRUCTION_DICT as IFBENCH_DICT
except Exception:  # pragma: no cover
    IFBENCH_DICT = {}

DetectorFactory.seed = 0


@dataclass
class IFEvalResult:
    score: float
    num_constraints: int
    num_satisfied: int


def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def parse_ground_truth(label: Any) -> dict:
    obj = label
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    if isinstance(obj, list):
        if not obj:
            raise ValueError("Empty ground_truth list")
        obj = obj[0]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported ground_truth type: {type(obj)}")
    return obj


def _check_one(instruction_key: str, args: dict, answer: str) -> bool:
    """Run one instruction checker. Tries IFEval registry first, then IFBench."""
    non_none_args = {k: v for k, v in (args or {}).items() if v is not None}
    # 1) IFEval / IFTrain (in-distribution): SimpleInstruction-style API
    checker_cls = INSTRUCTION_DICT.get(instruction_key)
    if checker_cls is not None:
        checker = checker_cls(instruction_key)
        checker.build_description(**non_none_args)
        return bool(checker.check_following(answer))
    # 2) IFBench OOD: upstream allenai/IFBench Instruction-style API
    checker_cls = IFBENCH_DICT.get(instruction_key)
    if checker_cls is not None:
        try:
            checker = checker_cls(instruction_key)
            checker.build_description(**non_none_args)
            return bool(checker.check_following(answer))
        except Exception:
            return False
    # Unknown id: count as failure (do not silently award credit).
    return False


def verify_ifeval(prediction: str, label: Any) -> IFEvalResult:
    answer = remove_thinking_section(prediction)
    if not answer:
        return IFEvalResult(score=0.0, num_constraints=0, num_satisfied=0)

    try:
        constraint = parse_ground_truth(label)
    except Exception:
        return IFEvalResult(score=0.0, num_constraints=0, num_satisfied=0)

    instruction_keys = constraint.get("instruction_id", [])
    args_list = constraint.get("kwargs", [])
    if not instruction_keys:
        return IFEvalResult(score=0.0, num_constraints=0, num_satisfied=0)

    # Keep compatibility for malformed labels where kwargs is missing/shorter.
    if len(args_list) < len(instruction_keys):
        args_list = list(args_list) + [None] * (len(instruction_keys) - len(args_list))

    rewards: list[float] = []
    for instruction_key, args in zip(instruction_keys, args_list):
        rewards.append(1.0 if _check_one(instruction_key, args, answer) else 0.0)

    num_constraints = len(rewards)
    num_satisfied = int(sum(rewards))
    score = 0.0 if num_constraints == 0 else num_satisfied / num_constraints
    return IFEvalResult(score=score, num_constraints=num_constraints, num_satisfied=num_satisfied)
