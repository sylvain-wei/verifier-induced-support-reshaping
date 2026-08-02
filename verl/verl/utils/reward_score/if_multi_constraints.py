from typing import Any

from verl.utils.reward_score.ifeval.verifier import verify_ifeval


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info=None,
    **kwargs,
):
    _ = (data_source, extra_info, kwargs)
    result = verify_ifeval(solution_str, ground_truth)
    score = float(result.score)
    return {
        # Keep reward keys aligned with math_dapo so multi-dataset validation
        # can aggregate val-core metrics (mean@N/best@N) consistently.
        "score": score,
        "acc": score,
        # Placeholder keeps schema compatible with datasets that emit `pred`.
        # String fields are ignored by validation metric reducer.
        "pred": solution_str,
    }
