"""
Utility: seed management.

Global seed defaults to 42. Per-sample seed for stochastic generation is
`global_seed + sample_id`, so that each (example, sample_id) pair is
independently seeded and reproducible across runs.
"""
from __future__ import annotations

import os
import random
from typing import Optional

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed Python random, numpy, and torch CPU. We intentionally do NOT
    call `torch.cuda.manual_seed*` here, because that would trigger CUDA
    initialization in the parent process, which breaks vLLM's fork-based
    engine worker startup ("Cannot re-initialize CUDA in forked subprocess").
    vLLM itself seeds CUDA inside the engine core process when we pass
    `seed=` to LLM() and `seed=` to SamplingParams.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        # NOTE: do NOT touch torch.cuda here (see docstring).
    except Exception:
        pass


def sample_seed(sample_id: int, global_seed: int = GLOBAL_SEED) -> int:
    """Deterministic per-sample seed."""
    return int(global_seed) + int(sample_id)


def seed_env_for_vllm(seed: Optional[int] = None) -> None:
    """vLLM picks up VLLM_USE_V1 etc. from env; we just set PYTHONHASHSEED."""
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
