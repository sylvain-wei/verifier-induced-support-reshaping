"""
Engine abstraction. Only one concrete backend today: vLLM.

Contract:
  Engine(model_cfg, engine_cfg) -> instance
  instance.generate(prompts, decoding) -> List[GenOut]
  instance.shutdown() -> None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenOut:
    text: str
    finish_reason: Optional[str]
    gen_tokens: int
    seed: int


@dataclass
class Decoding:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int  # per-request seed; vLLM SamplingParams exposes `seed`
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)


class BaseEngine:
    def __init__(self, model_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.engine_cfg = engine_cfg

    def generate(self, prompts: List[str], decoding: Decoding) -> List[GenOut]:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass
