"""
vLLM-backed inference engine.

We instantiate vLLM.LLM once per (model, script invocation) — this is the
expensive step. Then we call `generate` with batched prompts per decoding
config.

Notes:
- We use `SamplingParams.seed` per call, allowing reproducibility for a
  fixed seed. When a batch of prompts has per-sample seeds (e.g., K samples
  of the same prompt with seed=42, 43, 44, ...), we dispatch them as
  separate SamplingParams objects via vLLM's `generate(prompts,
  sampling_params: list)`.
- We always pass `skip_special_tokens=False` on decode so that we can
  detect EOS in the raw text if needed; however for saving we strip the
  raw response via tokenizer.decode skip_special_tokens=True.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .engine import BaseEngine, Decoding, GenOut


class VLLMEngine(BaseEngine):
    def __init__(self, model_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]):
        super().__init__(model_cfg, engine_cfg)
        # Avoid "Cannot re-initialize CUDA in forked subprocess" by forcing
        # the vLLM engine subprocess to use spawn. This must be set BEFORE
        # `import vllm.LLM` creates the engine.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM
        self.LLM = LLM
        tp = int(os.environ.get("TENSOR_PARALLEL_SIZE", engine_cfg.get("tensor_parallel_size", 1)))
        kwargs = dict(
            model=model_cfg["path"],
            tokenizer=model_cfg.get("tokenizer_path") or model_cfg["path"],
            dtype=engine_cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=float(engine_cfg.get("gpu_memory_utilization", 0.88)),
            max_model_len=int(engine_cfg.get("max_model_len", 8192)),
            trust_remote_code=bool(engine_cfg.get("trust_remote_code", False)),
            enforce_eager=bool(engine_cfg.get("enforce_eager", False)),
            tensor_parallel_size=tp,
            seed=int(engine_cfg.get("seed", 42)),
        )
        self._llm = self.LLM(**kwargs)
        self._tp = tp

    def _sp(self, dec: Decoding):
        from vllm import SamplingParams
        return SamplingParams(
            n=1,
            temperature=float(dec.temperature),
            top_p=float(dec.top_p),
            max_tokens=int(dec.max_new_tokens),
            repetition_penalty=float(dec.repetition_penalty),
            seed=int(dec.seed),
            stop=list(dec.stop),
            skip_special_tokens=True,
        )

    def generate_single(self, prompt: str, dec: Decoding) -> GenOut:
        outs = self.generate([prompt], dec)
        return outs[0]

    def generate(self, prompts: List[str], decoding: Decoding) -> List[GenOut]:
        sp = self._sp(decoding)
        t0 = time.time()
        req_outs = self._llm.generate(prompts, sp, use_tqdm=False)
        wall = time.time() - t0
        # vLLM preserves input order.
        out: List[GenOut] = []
        for ro in req_outs:
            if not ro.outputs:
                out.append(GenOut(text="", finish_reason="no_output", gen_tokens=0, seed=decoding.seed))
                continue
            o = ro.outputs[0]
            text = o.text or ""
            gen_tokens = len(getattr(o, "token_ids", []) or [])
            out.append(GenOut(
                text=text,
                finish_reason=getattr(o, "finish_reason", None),
                gen_tokens=gen_tokens,
                seed=decoding.seed,
            ))
        out = [
            GenOut(text=o.text, finish_reason=o.finish_reason, gen_tokens=o.gen_tokens, seed=o.seed)
            for o in out
        ]
        # Attach approximate per-request wall time (we only know batch total).
        self._last_wall = wall
        return out

    def generate_mixed(
        self,
        prompts: List[str],
        decodings: List[Decoding],
    ) -> List[GenOut]:
        """When each prompt has its own Decoding (e.g., per-sample seeds),
        we hand vLLM a parallel list of SamplingParams — vLLM supports that."""
        assert len(prompts) == len(decodings), f"{len(prompts)} vs {len(decodings)}"
        sps = [self._sp(d) for d in decodings]
        t0 = time.time()
        req_outs = self._llm.generate(prompts, sps, use_tqdm=False)
        self._last_wall = time.time() - t0
        out: List[GenOut] = []
        for ro, d in zip(req_outs, decodings):
            if not ro.outputs:
                out.append(GenOut(text="", finish_reason="no_output", gen_tokens=0, seed=d.seed))
                continue
            o = ro.outputs[0]
            out.append(GenOut(
                text=o.text or "",
                finish_reason=getattr(o, "finish_reason", None),
                gen_tokens=len(getattr(o, "token_ids", []) or []),
                seed=d.seed,
            ))
        return out

    def shutdown(self) -> None:
        try:
            del self._llm
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
