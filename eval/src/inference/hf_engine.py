"""
HuggingFace fallback engine. Used only when vLLM cannot instantiate (e.g.,
in a non-GPU unit-test context). It is NOT meant to match vLLM throughput.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine import BaseEngine, Decoding, GenOut


class HFEngine(BaseEngine):
    def __init__(self, model_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]):
        super().__init__(model_cfg, engine_cfg)
        dtype = torch.bfloat16 if engine_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        device_map = "auto"
        self.tok = AutoTokenizer.from_pretrained(
            model_cfg.get("tokenizer_path") or model_cfg["path"], use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["path"], torch_dtype=dtype, device_map=device_map
        )
        self.model.eval()

    def generate(self, prompts: List[str], decoding: Decoding) -> List[GenOut]:
        outs: List[GenOut] = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt").input_ids.to(self.model.device)
            gen_kwargs = dict(
                do_sample=decoding.temperature > 0,
                temperature=max(1e-5, decoding.temperature),
                top_p=decoding.top_p,
                max_new_tokens=decoding.max_new_tokens,
                repetition_penalty=decoding.repetition_penalty,
                pad_token_id=self.tok.eos_token_id,
            )
            torch.manual_seed(int(decoding.seed))
            t0 = time.time()
            out_ids = self.model.generate(ids, **gen_kwargs)
            wall = time.time() - t0
            new_ids = out_ids[0, ids.shape[-1]:]
            text = self.tok.decode(new_ids, skip_special_tokens=True)
            outs.append(GenOut(
                text=text,
                finish_reason=None,
                gen_tokens=int(new_ids.numel()),
                seed=int(decoding.seed),
            ))
        return outs

    def shutdown(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
