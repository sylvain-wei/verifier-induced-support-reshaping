"""
Tokenizer loading cache. We use the response-model's own tokenizer to count
`gen_tokens` and `response_length_tokens` so that length metrics are
comparable across models with identical vocab (Qwen3 family here).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def get_tokenizer(path: str, trust_remote_code: bool = False) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True)


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        # Fallback: whitespace token count. Not used in practice.
        return len(text.split())
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:
        return len(text.split())
