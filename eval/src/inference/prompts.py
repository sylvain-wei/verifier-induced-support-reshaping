"""
Prompt helper.

`build_inference_prompt` returns the final text blob that we hand to vLLM.

For RQ1 / RQ2 every model uses `prompt_style: chat`, mirroring verl training:
verl's RLHFDataset and SingleTurnAgentLoop both call
  tokenizer.apply_chat_template(messages, add_generation_prompt=True)
on the `[{"role":"user","content":"..."}]` messages stored in the training
parquet. Evaluating in `plain` completion mode would feed the RLVR actor an
out-of-distribution prompt format (no `<|im_start|>assistant\\n` trigger), so
we must mirror the training wrap here.

`plain` mode is kept as an ablation knob — set `prompt_style: plain` in
configs/models.yaml to skip apply_chat_template for a given model. It is NOT
used by the default RQ1 / RQ2 matrices.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def build_inference_prompt(
    record: Dict[str, Any],
    model_cfg: Dict[str, Any],
    tokenizer=None,
) -> str:
    """Return the text prompt the engine should feed to the model.

    Args:
        record: unified record with a pre-templated string in `record["prompt"]`
            (produced by converters.py applying the task prompt template).
        model_cfg: entry from configs/models.yaml. Must contain `prompt_style`
            and, for chat mode, `tokenizer_path` (or fall back to `path`).
        tokenizer: optional preloaded tokenizer. If None and chat mode is
            requested, we lazily load one via utils.tokenization.get_tokenizer
            (cached, so a repeated call for the same path is free).
    """
    raw = record["prompt"]
    style = model_cfg.get("prompt_style", "chat")

    if style == "plain":
        # Ablation path: feed the raw string with no chat_template wrap.
        # Not used by the RQ1 main matrix.
        return raw

    if style == "chat":
        if tokenizer is None:
            # Lazy-load from the model's tokenizer_path. Cached, so subsequent
            # calls for the same path are O(1).
            from ..utils.tokenization import get_tokenizer
            tok_path = model_cfg.get("tokenizer_path") or model_cfg["path"]
            tokenizer = get_tokenizer(
                tok_path,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            )
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                f"chat prompt_style requested but tokenizer has no chat_template "
                f"(model_cfg={model_cfg.get('path')}). Either ship a chat_template or "
                f"set prompt_style: plain."
            )
        messages = [{"role": "user", "content": raw}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(f"Unknown prompt_style: {style!r}")
