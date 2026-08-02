"""
HF dataset downloaders used by prepare_data.py when local files are missing.

We never silent-fail: every downloader raises RuntimeError on failure and the
caller decides whether to abort or mark a benchmark unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hf_download(repo: str, config: Optional[str], split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"`datasets` not available: {e}")
    try:
        if config:
            ds = load_dataset(repo, config, split=split)
        else:
            ds = load_dataset(repo, split=split)
        return [dict(x) for x in ds]
    except Exception as e:
        raise RuntimeError(f"HF load_dataset({repo!r}, config={config!r}, split={split!r}) failed: {e}")


def download_gsm8k_test(target_jsonl: str) -> List[Dict[str, Any]]:
    """Download openai/gsm8k main/test -> jsonl at target_jsonl."""
    rows = _hf_download("openai/gsm8k", "main", "test")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": i,
            "question": r["question"],
            "answer_full": r["answer"],
        })
    Path(target_jsonl).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(target_jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def download_hf_dataset_to_jsonl(repo: str, target_jsonl: str, config: Optional[str] = None, split: str = "train") -> List[Dict[str, Any]]:
    """Download an arbitrary HF dataset split and persist it as JSONL."""
    rows = _hf_download(repo, config, split)
    Path(target_jsonl).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(target_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def download_ifbench_test(target_jsonl: str) -> List[Dict[str, Any]]:
    """Download allenai/IFBench_test -> jsonl at target_jsonl."""
    # The canonical IFBench test set is at `allenai/IFBench_test` (single
    # parquet under data/train-*). Older plan notes called it "IFBench", but
    # that repo id does not exist; we try both forms for safety.
    candidates = [
        ("allenai/IFBench_test", None, "train"),
        ("allenai/IFBench_test", None, "test"),
    ]
    last_err: Optional[Exception] = None
    rows: Optional[List[Dict[str, Any]]] = None
    for repo, cfg, split in candidates:
        try:
            rows = _hf_download(repo, cfg, split)
            break
        except Exception as e:
            last_err = e
            continue
    if rows is None:
        raise RuntimeError(f"Could not download IFBench via `datasets`: {last_err}")

    Path(target_jsonl).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(target_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
