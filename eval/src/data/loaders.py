"""
Loader for unified processed jsonl.

Unified schema:
{
  "id": "math500_0001",
  "benchmark": "math500",
  "task_type": "math" | "if",
  "prompt": "...",               # already-prompted text (template applied)
  "raw_problem": "...",          # original problem text (math only)
  "gold": "...",                 # ground truth answer (math only; null for IF)
  "metadata": {...}              # source-specific fields
}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import read_jsonl


def load_processed(benchmark: str, processed_dir: str) -> List[Dict[str, Any]]:
    path = Path(processed_dir) / f"{benchmark}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}. Run scripts/prepare_data.py first."
        )
    return read_jsonl(path)
