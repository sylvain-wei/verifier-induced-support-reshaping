"""
IO helpers for JSONL append/read and resume indexing.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

PathLike = Union[str, os.PathLike]


def ensure_parent(path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def atomic_write_json(path: PathLike, obj: Any) -> None:
    ensure_parent(path)
    d = os.path.dirname(str(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def resume_keys(path: PathLike, key_fn) -> Set[Tuple]:
    """Scan an existing jsonl and collect composite keys for resume."""
    out: Set[Tuple] = set()
    for r in iter_jsonl(path):
        try:
            out.add(tuple(key_fn(r)))
        except Exception:
            continue
    return out
