#!/usr/bin/env python3
"""Rebuild the release SHA-256 manifest after all content changes."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.tsv"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp"}


def released_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def main() -> int:
    rows = []
    for path in released_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{path.relative_to(ROOT).as_posix()}\tCLEANED\t{digest}")
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
