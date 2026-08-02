#!/usr/bin/env python
"""
scripts/check_env.py

Reports:
  - Python / torch / CUDA / GPU
  - Relevant packages (transformers/vllm/datasets/numpy/pandas/scipy/sacrebleu/langdetect/sympy/immutabledict/absl-py)
  - Existence of model paths declared in configs/models.yaml
  - Existence of dataset source files declared in configs/datasets.yaml

The default mode is diagnostic and exits 0. Pass --strict to exit nonzero when
required packages or model paths are missing. Pass --require-gpu together with
--strict when GPU availability is part of the readiness gate.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # -> eval/
sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml  # noqa: E402


def _check_pkg(name: str):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-yaml", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--datasets-yaml", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()
    problems = []

    print("=== Python ===")
    print("python:", sys.version.split()[0])
    print("executable:", sys.executable)
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"))
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        print("  WARNING: VLLM_WORKER_MULTIPROC_METHOD is not 'spawn'. vLLM may crash")
        print("  with 'Cannot re-initialize CUDA in forked subprocess'. The bundled")
        print("  VLLMEngine sets it at runtime; but setting it in the shell is safer.")
    print("\n=== Packages ===")
    for p in ["torch", "transformers", "vllm", "datasets", "numpy", "pandas",
              "scipy", "sacrebleu", "langdetect", "sympy", "immutabledict",
              "absl", "regex", "pylatexenc", "tqdm", "yaml"]:
        ok, v = _check_pkg(p)
        print(f"  [{'OK' if ok else 'MIS'}] {p:15s} {v}")
        if not ok:
            problems.append(f"missing package: {p}")

    # GPU
    print("\n=== GPU ===")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print("torch.cuda.is_available:", cuda_available)
        print("device_count:", torch.cuda.device_count())
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                free, total = torch.cuda.mem_get_info(i)
                print(f"  [{i}] {name}  free={free/1e9:.1f}G total={total/1e9:.1f}G")
        elif args.require_gpu:
            problems.append("CUDA GPU is unavailable")
    except Exception as e:
        print("torch cuda query failed:", e)
        if args.require_gpu:
            problems.append(f"GPU query failed: {e}")

    # Model paths
    print("\n=== Models (from models.yaml) ===")
    m_cfg = load_yaml(args.models_yaml)
    for model_id, entry in (m_cfg.get("models") or {}).items():
        p = Path(entry["path"])
        ok = p.exists() and (p / "config.json").exists()
        size_gb = "?"
        if p.exists():
            try:
                s = sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
                size_gb = f"{s/1e9:.1f}G"
            except Exception:
                size_gb = "?"
        print(f"  [{'OK' if ok else 'MISSING'}] {model_id:12s} {entry['path']}  size={size_gb}")
        if not ok:
            problems.append(f"missing model path: {model_id}")

    # Dataset sources
    print("\n=== Datasets (from datasets.yaml) ===")
    d_cfg = load_yaml(args.datasets_yaml)
    for bench, entry in (d_cfg.get("benchmarks") or {}).items():
        srcs = entry.get("local_sources") or []
        found = [s for s in srcs if Path(s).exists()]
        print(f"  [{'OK' if found else 'MISSING-local'}] {bench:10s}  local_hits={len(found)}/{len(srcs)}  hf={entry.get('hf_repo')}")
        if not found and not entry.get("hf_repo"):
            problems.append(f"no local or public source for dataset: {bench}")

    print("\nEnv check complete.")
    if problems:
        print(f"Readiness problems: {len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("No readiness problems detected.")
    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
