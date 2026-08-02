#!/usr/bin/env python
"""
scripts/prepare_data.py

Read configs/datasets.yaml, ensure each benchmark's raw data is available
(either local or downloaded from HF), convert to the unified jsonl schema
under cache/processed_data/{benchmark}.jsonl, and emit a summary.

Usage:
  python scripts/prepare_data.py [--only math500 aime24 ifeval ifbench]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent  # eval/
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(ROOT.parent))).resolve()
DATA_ROOT = PROJECT_ROOT / "data"
sys.path.insert(0, str(ROOT))

from src.data import converters as C  # noqa: E402
from src.data.downloaders import download_gsm8k_test, download_ifbench_test, download_hf_dataset_to_jsonl  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.io import write_jsonl, read_jsonl  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _load_parquet(path: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return t.to_pylist()


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def _load_records(local_sources: List[str]) -> List[Dict[str, Any]]:
    for p in local_sources or []:
        q = Path(p)
        if not q.exists():
            continue
        if q.suffix == ".parquet":
            return _load_parquet(str(q))
        if q.suffix == ".jsonl":
            return _load_jsonl(str(q))
        if q.suffix == ".json":
            text = q.read_text(encoding="utf-8")
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return _load_jsonl(str(q))
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ("data", "examples", "records", "dataset"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
                return [obj]
    return []


def _gsm8k_subset(records: List[Dict[str, Any]], bench_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    import random as _r
    rng = _r.Random(int(bench_cfg.get("subset_seed", 42)))
    order = list(range(len(records)))
    rng.shuffle(order)
    subset_ids = sorted(order[: int(bench_cfg.get("subset_size", 300))])
    return subset_ids


def prepare_benchmark(name: str, bench_cfg: Dict[str, Any], prompts_cfg: Dict[str, Any], processed_dir: Path, log) -> Dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{name}.jsonl"
    converter = bench_cfg["converter"]

    # Alias handling: *_cot benchmarks declare `alias_of: <base>` in datasets.yaml.
    # They share the same raw data + converter as <base>, so we use <base>'s
    # download / subset branches, then re-label the unified records to `name`.
    base_name = bench_cfg.get("alias_of") or name

    records: List[Dict[str, Any]] = []

    if base_name == "gsm8k":
        records = _load_records(bench_cfg.get("local_sources") or [])
        if not records:
            log.info(f"[{name}] local not found; downloading openai/gsm8k main test ...")
            local_target = DATA_ROOT / "gsm8k" / "test.jsonl"
            records = download_gsm8k_test(str(local_target))
            log.info(f"[{name}] downloaded {len(records)} rows to {local_target}")
    elif base_name == "ifbench":
        records = _load_records(bench_cfg.get("local_sources") or [])
        if not records:
            log.info(f"[{name}] local not found; downloading allenai/IFBench ...")
            local_target = DATA_ROOT / "ifbench" / "test.jsonl"
            records = download_ifbench_test(str(local_target))
            log.info(f"[{name}] downloaded {len(records)} rows to {local_target}")
    else:
        records = _load_records(bench_cfg.get("local_sources") or [])
        if not records:
            repo = bench_cfg.get("hf_repo")
            if not repo:
                raise RuntimeError(f"No local data for {name} and no downloader wired; add one in src/data/downloaders.py")
            local_sources = bench_cfg.get("local_sources") or []
            local_target = Path(local_sources[0]) if local_sources else DATA_ROOT / name / f"{name}.jsonl"
            log.info(f"[{name}] local not found; downloading {repo} ...")
            records = download_hf_dataset_to_jsonl(
                repo,
                str(local_target),
                config=bench_cfg.get("hf_config"),
                split=bench_cfg.get("hf_split", "train"),
            )
            log.info(f"[{name}] downloaded {len(records)} rows to {local_target}")

    unified = getattr(C, converter)(records, bench_cfg, prompts_cfg)

    # If this is an alias (e.g. aime24_cot, math500_cot, gsm8k_cot), the
    # converter may have stamped records with the base benchmark name
    # (e.g. aime_from_parquet infers "aime24" from `origin`). Re-stamp so
    # downstream tools group these results under the cot key instead.
    if name != base_name:
        for j, rec in enumerate(unified):
            rec["benchmark"] = name
            rec["id"] = f"{name}_{j:04d}"
            md = rec.setdefault("metadata", {})
            md["alias_of"] = base_name
            md["prompt_template_id"] = bench_cfg["prompt_template_id"]

    # Write full unified data
    write_jsonl(out_path, unified)

    extras: Dict[str, Any] = {
        "benchmark": name,
        "converter": converter,
        "raw_count": len(records),
        "unified_count": len(unified),
        "output": str(out_path),
    }
    if name != base_name:
        extras["alias_of"] = base_name

    # GSM8K subset (deterministic 300 ids) — apply to both gsm8k and gsm8k_cot.
    # Use the same seed so the 300-example subset is IDENTICAL across the
    # cot / non-cot runs (critical for a clean ablation).
    if base_name == "gsm8k" and records:
        subset_ids = _gsm8k_subset(records, bench_cfg)
        # Write subset ids file under configs/ only for the canonical gsm8k
        # (so we don't overwrite it when preparing gsm8k_cot).
        if name == "gsm8k":
            subset_cfg = ROOT / "configs" / "gsm8k_subset_ids.json"
            subset_cfg.write_text(json.dumps({"ids": subset_ids, "seed": int(bench_cfg.get("subset_seed", 42))}, indent=2))
        id_set = set(subset_ids)
        subset_records = [unified[i] for i in subset_ids if i < len(unified)]
        subset_out = processed_dir / f"{name}_subset.jsonl"
        write_jsonl(subset_out, subset_records)
        extras[f"{name}_subset_output"] = str(subset_out)
        extras[f"{name}_subset_count"] = len(subset_records)

    return extras


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--prompts", default=str(ROOT / "configs/prompts.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    log = get_logger("prepare_data", run_id=None)
    d_cfg = load_yaml(args.config)
    p_cfg = load_yaml(args.prompts)

    processed_dir = Path(d_cfg.get("processed_dir", str(ROOT / "cache/processed_data")))
    benches = d_cfg.get("benchmarks") or {}
    targets = list(benches.keys()) if args.only is None else args.only
    summary: List[Dict[str, Any]] = []

    for name in targets:
        if name not in benches:
            log.error(f"Benchmark {name} not in datasets.yaml; skipping")
            continue
        try:
            extras = prepare_benchmark(name, benches[name], p_cfg, processed_dir, log)
            log.info(f"[OK] {name}: {extras}")
            summary.append(extras)
        except Exception as e:
            log.error(f"[FAIL] {name}: {e}")
            summary.append({"benchmark": name, "error": str(e)})

    summary_path = processed_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
