#!/usr/bin/env python
"""
scripts/run_inference.py

Run inference for one (model_id, benchmark, mode) and write responses to:
  responses/{run_id}/{model_id}/{benchmark}/{mode}.jsonl

Supports:
- Resume: skips (example_id, sample_id) pairs already present in the jsonl.
- Flush every N samples to survive interruption.
- vLLM (default) or --engine hf fallback.
- Per-sample seed = 42 + sample_id.

Usage:
  python scripts/run_inference.py \
      --model-id math_rlvr \
      --benchmark math500 \
      --mode sampling_k16 \
      --run-id 20260503_rq1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_processed  # noqa: E402
from src.inference.engine import Decoding  # noqa: E402
from src.inference.prompts import build_inference_prompt  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.io import append_jsonl, atomic_write_json, iter_jsonl  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402
from src.utils.tokenization import get_tokenizer  # noqa: E402


def _find_plan_entry(plan_cfg: Dict[str, Any], model: str, benchmark: str, mode: str) -> Dict[str, Any]:
    for section in ("required", "optional"):
        for e in plan_cfg.get(section) or []:
            if e["model"] == model and e["benchmark"] == benchmark and e["mode"] == mode:
                return e
    raise KeyError(f"({model}, {benchmark}, {mode}) not found in eval_plan.yaml")


def _decoding(plan_entry: Dict[str, Any], global_cfg: Dict[str, Any], sample_id: int) -> Decoding:
    base_seed = int(global_cfg.get("seed", 42))
    return Decoding(
        temperature=float(plan_entry["temperature"]),
        top_p=float(plan_entry["top_p"]),
        max_new_tokens=int(plan_entry.get("max_new_tokens", global_cfg.get("max_new_tokens", 4096))),
        seed=base_seed + int(sample_id),
        repetition_penalty=float(plan_entry.get("repetition_penalty", 1.0)),
        stop=list(plan_entry.get("stop", [])),
    )


def _build_engine(engine_name: str, model_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]):
    if engine_name == "vllm":
        from src.inference.vllm_engine import VLLMEngine
        return VLLMEngine(model_cfg, engine_cfg)
    if engine_name == "hf":
        from src.inference.hf_engine import HFEngine
        return HFEngine(model_cfg, engine_cfg)
    raise ValueError(f"Unknown engine: {engine_name}")


def _already_done(out_path: Path) -> set:
    done = set()
    if not out_path.exists():
        return done
    for r in iter_jsonl(out_path):
        done.add((r.get("example_id"), int(r.get("sample_id", 0))))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models-yaml", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--datasets-yaml", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--plan-yaml", default=str(ROOT / "configs/eval_plan.yaml"))
    ap.add_argument("--prompts-yaml", default=str(ROOT / "configs/prompts.yaml"))
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None, help="Process at most N examples (for dry runs).")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if (example,sample) already present.")
    ap.add_argument("--chunk-size", type=int, default=None, help="Override generation chunk size; useful for long-output probes.")
    args = ap.parse_args()

    log = get_logger("run_inference", run_id=args.run_id)

    m_cfg = load_yaml(args.models_yaml)
    d_cfg = load_yaml(args.datasets_yaml)
    plan_cfg = load_yaml(args.plan_yaml)

    global_cfg = plan_cfg.get("global", {})
    set_global_seed(int(global_cfg.get("seed", 42)))

    model_cfg = (m_cfg.get("models") or {}).get(args.model_id)
    if model_cfg is None:
        raise SystemExit(f"Unknown model_id: {args.model_id}")
    plan_entry = _find_plan_entry(plan_cfg, args.model_id, args.benchmark, args.mode)
    K = int(plan_entry["k"])

    processed_dir = d_cfg.get("processed_dir") or str(ROOT / "cache/processed_data")
    # Special case: gsm8k_subset reads the subset jsonl produced by prepare_data
    bench_key = args.benchmark
    records = load_processed(bench_key, processed_dir)
    if args.limit is not None:
        records = records[: args.limit]

    # Output layout
    out_dir = ROOT / "responses" / args.run_id / args.model_id / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.mode}.jsonl"
    manifest_path = out_dir / f"{args.mode}.manifest.json"

    done = set() if args.overwrite else _already_done(out_path)
    log.info(f"Model={args.model_id} benchmark={args.benchmark} mode={args.mode} K={K} "
             f"examples={len(records)} already_done={len(done)}")

    # Build list of (example_record, sample_id, decoding)
    jobs: List[Tuple[Dict[str, Any], int, Decoding]] = []
    for rec in records:
        for sid in range(K):
            key = (rec["id"], sid)
            if key in done and not args.overwrite:
                continue
            jobs.append((rec, sid, _decoding(plan_entry, global_cfg, sid)))
    log.info(f"Jobs to run: {len(jobs)}")
    if not jobs:
        log.info("Nothing to do (resume hit all)."); return 0

    # Build engine
    eng_cfg = dict(global_cfg.get("vllm") or {})
    eng_cfg["seed"] = int(global_cfg.get("seed", 42))
    engine = _build_engine(args.engine, model_cfg, eng_cfg)

    # Load tokenizer for apply_chat_template (needed when prompt_style == 'chat').
    # Cached in utils.tokenization.get_tokenizer so cheap to call.
    prompt_style = model_cfg.get("prompt_style", "chat")
    tokenizer = None
    if prompt_style == "chat":
        tok_path = model_cfg.get("tokenizer_path") or model_cfg["path"]
        tokenizer = get_tokenizer(
            tok_path,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        if getattr(tokenizer, "chat_template", None) is None:
            raise SystemExit(
                f"Model {args.model_id} requests prompt_style=chat but tokenizer at {tok_path} "
                f"has no chat_template. Either ship one or set prompt_style: plain."
            )

    # Build prompts (apply chat template if model_cfg says so)
    prompts = [build_inference_prompt(rec, model_cfg, tokenizer=tokenizer) for rec, _, _ in jobs]
    decodings = [d for _, _, d in jobs]

    # Sanity log: show the first prompt exactly as it will be fed to vLLM, so
    # any chat-template mismatch is obvious in the run log.
    if prompts:
        head = prompts[0]
        preview_n = 400
        log.info(
            f"prompt_style={prompt_style}  "
            f"first prompt (len={len(head)} chars): {head[:preview_n]!r}"
            + ("..." if len(head) > preview_n else "")
        )

    flush_every = int(global_cfg.get("flush_every", 16))
    t0 = time.time()
    batched_records: List[Dict[str, Any]] = []
    n_done = 0
    # Run in chunks so we can flush to disk periodically.
    chunk = int(args.chunk_size) if args.chunk_size else max(32, flush_every)
    for i in range(0, len(jobs), chunk):
        sub_prompts = prompts[i:i + chunk]
        sub_decs = decodings[i:i + chunk]
        sub_jobs = jobs[i:i + chunk]

        # For greedy mode, all seeds are 42 but vLLM still respects seed arg.
        outs = engine.generate_mixed(sub_prompts, sub_decs)

        batch: List[Dict[str, Any]] = []
        for (rec, sid, dec), go, p in zip(sub_jobs, outs, sub_prompts):
            r = {
                "run_id": args.run_id,
                "model_id": args.model_id,
                "model_path": model_cfg["path"],
                "checkpoint": model_cfg["path"],
                "benchmark": args.benchmark,
                "example_id": rec["id"],
                "sample_id": sid,
                "prompt_template": rec.get("metadata", {}).get("prompt_template_id"),
                "prompt": p,
                "gold": rec.get("gold"),
                "metadata": rec.get("metadata", {}),
                "decoding": {
                    "temperature": dec.temperature,
                    "top_p": dec.top_p,
                    "max_new_tokens": dec.max_new_tokens,
                    "seed": dec.seed,
                    "repetition_penalty": dec.repetition_penalty,
                },
                "response": go.text,
                "finish_reason": go.finish_reason,
                "gen_tokens": go.gen_tokens,
                "wall_time": getattr(engine, "_last_wall", 0.0) / max(1, len(outs)),
            }
            batch.append(r)
        append_jsonl(out_path, batch)
        n_done += len(batch)
        if (i // chunk) % 1 == 0:
            log.info(f"flushed {n_done}/{len(jobs)}  elapsed={time.time()-t0:.1f}s")

    atomic_write_json(manifest_path, {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "mode": args.mode,
        "k": K,
        "num_examples": len(records),
        "num_jobs": len(jobs),
        "elapsed_sec": time.time() - t0,
        "engine": args.engine,
        "engine_cfg": eng_cfg,
        "plan_entry": plan_entry,
    })

    engine.shutdown()
    log.info(f"Done. Wrote {n_done} responses in {time.time()-t0:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
