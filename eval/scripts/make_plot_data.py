#!/usr/bin/env python
"""
scripts/make_plot_data.py

Reads the per-(benchmark, mode) aggregate jsons under metrics/{run_id}/aggregate/
and writes to metrics/{run_id}/plot_data/:
  - {benchmark}_{mode}_by_model.csv (already produced by aggregate_metrics, but
    we mirror the key subset here for plotting convenience)
  - pattern_indexes.csv (z-scored Math-Style / IF-Style indexes per model)

Pattern indexes are intended FOR VISUALIZATION ONLY. Main claims should rely
on the per-metric profile in rq1_main_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.aggregate import compute_pattern_indexes  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    log = get_logger("make_plot_data", run_id=args.run_id)
    agg_dir = ROOT / "metrics" / args.run_id / "aggregate"
    plot_dir = ROOT / "metrics" / args.run_id / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load all rollups
    rollups = [json.loads(p.read_text()) for p in agg_dir.glob("*.json")]

    # Build a flat dict: {model_id: {metric_with_bench_prefix: value}}
    by_model: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rollups:
        model = r["model_id"]
        bench = r["benchmark"]
        mode = r["mode"]
        # Per-benchmark scalar map used by the pattern index.
        math_bench_is_primary = bench in {"math500", "aime24"}
        # Populate MODE-AGNOSTIC keys used by the pattern index, preferring
        # the main mode per benchmark (greedy for IF, sampling for math).
        preferred_math = mode.startswith("sampling") and math_bench_is_primary
        preferred_if = mode == "greedy" and bench in {"ifeval", "ifbench"}
        for k, v in r.items():
            if isinstance(v, (int, float)):
                by_model[model][f"{bench}__{mode}__{k}"] = v

        # canonical pattern-index inputs
        if preferred_math:
            for k in ["response_length_tokens_mean", "step_density_mean",
                      "equation_density_mean", "verification_marker_rate_mean",
                      "answer_position_ratio_mean", "distinct_2",
                      "best_minus_pass"]:
                if k in r and isinstance(r[k], (int, float)):
                    # Aggregate by averaging across primary math benchmarks
                    prev = by_model[model].get(f"__idx__{k}")
                    if prev is None:
                        by_model[model][f"__idx__{k}"] = r[k]
                    else:
                        by_model[model][f"__idx__{k}"] = (prev + r[k]) / 2
        if preferred_if:
            for src, dst in [
                ("strict_prompt_pass", "strict_prompt_pass_mean"),
                ("format_pass_rate", "format_pass_rate_mean"),
                ("compliance_efficiency", "compliance_efficiency_mean"),
                ("response_length_tokens", "response_length_tokens_mean"),
                ("over_reasoning_marker_rate", "over_reasoning_marker_rate_mean"),
                ("prefix_concentration_top1", "prefix_concentration_scalar"),
            ]:
                if src in r and isinstance(r[src], (int, float)):
                    prev = by_model[model].get(f"__idx__{dst}")
                    if prev is None:
                        by_model[model][f"__idx__{dst}"] = r[src]
                    else:
                        by_model[model][f"__idx__{dst}"] = (prev + r[src]) / 2

    # Build pattern-index input: strip the __idx__ prefix
    index_input: Dict[str, Dict[str, float]] = {}
    for model, d in by_model.items():
        index_input[model] = {k.replace("__idx__", ""): v for k, v in d.items() if k.startswith("__idx__")}
    # Also include distinct_2 alias as distinct_2_mean for Math-Style Index
    for m, d in index_input.items():
        if "distinct_2" in d and "distinct_2_mean" not in d:
            d["distinct_2_mean"] = d["distinct_2"]
        if "best_minus_pass" in d and "best_minus_pass_mean" not in d:
            d["best_minus_pass_mean"] = d["best_minus_pass"]

    idx = compute_pattern_indexes(index_input)

    # Write
    idx_path = plot_dir / "pattern_indexes.csv"
    cols = ["model_id", "math_style_index", "if_style_index"]
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for model in sorted(idx.keys()):
            row = [model, idx[model].get("math_style_index", ""), idx[model].get("if_style_index", "")]
            w.writerow(row)
    log.info(f"Wrote {idx_path}")
    # Also dump the raw index inputs (useful for plots)
    raw_path = plot_dir / "pattern_indexes_inputs.json"
    raw_path.write_text(json.dumps(index_input, indent=2))
    log.info(f"Wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
