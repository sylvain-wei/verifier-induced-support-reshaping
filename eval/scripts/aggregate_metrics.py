#!/usr/bin/env python
"""
scripts/aggregate_metrics.py

Collects metrics/{run_id}/aggregate/*.json produced by compute_metrics.py and
produces:
  - metrics/{run_id}/aggregate/{benchmark}_{mode}_summary.csv  (one row per model)
  - metrics/{run_id}/aggregate/rq1_main_summary.csv            (wide: one row per model, key columns across benchmarks)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging import get_logger  # noqa: E402


# Columns we surface in the main summary.
MAIN_COLUMNS_MATH = [
    "pass@1", "best@k", "maj@k", "best_minus_pass",
    "response_length_tokens_mean", "step_density_mean",
    "equation_density_mean", "verification_marker_rate_mean",
    "answer_position_ratio_mean", "distinct_2",
    "normalized_answer_entropy", "extraction_failure_rate",
]
MAIN_COLUMNS_IF = [
    "strict_prompt_pass", "instruction_level_pass_rate",
    "format_pass_rate", "compliance_efficiency",
    "response_length_tokens", "over_reasoning_marker_rate",
    "prefix_concentration_top1", "first_sentence_entropy",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    log = get_logger("aggregate_metrics", run_id=args.run_id)

    agg_dir = ROOT / "metrics" / args.run_id / "aggregate"
    if not agg_dir.exists():
        raise SystemExit(f"No aggregate dir: {agg_dir}")

    rollups = []
    for p in sorted(agg_dir.glob("*.json")):
        try:
            rollups.append(json.loads(p.read_text()))
        except Exception as e:
            log.warning(f"Skipping {p}: {e}")

    # Group by (benchmark, mode)
    by_bm: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rollups:
        by_bm[(r["benchmark"], r["mode"])] .append(r)

    # Per-(benchmark, mode) CSV
    for (bench, mode), rows in by_bm.items():
        out = agg_dir / f"{bench}_{mode}_summary.csv"
        # union of keys
        cols = sorted({k for r in rows for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in sorted(rows, key=lambda x: x["model_id"]):
                w.writerow([r.get(k, "") for k in cols])
        log.info(f"Wrote {out}")

    # Main RQ1 summary: one row per (model_id), columns = (benchmark__mode__metric)
    main: Dict[str, Dict[str, Any]] = defaultdict(dict)
    all_cols = []
    for r in rollups:
        model = r["model_id"]
        bench = r["benchmark"]
        mode = r["mode"]
        task = r.get("task_type", "math")
        keys = MAIN_COLUMNS_MATH if task == "math" else MAIN_COLUMNS_IF
        for k in keys:
            col = f"{bench}__{mode}__{k}"
            main[model][col] = r.get(k, "")
            all_cols.append(col)
    all_cols = sorted(set(all_cols))
    main_path = agg_dir / "rq1_main_summary.csv"
    with main_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_id"] + all_cols)
        for model in sorted(main.keys()):
            w.writerow([model] + [main[model].get(c, "") for c in all_cols])
    log.info(f"Wrote {main_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
