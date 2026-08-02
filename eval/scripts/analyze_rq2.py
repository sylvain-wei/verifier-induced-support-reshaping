#!/usr/bin/env python3
"""
analyze_rq2.py
--------------

Join the RQ1 summary (base / math_rlvr / if_rlvr) with the RQ2 sequential
summary (math_then_if / if_then_math) and produce the analysis that
answers the question:

    Does sequential RLVR (a) preserve both patterns (composition),
    (b) shift default style toward the second objective while
    preserving the first objective's task-level capability (rerouting),
    or (c) override the first objective entirely (catastrophic)?

For every headline metric we compute:
  * score / metric value for each of the 5 models
  * Retention(B→M→I on math)   = (B→M→I - Base) / (B→M - Base)    — how much of Math-RLVR's gain over base is preserved after the IF stage
  * Acquisition(B→M→I on IF)   = (B→M→I - Base) / (B→I  - Base)   — how much of IF-RLVR's gain over base is acquired
  * same for B→I→M (flip roles)
  * style-side: normalized distance to B→M and to B→I in pattern-metric space

Outputs:
  * CSV of the merged 5-model × metric table (columns = RQ1 schema)
  * Markdown report to stdout
  * Verdict (A/B/C) per axis with numerical justification
"""

import os
import argparse
import csv
import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---- column groups used in the narrative ----
MATH_SCORE_COLS = [
    ("math500__sampling_k16__pass@1", "math500_pass@1"),
    ("math500__sampling_k16__best@k", "math500_best@16"),
    ("math500__sampling_k16__best_minus_pass", "math500_best-pass"),
    ("aime24__sampling_k32__pass@1", "aime24_pass@1"),
    ("aime24__sampling_k32__best@k", "aime24_best@32"),
    ("aime24__sampling_k32__best_minus_pass", "aime24_best-pass"),
    ("gsm8k__greedy__pass@1", "gsm8k_pass@1"),
]

MATH_STYLE_COLS = [
    ("math500__sampling_k16__response_length_tokens_mean", "math500_len"),
    ("math500__sampling_k16__step_density_mean", "math500_step_dens"),
    ("math500__sampling_k16__equation_density_mean", "math500_eq_dens"),
    ("math500__sampling_k16__verification_marker_rate_mean", "math500_verif"),
    ("aime24__sampling_k32__response_length_tokens_mean", "aime24_len"),
    ("aime24__sampling_k32__step_density_mean", "aime24_step_dens"),
    ("aime24__sampling_k32__equation_density_mean", "aime24_eq_dens"),
    ("aime24__sampling_k32__verification_marker_rate_mean", "aime24_verif"),
    ("gsm8k__greedy__response_length_tokens_mean", "gsm8k_len"),
    ("gsm8k__greedy__step_density_mean", "gsm8k_step_dens"),
]

IF_SCORE_COLS = [
    ("ifeval__greedy__strict_prompt_pass", "ifeval_strict"),
    ("ifeval__greedy__instruction_level_pass_rate", "ifeval_instr_pass"),
    ("ifeval__greedy__format_pass_rate", "ifeval_format_pass"),
    ("ifbench__greedy__strict_prompt_pass", "ifbench_strict"),
    ("ifbench__greedy__instruction_level_pass_rate", "ifbench_instr_pass"),
    ("ifbench__greedy__format_pass_rate", "ifbench_format_pass"),
]

IF_STYLE_COLS = [
    ("ifeval__greedy__response_length_tokens", "ifeval_len"),
    ("ifeval__greedy__compliance_efficiency", "ifeval_compl_eff"),
    ("ifeval__greedy__over_reasoning_marker_rate", "ifeval_over_reason"),
    ("ifbench__greedy__response_length_tokens", "ifbench_len"),
    ("ifbench__greedy__compliance_efficiency", "ifbench_compl_eff"),
    ("ifbench__greedy__over_reasoning_marker_rate", "ifbench_over_reason"),
]


def load_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {model_id: {col: value}}."""
    out = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            m = row["model_id"]
            d = {}
            for k, v in row.items():
                if k == "model_id":
                    continue
                try:
                    d[k] = float(v)
                except Exception:
                    d[k] = None
            out[m] = d
    return out


def safe(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def pct(v):
    return f"{100*v:+.1f}pp" if isinstance(v, (int, float)) else "—"


def ratio(num, den):
    if num is None or den is None or abs(den) < 1e-9:
        return None
    return num / den


def retention_acquisition_tables(merged, out_lines):
    """Compute retention / acquisition for each metric.

    Retention_math(B→M→I) = (math_then_if - base) / (math_rlvr - base)
      -- 1.0 = perfect preservation of Math-RLVR's gain over base
      --  0  = none preserved, equal to base
      -- <0  = worse than base after the IF stage (catastrophic)

    Acquisition_if(B→M→I) = (math_then_if - base) / (if_rlvr - base)
      -- 1.0 = perfect acquisition of IF-RLVR's gain
      -- <0  = moved in the wrong direction

    Symmetric for B→I→M: Retention_if on IF benchmarks, Acquisition_math on math benchmarks.
    """

    def row(label, cols, direction, sequential, retained_stage_model, acquired_stage_model):
        # direction = +1 if higher is better (score) or signed-raw (length etc.)
        out = [label]
        for col, short in cols:
            base_v = merged.get("base", {}).get(col)
            seq_v = merged.get(sequential, {}).get(col)
            ret_v = merged.get(retained_stage_model, {}).get(col)
            acq_v = merged.get(acquired_stage_model, {}).get(col)

            delta_seq = None if (seq_v is None or base_v is None) else (seq_v - base_v)
            delta_ret = None if (ret_v is None or base_v is None) else (ret_v - base_v)
            delta_acq = None if (acq_v is None or base_v is None) else (acq_v - base_v)

            retention = ratio(delta_seq, delta_ret)
            acquisition = ratio(delta_seq, delta_acq)
            out.append(
                {
                    "col": short,
                    "base": base_v,
                    "seq": seq_v,
                    "retained_single": ret_v,
                    "acquired_single": acq_v,
                    "retention": retention,
                    "acquisition": acquisition,
                }
            )
        return out

    # B→M→I: first stage = Math (so retention over math_rlvr on math cols);
    #         second stage = IF   (so acquisition vs if_rlvr on IF cols)
    bmi_math = row("B→M→I math-axis (retention of Math)", MATH_SCORE_COLS,
                   +1, "math_then_if", "math_rlvr", "if_rlvr")
    bmi_if = row("B→M→I if-axis   (acquisition of IF)", IF_SCORE_COLS,
                 +1, "math_then_if", "math_rlvr", "if_rlvr")

    bim_if = row("B→I→M if-axis   (retention of IF)", IF_SCORE_COLS,
                 +1, "if_then_math", "if_rlvr", "math_rlvr")
    bim_math = row("B→I→M math-axis (acquisition of Math)", MATH_SCORE_COLS,
                   +1, "if_then_math", "if_rlvr", "math_rlvr")

    out_lines.append("\n## 2. Retention / Acquisition on task scores\n")
    out_lines.append(
        "Retention = (seq - base) / (first_stage_single - base); 1.0 = fully preserved, 0 = back to base.\n"
        "Acquisition = (seq - base) / (second_stage_single - base); 1.0 = as much gain as doing single-objective.\n"
    )

    def render(blob, mode):
        # mode in {"retention", "acquisition"}
        label = blob[0]
        out_lines.append(f"\n### {label}\n")
        out_lines.append("| metric | base | single-stage-1 | single-stage-2 | sequential | Δseq | Δstage1 | Δstage2 | retention | acquisition |")
        out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for e in blob[1:]:
            d_seq = (None if (e["seq"] is None or e["base"] is None) else (e["seq"] - e["base"]))
            d_s1 = (None if (e["retained_single"] is None or e["base"] is None) else (e["retained_single"] - e["base"]))
            d_s2 = (None if (e["acquired_single"] is None or e["base"] is None) else (e["acquired_single"] - e["base"]))
            out_lines.append(
                f"| {e['col']} | {safe(e['base'])} | {safe(e['retained_single'])} | {safe(e['acquired_single'])}"
                f" | {safe(e['seq'])} | {safe(d_seq)} | {safe(d_s1)} | {safe(d_s2)}"
                f" | {safe(e['retention'])} | {safe(e['acquisition'])} |"
            )

    render(bmi_math, "retention")
    render(bmi_if, "acquisition")
    render(bim_if, "retention")
    render(bim_math, "acquisition")


def style_distance_table(merged, out_lines):
    """For each sequential model, is its style profile closer to first-stage
    single model or second-stage single model?

    We compute two normalized distances:
      d(sequential, math_rlvr)  -- L1 over math-style cols, normalized per col
      d(sequential, if_rlvr)    -- L1 over if-style cols, normalized per col
    Using per-col normalizer = max(|math_rlvr_col - if_rlvr_col|, eps). That
    makes differences comparable across different scale metrics.
    """
    out_lines.append("\n## 3. Behavioral style proximity: which single-stage does the sequential model look like?\n")

    def distance(model_a, model_b, cols):
        total = 0.0
        n = 0
        for col, _ in cols:
            a = merged.get(model_a, {}).get(col)
            b = merged.get(model_b, {}).get(col)
            # normalizer: difference between the two single-obj models on this col
            ra = merged.get("math_rlvr", {}).get(col)
            ri = merged.get("if_rlvr", {}).get(col)
            if None in (a, b, ra, ri):
                continue
            denom = max(abs(ra - ri), 1e-9)
            total += abs(a - b) / denom
            n += 1
        return total / n if n else None

    def diag(label, seq_model, cols, name):
        d_math = distance(seq_model, "math_rlvr", cols)
        d_if = distance(seq_model, "if_rlvr", cols)
        d_base = distance(seq_model, "base", cols)
        out_lines.append(
            f"- **{label} ({name})**: "
            f"d({seq_model}, math_rlvr) = {safe(d_math)} | "
            f"d({seq_model}, if_rlvr) = {safe(d_if)} | "
            f"d({seq_model}, base) = {safe(d_base)}"
        )
        if d_math is not None and d_if is not None:
            closer = "math_rlvr" if d_math < d_if else "if_rlvr"
            ratio_ = d_math / d_if if d_if > 0 else float("inf")
            out_lines.append(f"  → closer to **{closer}** (math/if ratio = {ratio_:.2f}; <1 means closer to math_rlvr, >1 means closer to if_rlvr)")

    diag("B→M→I math-style distance", "math_then_if", MATH_STYLE_COLS, "should stay near math_rlvr if math-style is preserved")
    diag("B→M→I if-style distance", "math_then_if", IF_STYLE_COLS, "should move toward if_rlvr if IF-style is acquired")
    diag("B→I→M math-style distance", "if_then_math", MATH_STYLE_COLS, "should move toward math_rlvr if math-style is acquired")
    diag("B→I→M if-style distance", "if_then_math", IF_STYLE_COLS, "should stay near if_rlvr if IF-style is preserved")


def verdict(merged, out_lines):
    """Render a per-axis verdict.

    For each sequential model and each axis (first-stage retained, second-stage acquired):
      - if |retention| >= 0.7 AND |acquisition| >= 0.7  -> COMPOSITION (rare)
      - if |retention| < 0.3 AND |acquisition| >= 0.7   -> OVERRIDE / ATTRACTOR SWITCH
      - if 0.3 <= |retention| < 0.7 AND |acquisition| >= 0.7 -> PARTIAL OVERRIDE
      - if |retention| >= 0.7 AND |acquisition| < 0.3   -> SECOND-STAGE FAILED
      - else                                             -> BOTH PARTIAL

    We run this on a headline score per axis:
      math axis  -> aime24_pass@1      (hardest math, clearest signal)
      if axis    -> ifeval_strict      (IF-RLVR's win condition)
    """
    out_lines.append("\n## 4. Verdict per axis (headline scores only)\n")

    def one(seq, stage1_single, stage2_single, axis_col, axis_name):
        b = merged["base"][axis_col]
        s1 = merged[stage1_single][axis_col]
        s2 = merged[stage2_single][axis_col]
        sq = merged[seq][axis_col]
        ds1 = s1 - b
        ds2 = s2 - b
        dsq = sq - b
        ret = dsq / ds1 if abs(ds1) > 1e-9 else None
        acq = dsq / ds2 if abs(ds2) > 1e-9 else None

        def lbl(ret, acq):
            if ret is None or acq is None:
                return "UNDEFINED (stage delta was 0)"
            r_ok = ret >= 0.7
            a_ok = acq >= 0.7
            r_mid = 0.3 <= ret < 0.7
            r_low = ret < 0.3
            if r_ok and a_ok:
                return "COMPOSITION (rare, interesting)"
            if r_low and a_ok:
                return "OVERRIDE / ATTRACTOR SWITCH (second stage dominates)"
            if r_mid and a_ok:
                return "PARTIAL OVERRIDE (first stage retained in part)"
            if r_ok and (acq is None or acq < 0.3):
                return "SECOND STAGE FAILED (first stage still dominant, second didn't take)"
            return "MIXED / BOTH PARTIAL"

        out_lines.append(
            f"- **{seq} on {axis_name}** ({axis_col}):\n"
            f"  - base={b:.4f}, {stage1_single}={s1:.4f} (Δ={ds1:+.4f}), "
            f"{stage2_single}={s2:.4f} (Δ={ds2:+.4f}), **{seq}={sq:.4f} (Δ={dsq:+.4f})**\n"
            f"  - retention={safe(ret)}, acquisition={safe(acq)}\n"
            f"  - **verdict: {lbl(ret, acq)}**"
        )

    one("math_then_if", "math_rlvr", "if_rlvr",
        "aime24__sampling_k32__pass@1", "math axis (aime24 pass@1)")
    one("math_then_if", "math_rlvr", "if_rlvr",
        "math500__sampling_k16__pass@1", "math axis (math500 pass@1)")
    one("math_then_if", "math_rlvr", "if_rlvr",
        "ifeval__greedy__strict_prompt_pass", "if axis (ifeval strict)")

    one("if_then_math", "if_rlvr", "math_rlvr",
        "ifeval__greedy__strict_prompt_pass", "if axis (ifeval strict)")
    one("if_then_math", "if_rlvr", "math_rlvr",
        "aime24__sampling_k32__pass@1", "math axis (aime24 pass@1)")
    one("if_then_math", "if_rlvr", "math_rlvr",
        "math500__sampling_k16__pass@1", "math axis (math500 pass@1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1-csv", default=os.environ.get("PROJECT_ROOT", ".") + "/eval/metrics/20260504_rq1/aggregate/rq1_main_summary.csv")
    ap.add_argument("--rq2-csv", default=os.environ.get("PROJECT_ROOT", ".") + "/eval/metrics/20260509_rq2_seq/aggregate/rq1_main_summary.csv")
    ap.add_argument("--out-md", default=os.environ.get("PROJECT_ROOT", ".") + "/eval/reports_by_cc/RQ2_sequential_analysis_20260509.md")
    ap.add_argument("--out-csv", default=os.environ.get("PROJECT_ROOT", ".") + "/eval/metrics/20260509_rq2_seq/aggregate/merged_5model_summary.csv")
    args = ap.parse_args()

    rq1 = load_csv(Path(args.rq1_csv))
    rq2 = load_csv(Path(args.rq2_csv))
    merged = {**rq1, **rq2}

    expected = ["base", "math_rlvr", "if_rlvr", "math_then_if", "if_then_math"]
    missing = [m for m in expected if m not in merged]
    if missing:
        print(f"WARNING: missing models: {missing}", file=sys.stderr)

    # write merged csv
    all_cols = set()
    for d in merged.values():
        all_cols.update(d.keys())
    cols_sorted = sorted(all_cols)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id"] + cols_sorted)
        for m in expected:
            if m not in merged:
                continue
            row = [m] + [merged[m].get(c, "") for c in cols_sorted]
            w.writerow(row)
    print(f"[merged csv] {args.out_csv}")

    # headline table
    out_lines = [
        "# RQ2 Sequential RLVR Analysis (2026-05-09 run)\n",
        "Models: base, math_rlvr, if_rlvr, **math_then_if (B→M→I)**, **if_then_math (B→I→M)**.\n",
        "Comparing RQ1 run `20260504_rq1` with the sequential run `20260509_rq2_seq` using the same"
        " pipeline, prompts, decoding, and metric code (identical schema in `rq1_main_summary.csv`).\n",
    ]

    # --- Section 1: headline 5-model table ---
    out_lines.append("\n## 1. Headline 5-model table\n")
    headline = MATH_SCORE_COLS + IF_SCORE_COLS + MATH_STYLE_COLS + IF_STYLE_COLS
    out_lines.append("| metric | base | math_rlvr | if_rlvr | math_then_if | if_then_math |")
    out_lines.append("|---|---:|---:|---:|---:|---:|")
    for col, short in headline:
        row = [short]
        for m in expected:
            v = merged.get(m, {}).get(col)
            row.append(safe(v))
        out_lines.append("| " + " | ".join(row) + " |")

    retention_acquisition_tables(merged, out_lines)
    style_distance_table(merged, out_lines)
    verdict(merged, out_lines)

    md = "\n".join(out_lines)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f"[markdown report] {args.out_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
