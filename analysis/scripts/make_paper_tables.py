#!/usr/bin/env python3
"""Generate compact paper-ready tables from existing analysis CSVs.

The outputs are intentionally small and opinionated: they are not replacements
for the full analysis tables under analysis/tables/, but main-text tables that
pin each load-bearing claim to a few numbers.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis" / "tables"
OUT = ROOT / "analysis" / "paper" / "tables"
EXTERNAL_INPUTS = (
    "d2_pos1_ratio_bootstrap.csv",
    "c_summary.csv",
    "e3_val_metrics.csv",
    "e3_3class_share.csv",
    "mathifpaper_overall.csv",
    "reasonifpaper_overall.csv",
)


def read_csv(name: str) -> List[Dict[str, str]]:
    with (SRC / name).open(newline="") as f:
        return list(csv.DictReader(f))


def pct(x, digits=1):
    return f"{float(x) * 100:.{digits}f}"


def num(x, digits=3):
    return f"{float(x):.{digits}f}"


def latex_escape(s):
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_table(
    stem,
    caption,
    label,
    headers,
    rows,
    align=None,
):
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    if align is None:
        align = "l" + "r" * (len(headers) - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT / f"{stem}.tex").write_text("\n".join(lines))


def table_setup_paths() -> None:
    rows = [
        ["Base", "reference policy", "Qwen3-8B / Qwen2.5-Math", "none", "all sections"],
        ["Math-RLVR", "direct verifier contrast", "base", "exact-match math", "Sec. 4.1, 5, 7"],
        ["IF-RLVR", "direct verifier contrast", "base", "multi-constraint IF", "Sec. 4.2-4.3, 5, 7"],
        ["IF->math", "sequential trainability test", "IF endpoint", "exact-match math", "Sec. 4.4"],
        ["Math->IF", "sequential retention/adaptation test", "Math endpoint", "multi-constraint IF + reference KL scan", "Sec. 4.5"],
        ["Routing prior", "cold-start IF-RLVR", "base + SFT prior", "IF-RLVR", "Sec. 6.1"],
        ["OPD", "distillation", "base student + IF teacher", "top-K reverse KL", "Sec. 6.2-6.3"],
    ]
    write_table(
        "table_setup_paths",
        "Controlled training paths used to diagnose verifier-induced support reshaping.",
        "tab:setup-paths",
        ["Path", "Role", "Initialization", "Training signal", "Primary use"],
        rows,
        align="lllll",
    )


def table_core_support_shift() -> None:
    rows = [
        [
            "F1: Math->IF",
            "Q3 IFEval",
            "pass@1 +0.065; best@32 -0.098",
            "0/32 +67; 32/32 +73 prompts",
            "support redistribution",
        ],
        [
            "F2: IF->AIME",
            "Q3 AIME",
            "best@32 0.40 -> 0.00",
            "DRI 0.71 -> 0.00; DAI 0.04 -> 1.00",
            "routing collapse",
        ],
        [
            "F3: IF shortcuts",
            "IFEval/IFBench",
            "judge shortcut 73.5%",
            "short-pass 0.103 -> 0.491",
            "surface compliance",
        ],
        [
            "F4: IF->math",
            "math-7.5k probe",
            "mixed 0.336 -> 0.016",
            "all-wrong init 0.656; DAI 0.557 -> 1.000",
            "future-trainability failure",
        ],
    ]
    write_table(
        "table_core_support_shift",
        "Core support-shift signatures that motivate the paper's mechanism.",
        "tab:core-support-shift",
        ["Finding", "Setting", "Metric shift", "Support/mode shift", "Interpretation"],
        rows,
        align="lllll",
    )


def table_js_routing_localization() -> None:
    rows = []
    for row in read_csv("d2_pos1_ratio_bootstrap.csv"):
        if row["dataset"] != "aime":
            continue
        pair = f"{row['base_tag']} vs {row['rl_tag']}"
        rows.append(
            [
                pair,
                row["dataset"],
                num(row["pos1_mean"]),
                num(row["interior_mean"], 4),
                f"{float(row['ratio_mean']):.1f}x",
                f"[{float(row['ratio_lo']):.1f}, {float(row['ratio_hi']):.1f}]",
            ]
        )
    write_table(
        "table_js_routing_localization",
        "AIME position-1 JS is far larger than interior JS, especially for IF-RLVR.",
        "tab:js-routing-localization",
        ["Model pair", "Dataset", "JS@1", "Interior JS", "Ratio", "95% CI"],
        rows,
        align="llrrrr",
    )


def table_single_token_intervention() -> None:
    wanted = {
        "free": "B_q3 free",
        "single_token_C1": "B_q3 + IF token",
        "random_top2_B_q3": "B_q3 random top2",
        "single_token_C2": "I_q3 + base token",
        "free_q25m": "B_q25m free",
        "single_token_C2_q25m": "I_q25m + base token",
        "forced_DRI_I_q25m": "I_q25m + DRI prefix",
    }
    source = {r["condition"]: r for r in read_csv("c_summary.csv")}
    rows = []
    for condition, label in wanted.items():
        row = source[condition]
        rows.append(
            [
                label,
                row["primary_tag"],
                row["intervention_tag"] or "none",
                num(row["best@32"]),
                pct(row["DRI_rate"]),
                pct(row["DAI_rate"]),
                str(round(float(row["mean_n_resp_tokens"]))),
            ]
        )
    write_table(
        "table_single_token_intervention",
        "Single routing-token interventions reproduce or reverse AIME searchability collapse.",
        "tab:single-token-intervention",
        ["Condition", "Primary", "Token source", "best@32", "DRI %", "DAI %", "Tokens"],
        rows,
        align="lllrrrr",
    )


def table_opd_sweetspot() -> None:
    e3 = read_csv("e3_val_metrics.csv")
    judge = {r["pool"].replace("student_e3_", ""): r for r in read_csv("e3_3class_share.csv")}

    def value(run: str, dataset: str, metric: str) -> str:
        for row in e3:
            if row["run"] == run and row["global_step"] == "100" and row["dataset"] == dataset:
                return row[metric]
        raise KeyError((run, dataset, metric))

    rows = [
        ["full IF step100", "50", "0.0879", "--", "18.8", "rejected"],
    ]
    for run in ["step20", "step40", "step60", "step80"]:
        math = value(run, "math_dapo", "acc_mean@16")
        ifeval = value(run, "ifeval_test", "acc_mean@16")
        shortcut = pct(judge[run]["share_shortcut"])
        verdict = "sweet spot" if run == "step20" else "shortcut/math trade-off"
        rows.append([run, "100", num(math), num(ifeval), shortcut, verdict])

    write_table(
        "table_opd_sweetspot",
        "OPD preserves math support only for the early IF teacher.",
        "tab:opd-sweetspot",
        ["Teacher", "OPD steps", "MATH mean@16", "IFEval mean@16", "Shortcut %", "Verdict"],
        rows,
        align="llrrrl",
    )


def table_cold_start() -> None:
    rows = [
        ["DAI control", "-25.79", "already collapsed", "0.000 / 1.000", "0.000 / 1.000"],
        ["Random control", "-0.058", "step 35-50", "0.500 / 0.006", "0.133 / 1.000"],
        ["Hard DRI prior", "0.000", "step 50-60", "0.500 / 0.000", "0.133 / 1.000"],
        ["Soft DRI prior", "-0.173", "step 70-80", "0.467 / 0.033", "0.067 / 1.000"],
    ]
    write_table(
        "table_cold_start",
        "Routing-prior cold starts control collapse timing but not the final attractor.",
        "tab:cold-start",
        ["Prior", "post-SFT logp DRI", "Transition", "Best AIME/DAI", "Step-100 AIME/DAI"],
        rows,
        align="lrlrr",
    )


def table_external_validation() -> None:
    mathif = {r["model"]: r for r in read_csv("mathifpaper_overall.csv")}
    reasonif = {r["model"]: r for r in read_csv("reasonifpaper_overall.csv")}
    names = [("base", "Base"), ("math_rlvr", "Math-RLVR"), ("if_rlvr", "IF-RLVR")]
    rows = []
    for key, label in names:
        m = mathif[key]
        r = reasonif[key]
        rows.append(
            [
                label,
                num(m["SAcc"]),
                num(m["IFAcc"]),
                num(m["HAcc"]),
                num(r["Reasoning_IFS"]),
                num(r["Genuine_IFS"]),
                num(r["bypass_rate"]),
                num(r["joint"]),
            ]
        )
    write_table(
        "table_external_validation",
        "External MathIF and ReasonIF benchmarks reproduce the two-attractor support-reshaping pattern: Math-RLVR improves content accuracy with little IF gain, while IF-RLVR improves surface following and bypass without increasing joint correct-and-genuine following.",
        "tab:external-validation",
        [
            "Model",
            "MathIF SAcc",
            "MathIF IFAcc",
            "MathIF HAcc",
            "ReasonIF IFS",
            "Genuine IFS",
            "Bypass",
            "Joint",
        ],
        rows,
        align="lrrrrrrr",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate released paper tables after checking source inputs."
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Generate only tables whose values are embedded in this release.",
    )
    args = parser.parse_args()

    missing = [name for name in EXTERNAL_INPUTS if not (SRC / name).is_file()]
    if missing:
        print("Missing intermediate analysis inputs:", file=sys.stderr)
        for name in missing:
            print(f"  - {SRC / name}", file=sys.stderr)
        print(
            "These raw analysis summaries are not bundled in the public release.",
            file=sys.stderr,
        )
        if not args.available_only:
            print("Use --available-only to regenerate the self-contained tables.", file=sys.stderr)
            return 2

    table_setup_paths()
    table_core_support_shift()
    table_cold_start()
    if not missing:
        table_js_routing_localization()
        table_single_token_intervention()
        table_opd_sweetspot()
        table_external_validation()
    print(f"Wrote paper tables to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
