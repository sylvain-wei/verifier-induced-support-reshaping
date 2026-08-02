#!/usr/bin/env python
"""h5 — Final eval table for WAVE 5 main experiments.

For each (run, step) combination:
  1. Read val_rollout jsonl (AIME 30 prompts × 32 = 960 rows; IFEval 540 × 32).
  2. Compute AIME best@32, mean@32 (pass@1), maj@32.
  3. Compute IFEval mean@32 (pass@1), best@32.
  4. Classify each AIME response by opening mode (DRI / DAI / Other / CSI),
     compute per-step DRI/DAI/Other rate.
  5. Aggregate to analysis/tables/wave5_final_eval.csv.

CPU-only, ~10 minutes for 2 runs × 21 steps × ~18k rows each.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(ROOT / "analysis/scripts"))
from classify_opening_modes import classify_mode  # noqa


RUNS = [
    ("S2_hard", "b2r1_h1_S2_RL_dri50_local_H20"),
    ("S1_soft", "b2r1_h1_S1_RL_dri50_local_H20"),
    ("DAI_ctl", "b2r1_h7_negctl_DAI_dri50_local_H20"),
    ("Random_ctl", "b2r1_h7_negctl_random_dri50_local_H20"),
]

VAL_DIR = ROOT / "rollout/val_rollout/DAPO_sh_repro"


def is_aime(input_text: str) -> bool:
    return "Solve the following math problem step by step" in input_text


def compute_metrics_for_rollout(jsonl_path: Path):
    """Read a step.jsonl, separate AIME / IFEval, compute metrics."""
    aime_by_prompt = defaultdict(list)  # prompt_text -> [(acc, response), ...]
    ifeval_by_prompt = defaultdict(list)
    aime_modes = Counter()
    aime_mode_per_prompt = defaultdict(Counter)  # prompt -> Counter

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            inp = o.get("input", "")
            resp = o.get("output", "")
            acc = bool(o.get("acc", False))
            score = float(o.get("score", 0) or 0)
            # Use prompt_text up to the assistant separator as the key.
            # All 32 samples for a prompt will share this key.
            sep = "\nassistant\n"
            if sep in inp:
                pkey = inp[:inp.index(sep)]
            else:
                pkey = inp[:200]

            if is_aime(inp):
                aime_by_prompt[pkey].append((acc, resp, score))
                # classify mode from response
                mode = classify_mode(resp)
                aime_modes[mode] += 1
                aime_mode_per_prompt[pkey][mode] += 1
            else:
                ifeval_by_prompt[pkey].append((acc, resp, score))

    # AIME metrics
    aime_pass1 = []   # mean@32 = avg accuracy across 32
    aime_best32 = []  # any-correct
    aime_mean_chars = []
    for pkey, lst in aime_by_prompt.items():
        accs = [a for a, _, _ in lst]
        aime_pass1.append(sum(accs) / len(accs))
        aime_best32.append(1.0 if any(accs) else 0.0)
        aime_mean_chars.append(sum(len(r) for _, r, _ in lst) / len(lst))

    # IFEval metrics — score is already a float in [0,1] for IFEval rewards,
    # but we use acc as a binary; verl's reward fn is structured so 'acc'
    # for IFEval = strict_pass.
    if_pass1 = []
    if_best32 = []
    for pkey, lst in ifeval_by_prompt.items():
        accs = [a for a, _, _ in lst]
        scores = [s for _, _, s in lst]
        # Average score works as IF pass@1
        if_pass1.append(sum(scores) / len(scores))
        if_best32.append(max(scores) if scores else 0.0)

    n_aime_total = sum(aime_modes.values())
    if n_aime_total == 0:
        # no AIME rollouts (shouldn't happen)
        return None

    return {
        "n_aime_prompts": len(aime_by_prompt),
        "n_aime_samples": n_aime_total,
        "aime_pass1": sum(aime_pass1) / len(aime_pass1) if aime_pass1 else 0,
        "aime_best32": sum(aime_best32) / len(aime_best32) if aime_best32 else 0,
        "aime_mean_chars": sum(aime_mean_chars) / len(aime_mean_chars) if aime_mean_chars else 0,
        "DRI_rate": aime_modes.get("DRI", 0) / n_aime_total,
        "DAI_rate": aime_modes.get("DAI", 0) / n_aime_total,
        "Other_rate": aime_modes.get("Other", 0) / n_aime_total,
        "CSI_rate": aime_modes.get("CSI", 0) / n_aime_total,
        "n_ifeval_prompts": len(ifeval_by_prompt),
        "n_ifeval_samples": sum(len(v) for v in ifeval_by_prompt.values()),
        "ifeval_pass1": sum(if_pass1) / len(if_pass1) if if_pass1 else 0,
        "ifeval_best32": sum(if_best32) / len(if_best32) if if_best32 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis/tables/wave5_final_eval.csv"))
    args = ap.parse_args()

    rows = []
    for run_label, run_name in RUNS:
        run_dir = VAL_DIR / run_name
        if not run_dir.is_dir():
            print(f"  skip: {run_dir} not found")
            continue
        jsonls = sorted(run_dir.glob("*.jsonl"),
                        key=lambda p: int(p.stem))
        for jp in jsonls:
            step = int(jp.stem)
            print(f"  processing {run_label} step={step} ({jp.name})...")
            m = compute_metrics_for_rollout(jp)
            if m is None: continue
            row = {"run": run_label, "step": step, "path": str(jp)}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["run", "step"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    # Headline summary
    cols = ["run", "step", "aime_best32", "aime_pass1", "DRI_rate", "DAI_rate",
            "Other_rate", "ifeval_pass1", "ifeval_best32"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
