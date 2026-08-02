#!/usr/bin/env python
"""C1 — Build minimal-dose format SFT corpus from b0's correct DRI rollouts.

Pipeline:
  1) Read primary rollouts JSONL(s) (step_0.jsonl from math_support_probe — i.e.,
     b0 on the 128-prompt math-7.5k subset).
  2) Filter to (acc=True, classify_mode(response)=='DRI').
  3) Per unique prompt, take the SHORTEST correct DRI response.
  4) Optionally augment from --backup_rollouts (e.g., a fresh wider rollout).
  5) Cap at --cap unique prompts.
  6) Emit messages-key parquet for verl SFT trainer:
       row = {messages: [{role:user, content:...}, {role:assistant, content:...}]}

The "user" content is recovered from the rollout `input` field by stripping
"user\\n" prefix and "\\nassistant\\n" suffix — same as the rest of the analysis
pipeline.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_opening_modes import classify_mode  # noqa: E402


def _extract_user_content(s: str) -> str:
    s2 = s
    if s2.startswith("user\n"):
        s2 = s2[len("user\n"):]
    if s2.endswith("\nassistant\n"):
        s2 = s2[:-len("\nassistant\n")]
    elif s2.endswith("assistant\n"):
        s2 = s2[:-len("assistant\n")]
    return s2.strip()


def _read_correct_dri(jsonl_path: str) -> Dict[str, dict]:
    """prompt_id -> {user_content, response (shortest correct DRI), pid}."""
    if not os.path.isfile(jsonl_path):
        return {}
    by_pid: Dict[str, dict] = {}
    with open(jsonl_path) as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except Exception:
                continue
            acc_raw = o.get("acc")
            if isinstance(acc_raw, bool):
                acc = acc_raw
            else:
                try:
                    acc = bool(float(o.get("score", 0.0)) >= 1.0)
                except Exception:
                    continue
            if not acc:
                continue
            resp = o.get("output", "")
            if not resp:
                continue
            if classify_mode(resp) != "DRI":
                continue
            inp = o.get("input", "")
            uc = _extract_user_content(inp)
            pid = str(o.get("prompt_id") or uc[:80])
            cur = by_pid.get(pid)
            if cur is None or len(resp) < len(cur["response"]):
                by_pid[pid] = {
                    "prompt_id": pid,
                    "user_content": uc,
                    "response": resp,
                }
    return by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="Primary jsonl (e.g. b0 math probe step 0)")
    ap.add_argument("--backup_rollouts", default=None,
                    help="Optional fresh wider rollout to fill shortfalls")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--out", required=True,
                    help="Output parquet path (messages-key SFT format)")
    args = ap.parse_args()

    primary = _read_correct_dri(args.rollouts)
    print(f"[c1-corpus] primary: {len(primary)} unique correct-DRI prompts")
    pool: Dict[str, dict] = dict(primary)
    if args.backup_rollouts:
        backup = _read_correct_dri(args.backup_rollouts)
        added = 0
        for pid, d in backup.items():
            if pid not in pool:
                pool[pid] = d
                added += 1
        print(f"[c1-corpus] backup: {len(backup)} extra unique; added {added} new")
    print(f"[c1-corpus] total pool: {len(pool)}")

    # Sort deterministically by prompt_id for reproducibility.
    items = sorted(pool.values(), key=lambda d: d["prompt_id"])[: args.cap]
    print(f"[c1-corpus] keeping {len(items)} (cap={args.cap})")

    rows = []
    for d in items:
        rows.append({
            "messages": [
                {"role": "user", "content": d["user_content"]},
                {"role": "assistant", "content": d["response"]},
            ],
            "prompt_id": d["prompt_id"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[c1-corpus] wrote {args.out} ({len(df)} rows)")
    # Also dump a JSONL sibling for inspection.
    sib = os.path.splitext(args.out)[0] + ".jsonl"
    with open(sib, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[c1-corpus] wrote {sib}")


if __name__ == "__main__":
    main()
