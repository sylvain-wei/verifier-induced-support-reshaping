#!/usr/bin/env python
"""Phase 2 — Three-class judge (contentful / shortcut / fail) via DeepSeek.

Why this script
---------------
For the OPD sanity check we must distinguish *contentful-pass* from
*shortcut-pass* among verifier-passed responses, and we must do so on the
same 200-prompt × 32-rollout sample for both base and teacher_if.

Simple rule-based proxies only fire on placeholder, keyword, or very-short
patterns and miss other shortcut shapes such as format-only or repeated-prompt
responses. We therefore use a semantic judge over every response, including
verifier-fail ones, labeled
{contentful, shortcut, fail}, and consider the proxies a redundant cross-check.

Judge prompt
------------
The judge is told the **original IF prompt**, the **constraint list**, and the
**verifier outcome** (so it doesn't have to re-derive constraint compliance).
It returns a JSON object with `{label, reason}`. Label semantics:

  contentful  Verifier passed AND the response contains a substantive answer
              to the prompt's main task. The constraints are satisfied as a
              consequence of producing a real reply, not by gaming the
              verifier (e.g. dumping placeholder/keyword tokens).
  shortcut    Verifier passed BUT the response is a verifier-compatible hack:
              placeholder dump, keyword dump, single short echo, format-only
              wrapper without an actual answer, etc.
  fail        Verifier did not pass, regardless of how substantive the prose
              looks. (We do not split fail-but-substantive from fail-and-empty
              because for the OPD-gradient question we care only whether the
              base trajectory is in the set the teacher will reward.)

Caching
-------
Each (model_tag, prompt_id, sample_id) is cached in `labels_cache.jsonl` so
re-runs are free. Cache keys are stable hashes of (model_tag, prompt_id, sample_id).

Outputs
-------
analysis/data/opd_sanity/labels_cache.jsonl   append-only cache
analysis/data/opd_sanity/responses_labelled.parquet  joined: response + label

Usage
-----
  export DEEPSEEK_API_KEY=...
  python label_responses.py --model_tag base
  python label_responses.py --model_tag teacher_if
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, f"{ROOT}/analysis/scripts")
from deepseek_client import DeepseekClient  # noqa: E402


SYSTEM_PROMPT = """\
You are an expert evaluator that labels candidate responses to instruction-following \
prompts. You will be given (a) the original instruction-following prompt, (b) the \
list of constraint identifiers used by an automatic verifier, (c) whether that \
verifier scored the response as passing all constraints (`verifier_pass`), and \
(d) the candidate response itself.

Your job is to assign exactly one of these labels:

- "contentful": verifier_pass is true AND the response is a substantive answer \
to the prompt's main task. Constraints are met as a side effect of producing a \
real reply.
- "shortcut":   verifier_pass is true BUT the response is a verifier-compatible \
hack: it dumps placeholders/keywords, repeats the prompt, emits only a wrapper \
(JSON shell, brackets, format-only), or is a single short echo without actually \
answering. The verifier rewards it but the response does not address the task.
- "fail":       verifier_pass is false, regardless of how substantive the prose looks.

Output a SINGLE JSON object with exactly two keys:
{
  "label": "contentful" | "shortcut" | "fail",
  "reason": "<one sentence explaining the choice>"
}
Do not output anything else. No markdown fences, no commentary.
"""


def build_user_msg(prompt_text: str, iids: list, verifier_pass: bool,
                   verifier_score: float, response: str,
                   max_response_chars: int = 6000) -> str:
    resp = response if len(response) <= max_response_chars else response[:max_response_chars] + " […TRUNC]"
    return (
        f"Original prompt:\n```\n{prompt_text}\n```\n\n"
        f"Constraint identifiers: {list(iids)}\n"
        f"verifier_pass: {bool(verifier_pass)} (score={float(verifier_score):.3f})\n\n"
        f"Candidate response:\n```\n{resp}\n```\n\n"
        f"Return the JSON object now."
    )


def parse_label(content: str) -> dict:
    """Parse {label, reason} JSON; tolerate stray prose around it."""
    txt = content.strip()
    # Strip markdown fences if any
    if txt.startswith("```"):
        # remove first fence line + trailing ```
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    # Find first { ... } object
    start = txt.find("{")
    end = txt.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in: {content[:200]!r}")
    obj = json.loads(txt[start:end + 1])
    label = str(obj.get("label", "")).strip().lower()
    if label not in ("contentful", "shortcut", "fail"):
        raise ValueError(f"unexpected label: {label!r}")
    reason = str(obj.get("reason", "")).strip()[:400]
    return {"label": label, "reason": reason}


def cache_key(model_tag: str, prompt_id: str, sample_id: int) -> str:
    return f"{model_tag}|{prompt_id}|{sample_id}"


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
                out[o["k"]] = o["v"]
            except Exception:
                continue
    return out


def append_cache(path: str, k: str, v: dict, fh=None):
    line = json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n"
    if fh is not None:
        fh.write(line); fh.flush()
        return
    with open(path, "a") as f:
        f.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True,
                    help="Subdirectory name under rollout/opd_sanity/ (no choice restriction; "
                         "any string OK so this script can be reused for E3 students etc.).")
    ap.add_argument("--rollout_dir", default=f"{ROOT}/rollout/opd_sanity")
    ap.add_argument("--out_dir", default=f"{ROOT}/analysis/data/opd_sanity")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, label only the first N rows (for smoke).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent DeepSeek requests.")
    ap.add_argument("--max_tokens", type=int, default=8192)
    args = ap.parse_args()

    cache_path = args.cache or os.path.join(args.out_dir, "labels_cache.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    in_jsonl = os.path.join(args.rollout_dir, args.model_tag, "responses.jsonl")
    print(f"[label] reading {in_jsonl}")
    rows = []
    with open(in_jsonl) as f:
        for ln in f:
            rows.append(json.loads(ln))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[label]   rows: {len(rows)}")

    print(f"[label] cache at {cache_path}")
    cache = load_cache(cache_path)
    print(f"[label]   cache hits available: {len(cache)}")

    client = DeepseekClient()

    todo = []
    for r in rows:
        k = cache_key(args.model_tag, r["prompt_id"], r["sample_id"])
        if k in cache:
            continue
        todo.append((k, r))
    print(f"[label] need to query: {len(todo)}")

    n_done = 0
    t0 = time.time()
    cache_fh = open(cache_path, "a")

    def _query(item):
        k, r = item
        msg_user = build_user_msg(
            prompt_text=r["input"],
            iids=r["instruction_id_list"],
            verifier_pass=bool(r["acc"]),
            verifier_score=float(r["score"]),
            response=r["output"],
        )
        try:
            content = client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg_user},
                ],
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            v = parse_label(content)
        except Exception as e:
            v = {"label": "ERROR", "reason": repr(e)[:300]}
        return k, v

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_query, it) for it in todo]
            for fut in as_completed(futs):
                k, v = fut.result()
                cache[k] = v
                append_cache(cache_path, k, v, fh=cache_fh)
                n_done += 1
                if n_done % 50 == 0:
                    rate = n_done / (time.time() - t0 + 1e-9)
                    print(f"[label]   {n_done}/{len(todo)}  ({rate:.2f} req/s)")
    cache_fh.close()

    # Build labelled DataFrame.
    labels = []
    reasons = []
    for r in rows:
        k = cache_key(args.model_tag, r["prompt_id"], r["sample_id"])
        v = cache.get(k, {"label": "MISSING", "reason": ""})
        labels.append(v["label"])
        reasons.append(v["reason"])

    df = pd.DataFrame(rows)
    df["label"] = labels
    df["label_reason"] = reasons
    df["model_tag"] = args.model_tag
    out_parquet = os.path.join(args.out_dir, f"responses_labelled_{args.model_tag}.parquet")
    df.to_parquet(out_parquet, index=False)
    print(f"[label] wrote {out_parquet}")
    print()
    print(df["label"].value_counts(dropna=False).to_string())
    err_rate = (df["label"] == "ERROR").mean()
    if err_rate > 0:
        print(f"[label] ERROR rate: {err_rate:.4f}  ({(df['label']=='ERROR').sum()} rows)")


if __name__ == "__main__":
    main()
