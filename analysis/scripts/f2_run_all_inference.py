#!/usr/bin/env python
"""
f2_run_all_inference.py

Runs inference for all 4 models across all 6 format variants.
Uses vLLM with tensor parallelism across 4 GPUs.
Processes models sequentially (to avoid OOM), formats batched per model.
"""

import os
import json
import sys
import time
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(os.environ.get("PROJECT_ROOT", "."))
OUTPUT_DIR = BASE_DIR / "analysis" / "data" / "f2_answer_format"

MODELS = {
    "qwen3_base": str(BASE_DIR / "models" / "Qwen3-8B-Base"),
    "qwen3_ifrlvr": str(BASE_DIR / "checkpoints" / "verl_exp" / "DAPO_sh_repro" / "b2r1_Qwen3-8B-Base_IFTrain_local_H20" / "global_step_100" / "actor" / "huggingface"),
    "qwen25_base": str(BASE_DIR / "models" / "Qwen2.5-Math-7B"),
    "qwen25_ifrlvr": str(BASE_DIR / "checkpoints" / "verl_exp" / "DAPO_sh_repro" / "b2r1_Qwen2.5-math-7B_IFTrain_local_H20" / "global_step_380" / "actor" / "huggingface"),
}

FORMAT_KEYS = ['answer_colon', 'the_answer_is', 'solution_colon', 'final_answer', 'boxed', 'answer_equals']

N_SAMPLES = 16
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 8192


def run_model(model_key: str, model_path: str):
    """Run inference for one model across all format variants."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_key} -> {model_path}")
    print(f"{'='*80}")
    sys.stdout.flush()

    # Check which formats still need processing
    formats_todo = []
    for fmt_key in FORMAT_KEYS:
        out_path = OUTPUT_DIR / f"rollouts_{model_key}_{fmt_key}.jsonl"
        if out_path.exists():
            with open(out_path) as f:
                n_lines = sum(1 for _ in f)
            expected = 158 * N_SAMPLES
            if n_lines >= expected:
                print(f"  [SKIP] {fmt_key}: already complete ({n_lines} lines)")
                continue
            else:
                print(f"  [RESUME] {fmt_key}: have {n_lines}/{expected} lines")
                formats_todo.append(fmt_key)
        else:
            formats_todo.append(fmt_key)

    if not formats_todo:
        print(f"  All formats complete for {model_key}, skipping model load.")
        return

    print(f"  Formats to process: {formats_todo}")
    sys.stdout.flush()

    # Load tokenizer
    t0 = time.time()
    print(f"  Loading tokenizer...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Load vLLM engine
    t0 = time.time()
    print(f"  Loading vLLM engine (tp=4)...")
    sys.stdout.flush()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=4,
        max_model_len=MAX_NEW_TOKENS + 2048,
        seed=42,
        gpu_memory_utilization=0.90,
    )
    print(f"  vLLM loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        n=N_SAMPLES,
        seed=42,
    )

    # Process each format variant
    for fmt_key in formats_todo:
        out_path = OUTPUT_DIR / f"rollouts_{model_key}_{fmt_key}.jsonl"
        fmt_df = pd.read_parquet(OUTPUT_DIR / f"prompts_{fmt_key}.parquet")

        print(f"\n  [{fmt_key}] Generating {len(fmt_df)} prompts × {N_SAMPLES} samples...")
        sys.stdout.flush()

        # Build prompts with chat template
        prompts_text = []
        for _, row in fmt_df.iterrows():
            msgs = row['prompt']
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts_text.append(text)

        # Run inference
        t0 = time.time()
        outputs = llm.generate(prompts_text, sampling_params)
        elapsed = time.time() - t0
        print(f"  [{fmt_key}] Generated in {elapsed:.1f}s")
        sys.stdout.flush()

        # Write results
        with open(out_path, 'w') as f:
            for i, output in enumerate(outputs):
                row = fmt_df.iloc[i]
                for j, completion in enumerate(output.outputs):
                    record = {
                        'id': row['id'],
                        'source': row['source'],
                        'format_variant': fmt_key,
                        'ground_truth': row['ground_truth'],
                        'sample_id': j,
                        'output': completion.text,
                        'model': model_key,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        n_written = 158 * N_SAMPLES
        print(f"  [{fmt_key}] Saved {n_written} lines -> {out_path.name}")
        sys.stdout.flush()

    # Free GPU memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    print(f"\n  Model {model_key} complete, GPU memory freed.")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("F2: Answer Format Generalization Experiment - Full Inference")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Formats: {FORMAT_KEYS}")
    print(f"  Prompts: 158 (30 AIME24 + 128 math_probe)")
    print(f"  Samples per prompt: {N_SAMPLES}")
    print(f"  Total generations: {len(MODELS) * len(FORMAT_KEYS) * 158 * N_SAMPLES}")
    print("=" * 80)
    sys.stdout.flush()

    for model_key, model_path in MODELS.items():
        run_model(model_key, model_path)

    print("\n\nALL DONE! All 4 models × 6 formats complete.")


if __name__ == '__main__':
    main()
