#!/usr/bin/env python
"""C1 — Launch the bundled verl FSDP SFT trainer with a chat_template override
that matches the bare-RL training format `user\\n<content>\\nassistant\\n`.

Why a wrapper: verl's single-turn SFT dataset hard-codes
    tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                                  add_generation_prompt=True, tokenize=False)
and Qwen3-8B-Base ships with the Qwen ChatML template
    `<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n`
which is NOT the format DAPO RL training/eval uses. To keep the SFT signal
on-distribution with RL, we install a custom Jinja chat_template that emits
the exact `user\\n{content}\\nassistant\\n` format used by all rollouts.

This script:
  1) Copies the b2r1-100 model dir into a fresh dir under analysis/data/c1_ckpt_init/
     and rewrites its tokenizer_config.json's chat_template field.
  2) Then launches the bundled verl FSDP SFT trainer with the patched dir as
     model.partial_pretrain.

Output checkpoint goes to checkpoints/verl_exp/disentangle/c1_format_sft/...
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_ROOT", "."))

BARE_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}\n{{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant\n{% endif %}"
)


def patch_init_dir(src_model: str, dst_dir: str) -> str:
    """Copy model files (config, tokenizer) into dst_dir and overwrite chat_template.

    Hard-link safetensors to avoid 30 GB duplication.
    """
    if os.path.isdir(dst_dir):
        print(f"[c1-launch] reusing existing {dst_dir}")
    else:
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_model):
            src = os.path.join(src_model, fname)
            dst = os.path.join(dst_dir, fname)
            # Skip subdirectories (e.g. .cache/) and dotfiles that are not
            # required by HF tokenizer/model loading.
            if os.path.isdir(src):
                continue
            if fname.startswith(".git"):
                continue
            if fname.endswith(".safetensors") or fname.endswith(".bin"):
                # hard link (same fs assumed)
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy(src, dst)
            else:
                shutil.copy(src, dst)
        print(f"[c1-launch] staged init dir -> {dst_dir}")
    # Patch tokenizer_config.json
    tc_path = os.path.join(dst_dir, "tokenizer_config.json")
    if not os.path.isfile(tc_path):
        raise FileNotFoundError(tc_path)
    with open(tc_path) as f:
        tc = json.load(f)
    tc["chat_template"] = BARE_TEMPLATE
    with open(tc_path, "w") as f:
        json.dump(tc, f, indent=2)
    print(f"[c1-launch] patched chat_template in {tc_path}")
    # Sanity-check round-trip via transformers.
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(dst_dir, trust_remote_code=False)
    s = tk.apply_chat_template(
        [{"role": "user", "content": "TEST_PROMPT"}],
        add_generation_prompt=True, tokenize=False,
    )
    print(f"[c1-launch] sanity template output: {s!r}")
    expected = "user\nTEST_PROMPT\nassistant\n"
    if s != expected:
        raise RuntimeError(f"chat_template patch did not produce expected format. got {s!r} want {expected!r}")
    return dst_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model", default=os.path.join(PROJECT_ROOT, "checkpoints", "verl_exp", "DAPO_sh_repro", "b2r1_Qwen3-8B-Base_IFTrain_local_H20", "global_step_100", "actor", "huggingface"))
    ap.add_argument("--staged_init_dir", default=os.path.join(PROJECT_ROOT, "analysis", "data", "c1_ckpt_init_b2r1_100_baretpl"))
    ap.add_argument("--train_parquet", default=os.path.join(PROJECT_ROOT, "analysis", "data", "c1_sft_corpus_dri.parquet"))
    ap.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "verl_exp", "disentangle", "c1_format_sft"))
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--micro_batch_size_per_gpu", type=int, default=1)
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--prompt_key", default="prompt_text")
    ap.add_argument("--response_key", default="response_text")
    ap.add_argument("--subsample_n", type=int, default=0,
                    help="If >0, deterministically subsample N rows from the flat parquet "
                         "after building it. Useful for dose sweeps (WAVE 5).")
    ap.add_argument("--subsample_seed", type=int, default=0,
                    help="Seed for subsample shuffle (used only when --subsample_n>0).")
    ap.add_argument("--experiment_name", default="c1_format_sft_b2r1_100_dri",
                    help="verl trainer.experiment_name. Override per dose for WAVE 5.")
    args = ap.parse_args()

    init_dir = patch_init_dir(args.init_model, args.staged_init_dir)

    # Build a flattened (prompt_text, response_text) parquet from the
    # messages-key parquet so the single-turn SFT dataset can read it.
    import pandas as pd
    df = pd.read_parquet(args.train_parquet)
    if "messages" in df.columns:
        prompts = []
        responses = []
        for msgs in df["messages"]:
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prompts.append(user)
            responses.append(asst)
        flat_df = pd.DataFrame({"prompt_text": prompts, "response_text": responses})
        flat_path = os.path.splitext(args.train_parquet)[0] + "_flat.parquet"
        flat_df.to_parquet(flat_path, index=False)
        print(f"[c1-launch] wrote flat parquet -> {flat_path} ({len(flat_df)} rows)")
        train_files = flat_path
    else:
        train_files = args.train_parquet

    # Optional dose subsample: deterministic shuffle (numpy) + head(N).
    if args.subsample_n and args.subsample_n > 0:
        sub_df = pd.read_parquet(train_files)
        n_full = len(sub_df)
        if args.subsample_n > n_full:
            print(f"[c1-launch] WARN: subsample_n={args.subsample_n} > corpus size {n_full}; "
                  f"using all {n_full} rows.")
            n_keep = n_full
        else:
            n_keep = args.subsample_n
        # Deterministic shuffle by seed; reset index for parquet hygiene.
        sub_df = sub_df.sample(frac=1.0, random_state=args.subsample_seed).reset_index(drop=True)
        sub_df = sub_df.head(n_keep).reset_index(drop=True)
        sub_path = (
            os.path.splitext(train_files)[0]
            + f"__sub{n_keep}_seed{args.subsample_seed}.parquet"
        )
        sub_df.to_parquet(sub_path, index=False)
        print(f"[c1-launch] wrote subsampled parquet -> {sub_path} ({n_keep} rows)")
        train_files = sub_path

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", f"--nproc_per_node={args.n_gpus}",
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={train_files}",
        f"data.val_files={train_files}",  # tiny set; reuse for val
        f"data.prompt_key={args.prompt_key}",
        f"data.response_key={args.response_key}",
        f"data.max_length={args.max_length}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"model.partial_pretrain={init_dir}",
        f"trainer.default_local_dir={save_dir}",
        "trainer.project_name=disentangle",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.total_epochs={args.epochs}",
        "trainer.logger=['console']",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        f"optim.lr={args.lr}",
        "optim.lr_scheduler=wsd",
        "optim.warmup_steps_ratio=0.0",
        "model.fsdp_config.model_dtype=bf16",
    ]
    print("[c1-launch] CMD:")
    print("  " + " \\\n    ".join(cmd))
    env = os.environ.copy()
    verl_path = os.path.join(PROJECT_ROOT, "verl")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (verl_path, existing_pythonpath) if p)
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
