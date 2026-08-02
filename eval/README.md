# Evaluation Pipeline

This directory evaluates the base, Math-RLVR, IF-RLVR, and configured
sequential checkpoints on math and instruction-following benchmarks. It
supports data preparation, vLLM or Hugging Face inference, per-response
metrics, aggregation, and small smoke tests.

## Configuration

All filesystem locations are portable:

- `configs/models.yaml` lists model and tokenizer locations.
- `configs/datasets.yaml` lists local dataset candidates and public fallbacks.
- `configs/prompts.yaml` defines task prompts.
- `configs/eval_plan.yaml` defines models, benchmarks, decoding, and seeds.

The literal `PROJECT_ROOT` token in YAML files is expanded by the configuration
loader. Set it to the extracted package root:

```bash
export PROJECT_ROOT="$(cd .. && pwd)"
```

The model registry uses the Qwen chat template for all compared checkpoints,
matching the serialized prompt format used during RLVR training. The prompt
content itself does not request step-by-step reasoning unless a benchmark name
ends in `_cot`.

## Dependencies

Install the bundled training framework and the evaluation additions from the
package root:

```bash
python -m pip install -e "${PROJECT_ROOT}/verl"
python -m pip install -r "${PROJECT_ROOT}/eval/requirements.txt"
```

The base environment must also provide compatible versions of PyTorch,
Transformers, vLLM, Datasets, NumPy, pandas, PyArrow, PyYAML, SymPy, regex,
pylatexenc, and tqdm. Run `python scripts/check_env.py` to report missing
packages and unresolved model or dataset locations.

The primary training setup for the reported experiments was **8 NVIDIA H20
GPUs with 96 GB per GPU**. Evaluation GPU count can be selected with
`CUDA_VISIBLE_DEVICES` and `TENSOR_PARALLEL_SIZE`; memory requirements depend on
the chosen model, context length, and batch settings.

## Quick validation

From `eval/`:

```bash
export CUDA_VISIBLE_DEVICES=0
export TENSOR_PARALLEL_SIZE=1
python scripts/check_env.py
bash scripts/dry_run.sh
```

The smoke run processes two examples per required evaluation cell and writes to
`responses/${RUN_ID}` and `metrics/${RUN_ID}`. Missing model weights or data are
reported rather than silently replaced.

## Full required matrix

```bash
bash scripts/run_rq1_required.sh
```

The launcher performs:

1. environment and path reporting;
2. preparation of MATH-500, AIME 2024, GSM8K, IFEval, and IFBench;
3. inference for the base, Math-RLVR, and IF-RLVR checkpoints;
4. per-run metric computation; and
5. aggregation into `metrics/${RUN_ID}/aggregate/`.

The current plan uses sampling with `K=16` for MATH-500, sampling with `K=32`
for AIME 2024, and greedy decoding for GSM8K, IFEval, and the rule-checkable
IFBench set. Exact decoding values are defined in `configs/eval_plan.yaml`.

## Individual stages

```bash
python scripts/prepare_data.py --only math500 aime24 gsm8k ifeval ifbench

python scripts/run_inference.py \
  --run-id smoke_manual \
  --model-id base \
  --benchmark math500 \
  --mode sampling_k16 \
  --limit 2

python scripts/compute_metrics.py \
  --run-id smoke_manual \
  --model-id base \
  --benchmark math500 \
  --mode sampling_k16

python scripts/aggregate_metrics.py --run-id smoke_manual
python scripts/make_plot_data.py --run-id smoke_manual
```

Inference output is resume-safe. Use `--overwrite` only when intentionally
replacing an existing cell.

## Data and prompt notes

- Global seed and decoding settings are stored in the plan and response
  records.
- Dataset preparation first checks the configured local candidates, then the
  listed public fallback when supported and network access is available.
- IFBench preparation keeps only constraints supported by the bundled
  rule-based verifier; the configured maximum is 300.
- Step-by-step prompt ablations are represented by the `_cot` dataset aliases
  and are separate from the default prompt content.
- Model weights, checkpoints, and generated responses are not included in this
  package.

## Tests

```bash
python -m pip install -r requirements-test.txt
PYTHONPATH=. python -m unittest discover -s tests -v
```

`requirements-test.txt` is a pinned CPU-only dependency snapshot for the unit
tests. It is separate from the CUDA-specific training and vLLM environment.

The package-level `verify_release.py` additionally checks first-party syntax,
manifest integrity, junk files, symlinks, and common privacy or secret markers.

If vLLM reports CUDA reinitialization after a process fork, set:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```
