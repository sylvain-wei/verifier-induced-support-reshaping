<div align="center">

# Verifier-Induced Support Reshaping in On-Policy Optimization

[![Paper](https://img.shields.io/badge/Paper-Preprint-b31b1b?style=for-the-badge)](#-paper-and-citation)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Matrix-2f6f9f?style=for-the-badge)](REPRODUCIBILITY.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c8c2b?style=for-the-badge)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)

**Shaohang Wei<sup>1</sup>, Zikun Su<sup>2</sup>, Feifan Song<sup>1</sup>, Wen Luo<sup>1</sup>, Wei Li<sup>1</sup>, Guangyue Peng<sup>1</sup>, and Houfeng Wang<sup>1</sup>**

<sup>1</sup>Peking University &nbsp;&nbsp; <sup>2</sup>BUPT

**Correspondence:** [Houfeng Wang](mailto:wanghf@pku.edu.cn) and [Shaohang Wei](mailto:shaohang@stu.pku.edu.cn)

</div>

## 📋 Project Information

This repository contains the public research code and selected analysis
artifacts for **Verifier-Induced Support Reshaping in On-Policy Optimization**.
It supports inspection of the training recipes, verifier implementations,
evaluation protocol, analysis code, and released numeric tables.

The arXiv link will be added after the public record is available. For the
current artifact boundary, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 📖 Abstract

> We show that on-policy reinforcement learning with verifiable rewards (RLVR)
> can improve the current objective while making successful behaviors for later
> objectives too rare to sample and reinforce. We call this verifier-induced
> support reshaping and define effective rewardable support as successful
> trajectories reachable within a fixed rollout budget. Across two model
> families, we study this effect through repeated verifier-scored sampling and
> bidirectional training on mathematical reasoning and constrained instruction
> following, including sequential training with the opposite verifier.
> Math-RLVR raises average instruction-following success but reduces the number
> of prompts with any successful response under repeated sampling. On IFEval
> with Qwen3-8B-Base, pass@1 rises by 6.5 percentage points while best@32 falls
> by 9.8 percentage points, and the same divergence appears across both models
> and IF benchmarks. Conversely, IF-RLVR shifts math responses from step-by-step
> openings toward direct answers, lowers best@k across sampling budgets, and
> reduces reward variation for later Math-RLVR. Token-distribution analyses and
> controlled opening interventions show that these changes concentrate in the
> first few response tokens. RLVR mainly reranks openings already available in
> the base policy, and the selected opening causally affects math searchability.
> The tested reference-policy constraints, routing priors, and on-policy
> distillation preserve cross-task support only partially; MathIF and ReasonIF
> show that marginal gains translate only partly into responses that are both
> correct and constraint-following. Therefore, endpoint improvements do not
> guarantee future trainability or joint capability under on-policy
> optimization.

## ✨ Highlights

- **Future trainability:** We measure whether successful trajectories for a
  later objective remain reachable within a fixed rollout budget.
- **Bidirectional evaluation:** The release covers mathematical reasoning and
  constrained instruction following under both Math-RLVR and IF-RLVR.
- **Opening-route analysis:** The released analysis code examines how changes
  in the first response tokens affect later sampling and optimization.
- **Explicit artifact boundary:** The repository distinguishes released code
  and tables from unavailable checkpoints, rollouts, logs, and intermediate
  analysis files.

## 🚀 Get Started

### Step 1: Validate the release

From the repository root, run the integrity and release-safety checker:

```bash
python verify_release.py
```

### Step 2: Run the lightweight tests

Python 3.10 or newer is required. These tests do not require a GPU or paper
checkpoints:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r eval/requirements-test.txt
PYTHONPATH=eval python -m unittest discover -s eval/tests -v
```

Four converter tests are skipped when the optional local benchmark files are
absent. The remaining tests cover answer extraction, diversity metrics, text
metrics, and the rule-based instruction-following verifier.

### Step 3: Prepare the full evaluation environment

The full training and vLLM evaluation stacks are CUDA-specific:

```bash
export PROJECT_ROOT="$(pwd)"
python -m pip install -e "${PROJECT_ROOT}/verl"
python -m pip install -r "${PROJECT_ROOT}/eval/requirements.txt"
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"
```

Install PyTorch, Transformers, vLLM, and their GPU dependencies using versions
compatible with the target CUDA driver. Configure
`eval/configs/models.yaml` and `eval/configs/datasets.yaml`, then inspect the
environment:

```bash
cd "${PROJECT_ROOT}/eval"
python scripts/check_env.py
python scripts/check_env.py --strict
```

The default command reports unresolved components. The strict command exits
nonzero when a required package or model path is unavailable.

### Step 4: Run evaluation

After configuring the required checkpoint and dataset paths:

```bash
cd "${PROJECT_ROOT}/eval"
bash scripts/dry_run.sh
bash scripts/run_rq1_required.sh
```

The smoke run uses two examples per required evaluation cell. The full launcher
prepares MATH-500, AIME 2024, GSM8K, IFEval, and the rule-checkable IFBench
subset; performs inference for the configured base, Math-RLVR, and IF-RLVR
checkpoints; computes metrics; and aggregates the results. Exact decoding
settings are in `eval/configs/eval_plan.yaml`.

## 📦 Release Scope

This is a research-code release, not a one-command reproduction of every paper
result.

| Status | Included material |
|---|---|
| Included | A bundled verl/DAPO framework snapshot, reward and verifier code, evaluation and analysis scripts, unit tests, and selected paper tables |
| Public external input | The Qwen3-8B-Base model and benchmark datasets listed in `eval/configs/` |
| Required but not included | Fine-tuned paper checkpoints, generated rollouts, raw experiment logs, most intermediate analysis CSVs, and site-specific cluster launchers |
| Out of scope | Manuscript figure rendering and styling code |

The missing large artifacts prevent end-to-end numerical reproduction from a
fresh clone. They do not prevent inspection of the released algorithms,
verification logic, evaluation protocol, or published numeric tables.

## 🗂️ Repository Layout

```text
.
├── configs/dapo/          DAPO recipe and reference launch configurations
├── verl/                  bundled verl training framework snapshot
├── eval/                  inference, scoring, aggregation, and unit tests
├── scripts_eval/          checkpoint-evaluation entry points
├── analysis/scripts/      non-visual analyses and training support
├── analysis/paper/tables/ selected released paper tables
├── REPRODUCIBILITY.md     artifact availability and reproduction boundary
├── THIRD_PARTY_NOTICES.md vendored-code provenance and licenses
├── MANIFEST.tsv           SHA-256 inventory of released files
└── verify_release.py      integrity, privacy, and release-safety checker
```

## 🧪 Training and Analysis

`configs/dapo/` and `verl/` expose the training implementation and reference
recipes. Several experiment wrappers require an external `RL_LAUNCH_SCRIPT`,
`OPD_LAB_ROOT`, or cluster scheduler configuration. These site-specific
components are not included, so the repository does not provide a turnkey
retraining command for every paper checkpoint.

Selected final numeric tables are stored under `analysis/paper/tables/`.
`analysis/scripts/make_paper_tables.py` checks for the intermediate analysis
CSVs needed to regenerate derived tables and reports missing inputs explicitly.
The raw logs and intermediate CSVs are not bundled.

The primary setup used for the reported training experiments was one node with
8 NVIDIA H20 GPUs with 96 GB of memory per GPU.

### Common environment variables

- `CUDA_VISIBLE_DEVICES` and `TENSOR_PARALLEL_SIZE` select evaluation GPUs.
- `CONDA_ENV` lets a launcher activate a caller-selected Conda environment.
- `RL_LAUNCH_SCRIPT` points to a site-specific IF-RLVR launcher.
- `OPD_LAB_ROOT` and `OPD_OUTPUT_ROOT` point to an external OPD runner and its
  outputs.
- `DEEPSEEK_API_KEY` is used only by optional external-judge scripts; no
  credential is included.

## 🔎 Release Verification

Run `python verify_release.py` after extraction. The checker validates required
metadata, manifest coverage and hashes, first-party Python syntax, shell and
junk-file hygiene, symlinks, large Git blobs, private paths, common secret
patterns, and the no-figure-code release boundary.

After intentional file changes, rebuild the manifest last:

```bash
python scripts/rebuild_manifest.py
python verify_release.py
```

These checks establish repository integrity and release hygiene. They do not
prove that GPU training or checkpoint-dependent evaluation completed.

## 💬 Paper and Citation

If this repository supports your research, please consider citing the paper.
The arXiv identifier and repository URL will be added to `CITATION.cff` after
their public records exist.

```bibtex
@misc{wei2026verifier,
  title  = {Verifier-Induced Support Reshaping in On-Policy Optimization},
  author = {Wei, Shaohang and Su, Zikun and Song, Feifan and Luo, Wen and Li, Wei and Peng, Guangyue and Wang, Houfeng},
  year   = {2026},
  note   = {Preprint}
}
```

Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

## 📄 License and Acknowledgments

First-party code is released under the Apache License 2.0 in [LICENSE](LICENSE).
Vendored components retain their own notices; see [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and component-local license
files.

This release builds on the verl training framework and includes adapted
evaluation components from Google Research IFEval and AllenAI IFBench. For
manuscript preparation, AI assistants were used only for translation and
language polishing.

