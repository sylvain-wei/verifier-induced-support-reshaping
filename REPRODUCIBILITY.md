# Reproducibility status

This document separates release checks, executable code paths, and unavailable
paper artifacts. A passing integrity check is not evidence of end-to-end paper
reproduction.

## Support matrix

| Level | Status from a fresh clone | Requirements |
|---|---|---|
| Manifest, privacy, syntax, and package-boundary checks | Supported | Python 3.10+ |
| CPU unit tests for released metrics and verifiers | Supported after installing `eval/requirements-test.txt` | No GPU or model weights |
| Public benchmark preparation | Supported where a public fallback is listed | Network access and provider terms |
| Base-model inference | Configured but external | `Qwen/Qwen3-8B-Base`; the original local snapshot revision was not recorded |
| Evaluation of paper RLVR checkpoints | Code included, artifacts unavailable | Fine-tuned checkpoints at the paths in `eval/configs/models.yaml` |
| Full paper training runs | Partial code release | GPU stack, site-specific launcher/scheduler configuration, datasets, and compute |
| OPD experiments | Partial code release | External `OPD_LAB_ROOT`, teacher checkpoints, and output storage |
| Published numeric table inspection | Supported | Released CSV/TeX files under `analysis/paper/tables/` |
| Regeneration of every table and figure | Not supported from this repository alone | Raw logs, intermediate CSVs, checkpoints, rollouts, and excluded rendering code |

## External model artifacts

The base model is publicly identified as `Qwen/Qwen3-8B-Base`. The exact Hub
revision used to create the original local snapshot is not recoverable from the
release source, so no revision hash is asserted.

The following paper artifacts are expected locally and are not included:

- Math-RLVR checkpoint at the configured global step.
- IF-RLVR checkpoint at the configured global step.
- Math-to-IF and IF-to-Math sequential checkpoints.
- Routing-prior cold-start checkpoints and trajectories.
- OPD teacher and student checkpoints.
- Generated response files, rollout logs, and most intermediate analysis CSVs.

The paths and experiment labels are recorded in `eval/configs/models.yaml` and
`eval/configs/models_mathif_reasonif.yaml`.

## Training boundary

The bundled `verl/` and `configs/dapo/` trees expose the framework and reference
DAPO recipes. Paper-specific wrappers under `analysis/scripts/` still require
site infrastructure through variables such as `RL_LAUNCH_SCRIPT`,
`OPD_LAB_ROOT`, and `OPD_OUTPUT_ROOT`. Scheduler setup, storage layout, and some
launch commands are external. Therefore, this release supports code inspection
and adaptation but not a turnkey recreation of all paper checkpoints.

The primary reported setup was one node with 8 NVIDIA H20 GPUs with 96 GB per
GPU. Compute time and full cluster-software versions are not recoverable from
the archived package.

## Evaluation boundary

The evaluation pipeline provides public dataset fallbacks and fixed decoding
plans. It requires the configured model directories. Use:

```bash
cd eval
python scripts/check_env.py --strict
bash scripts/dry_run.sh
bash scripts/run_rq1_required.sh
```

`check_env.py --strict` is the gate: it exits nonzero until required packages
and model paths are available. A non-strict environment report is diagnostic
only.

## Table artifacts

Selected paper tables are included as CSV and TeX under
`analysis/paper/tables/`. These files make the reported numbers directly
inspectable. They are derived artifacts, not raw experiment logs.

`analysis/scripts/make_paper_tables.py` requires these unbundled intermediate
files for full regeneration:

- `analysis/tables/d2_pos1_ratio_bootstrap.csv`
- `analysis/tables/c_summary.csv`
- `analysis/tables/e3_val_metrics.csv`
- `analysis/tables/e3_3class_share.csv`
- `analysis/tables/mathifpaper_overall.csv`
- `analysis/tables/reasonifpaper_overall.csv`

The script now reports this list cleanly instead of failing with an unexplained
traceback.

## Release procedure

1. Finish all content changes.
2. Remove caches, generated responses, local models, checkpoints, and secrets.
3. Run the CPU unit tests in the pinned test environment.
4. Run shell syntax checks for released `.sh` files.
5. Run `python scripts/rebuild_manifest.py`.
6. Run `python verify_release.py`.
7. Create a clean Git repository or release archive and scan its full history.
8. Tag the commit used by the corresponding arXiv version.

The manifest must be rebuilt last because every intentional content change
changes one or more SHA-256 values.
