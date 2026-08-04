# Third-party notices

The repository-level Apache License 2.0 applies to first-party material unless
a file or component states otherwise. The following third-party components are
included or adapted in this release.

## Peking University institutional mark

- Official source: <https://vim.pku.edu.cn/xzzq/index.htm>
- Included path: `docs/assets/pku-logo.svg`
- Source file: `标志与中英文校名组合规范.eps` from the official visual identity download package
- Rights: Peking University retains all rights in its name, emblem, and institutional identity

The mark is included only to identify the authors' institutional affiliation.
It is not covered by this repository's Apache License 2.0. The vector paths,
wordmarks, colors, and proportions were not redrawn or altered; the official
EPS was deterministically converted and cropped to its supplied horizontal
lockup for web delivery. See `docs/assets/README.md` for conversion details.

## verl and DAPO recipe

- Upstream: <https://github.com/volcengine/verl>
- Included paths: `verl/` and `configs/dapo/`
- Bundled version string: `0.7.0.dev`
- License: Apache License 2.0
- Copyright notice: Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
- Local notices: `verl/LICENSE` and `verl/Notice.txt`

The exact upstream Git commit of the bundled snapshot was not recorded in the
source archive from which this public package was prepared. Files may contain
project-specific modifications. This limitation is recorded here rather than
claiming commit-level provenance that cannot be verified.

## Google Research IFEval verifier

- Upstream: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- Adapted paths: `eval/src/metrics/if_verifier/`
- License: Apache License 2.0
- Copyright notice: The Google Research Authors

The released verifier is an adapted, reduced implementation for the supported
rule-based constraints. The exact upstream Git commit was not recorded in the
source archive.

## Allen Institute for AI IFBench verifier

- Upstream: <https://github.com/allenai/IFBench>
- Included paths: `eval/src/metrics/if_verifier_ifbench/`
- License: Apache License 2.0
- Copyright notice: Copyright 2025 Allen Institute for AI

Original file-level copyright and license headers are retained. The exact
upstream Git commit was not recorded in the source archive.

## Models and datasets

Model weights and benchmark corpora are not redistributed in this repository.
Their identifiers in `eval/configs/` are references to external projects and do
not place those assets under this repository's license. Users must follow the
licenses and terms published by each model or dataset provider.
