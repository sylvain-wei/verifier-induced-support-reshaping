# Project-page asset provenance

This directory separates scientific paper figures, an official institutional
mark, and non-scientific editorial artwork. These asset classes have different
sources and license boundaries.

## Peking University mark

- **Included file:** `pku-logo.svg`
- **Official source:** [Peking University Visual Identity Management Office —
  downloads](https://vim.pku.edu.cn/xzzq/index.htm)
- **Official package:** `标志与中英文校名组合规范.zip`
- **Original vector:** `标志与中英文校名组合规范.eps`
- **Selected lockup:** the supplied horizontal emblem, Chinese name, and
  `PEKING UNIVERSITY` composition

The EPS PostScript stream was extracted without editing its drawing content,
converted to PDF with Ghostscript using EPS cropping, converted to SVG with
`pdftocairo`, and limited through the SVG `viewBox` to the official horizontal
lockup. No vector paths, wording, colors, labels, or proportions were redrawn
or altered. The SVG remains a Peking University institutional asset and is not
licensed under the repository's Apache License 2.0.

The official identity guide specifies Beida Red and minimum-size and clear-space
rules. The project page preserves the supplied color and proportions, renders
the lockup above the official minimum size, and keeps it separated from other
marks.

## Scientific figures

Paper figures are stored in `figures/`. Their figure-level source, conversion,
and CC BY 4.0 information is recorded in
[`figures/README.md`](figures/README.md).

## Editorial illustrations

Three generated zine-style interludes are stored in `illustrations/`. They are
conceptual editorial artwork, not paper figures, measurements, model-state
visualizations, or experimental evidence. Generation prompts, tool provenance,
and deterministic WebP conversion details are recorded in
[`illustrations/README.md`](illustrations/README.md).
