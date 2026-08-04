# Paper figure provenance

The figures in this directory are curated static exports from **Verifier-Induced
Support Reshaping in On-Policy Optimization**, arXiv:2608.00220v1. The paper was
first submitted to arXiv on July 31, 2026 and is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

The original vector PDF files were recovered from the official arXiv TeX source
at <https://arxiv.org/e-print/2608.00220>. They were converted deterministically
to standalone SVG with Poppler's `pdftocairo -svg`. No data, labels, legends,
colors, or scientific content were changed. The conversion only changes the
web delivery format; no OCR, screenshot reconstruction, manual redrawing, or
generative image model was used.

| Web asset | Paper item | Original arXiv source |
|---|---|---|
| `fig1-overview.svg` | Figure 1, paper overview | `figures/fig1_intro.pdf` |
| `fig2-if-polarization.svg` | Figure 2, IF support polarization | `figures/fig_4_1_support_shift_times.pdf` |
| `fig3-math-searchability.svg` | Figure 3, math searchability and visible opening routes | `figures/fig_4_1_2_opening_routes_singlecol.pdf` |
| `fig6-opening-divergence.svg` | Figure 6, position-wise distribution shift | `figures/fig_5_1_js_position_lollipop_singlecol.pdf` |
| `fig8-opening-intervention-a.svg` | Figure 8a, route-token and prefix interventions | `figures/fig_5_4_six_intervention_dotlines.pdf` |
| `fig8-opening-intervention-b.svg` | Figure 8b, intervention position sweep | `figures/fig_5_4_position_sweep.pdf` |
| `fig9-dri-prior.svg` | Figure 9, routing-prior cold start | `figures/fig_6_1_cold_start_stacked.pdf` |
| `fig11-teacher-state.svg` | Figure 11, OPD teacher-state comparison | `figures/fig_6_3_teacher_selection_scan.pdf` |

No whitespace cropping was required for these standalone source figures. Figure
8 is preserved as the two vector panels supplied in the arXiv source and is
grouped semantically on the project page.

`../og-card.png` is a deterministic 1200×630 social preview composed from the
unmodified Figure 1 export and code-rendered paper title and author text. The
Figure 1 region was resized to fit the preview but was not relabeled or edited.

## License boundary

The paper figures above remain covered by the paper's CC BY 4.0 license. The
repository's Apache License 2.0 covers first-party code and site text, but it
does not replace the paper-figure license. Vendored software and third-party
evaluation components retain their own notices as documented in the repository
root.
