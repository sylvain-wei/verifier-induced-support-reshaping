# Analysis artifacts

`scripts/` contains non-visual analysis and training-support code.
`paper/tables/` contains selected released numeric tables in CSV and TeX form.

The final tables are inspectable release artifacts. Most raw logs, model
checkpoints, generated rollouts, and intermediate CSVs are not bundled. As a
result, `scripts/make_paper_tables.py` cannot regenerate every table from a
fresh clone. It performs an input preflight and lists the missing inputs; pass
`--available-only` only when intentionally regenerating the tables that do not
depend on external intermediate CSVs.

No manuscript figure rendering or styling code is included in this public
package.
