# SROIE primary-versus-casefold best-of-K report

CHG-20260926-13. Post hoc sensitivity only; primary case-sensitive outputs remain
authoritative. This report does not evaluate one-output selectors or Vietnamese
casefold performance. A significance-threshold change is not a test of a
metric-by-policy interaction; nonsignificance is not equivalence.

## Sources and output registry

Under `SROIRE_COMPARISON/output_sroie/`, the frozen input snapshots are copied from:

- Primary surrogate: `surrogate_kfold_sroie_level1_fix_all616_g1_nested_fixedtopk_perpage_v2_exactrandom_20260906/per_page_results_{tesseract,easyocr}.csv`.
- Primary NN: `nn_shortlist_sroie_all616_g1_paired_exactrandom_20260906/nn_per_image.csv`.
- Casefold surrogate: `surrogate_kfold_sroie_all616_casefold_sensitivity_20260926_v1/per_page_results_{tesseract,easyocr}.csv`.
- Casefold NN: `nn_shortlist_sroie_all616_casefold_sensitivity_20260926_v1/nn_per_image.csv`.
- Primary statistics: `ijdar_public_release/results/statistical_tests/two_sided_wilcoxon_summary.csv`.

New root: `SROIRE_COMPARISON/output_sroire/casefold_bestk_comparison_20260926_v2/`.
Seven CSVs in `inputs/` retain source bytes; no OCR text, labels or images are
added to the release. `tables/` contains:

- `paired_values.csv`: 9,856 metric/engine/page/K rows, four policy CERs and fold.
- `statistics.csv`: 80 metric/engine/K/contrast rows, mean CERs/differences,
  paired bootstrap CIs, two-sided Wilcoxon p-values and Holm correction.
- `primary_casefold_comparison.csv`: 40 aligned primary/alternative comparisons.
- `baseline_means.csv`: 12 NoPre/oracle/FixedBest means.
- `zero_method_sensitivity.csv`: 24 casefold Surrogate--FixedTopK rows for
  wilcox/pratt/zsplit, separate eight-test Holm family per convention.
- `sroie_casefold_tables.tex`: generated count and NN-versus-Surrogate tables.
- `manifest.json`: source/script/output hashes, package versions, seeds/conventions.

V1 is a retained preview, not publication evidence: its CSV calculations were
valid but two-decimal CI endpoints obscured near-zero bounds as +0.00/-0.00.
V2 uses three significant digits for CI endpoints. V1 includes its script
snapshot and supersession notice; do not silently overwrite either version.

## Method and validation

`export_sroie_casefold_sensitivity.py` consumes only saved per-page results.
It does not OCR or refit models. Both metrics must have the same 616 unique
page IDs and five NN fold assignments, and NN's paired surrogate/baseline
columns must match the surrogate output. Each metric and contrast has exactly
eight tests (two engines x four K); reverse duplicates are not additional tests.

Use Wilcoxon two-sided, zero_method=wilcox, and Holm separately for each metric
and contrast. Primary p-values must reproduce the published primary CSV.
Paired mean-difference percentile CIs use 10,000 resampled pages, 95% coverage
per comparison, no multiplicity correction. The exporter reuses the existing
bootstrap script's base-12345 checksum seed rule for Surrogate comparisons;
NN comparisons use explicitly namespaced comparator tokens. Seeds are stored
in every statistics row and held the same across metrics. This is not the
training runner's separate seed=5200+K CI stream; small Monte Carlo differences
between these two streams are expected. Neither changes the hypothesis family.

## Commands and public reproduction

```bash
./venv_py310/bin/python analysis/test_sroie_casefold_export.py
./venv_py310/bin/python analysis/export_sroie_casefold_sensitivity.py \
  --inputs SROIRE_COMPARISON/output_sroire/casefold_bestk_comparison_20260926_v2/inputs \
  --output-dir SROIRE_COMPARISON/output_sroire/casefold_bestk_comparison_20260926_v2/tables
./venv_py310/bin/python analysis/audit_sroie_casefold_export.py
./venv_py310/bin/python analysis/audit_submission_artifacts.py
```

The exporter refuses differing output artifacts. After code/input changes,
choose a new versioned output root. Input and output directories must be
separate, not nested. See the released bundle's README for a public-only
reproduction command into a fresh output directory.

The focused audit independently replays all 80 CIs and tests, checks source and
public copies, and verifies generated TeX inclusion. The existing submission
audit remains a separate internal-consistency check, not scientific validation.
After manuscript edits, rebuild both PDFs, copy both TeX/PDF pairs and the
generated table to the local public package, then rerun both audits.
