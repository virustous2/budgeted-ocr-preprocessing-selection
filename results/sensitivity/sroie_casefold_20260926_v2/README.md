# SROIE casefold sensitivity: best-of-K only

This post hoc analysis does not replace the primary case-sensitive results.
Inputs contain page-level CER/fold/policy information, not OCR text or documents.
All code/data paths below are relative to the public release root.

Reproduce into a fresh directory (no OCR, labels, training, or private repo):

```bash
python code/analysis/export_sroie_casefold_sensitivity.py \
  --inputs results/sensitivity/sroie_casefold_20260926_v2/inputs \
  --output-dir /tmp/sroie_casefold_reproduced/tables
```

Compare the produced files with `tables/`; they are deterministic under the
package versions recorded in `tables/manifest.json`. The exporter refuses to
overwrite differing files. Use a new output directory after code/input changes.

Both Surrogate and NN-K retain 8/8 Holm-significant advantages over Random and
FixedTopK. NN-K's primary EasyOCR K=3/5 advantages over Surrogate are not
significant after casefolding, despite still lower NN-K means; Surrogate's
Tesseract advantage extends from K=3 to K=5. This is not a formal test of a
metric-by-policy interaction or proof of equivalence for nonsignificant rows.
No claim about casefolded one-output selectors or Vietnamese results is made.

Each metric/policy contrast is a separate eight-test two-sided Wilcoxon/wilcox
Holm family. Bootstrap CIs are paired-page percentile intervals (10,000 draws),
95% per comparison, not multiplicity-adjusted. Full-precision endpoints are in
the CSVs; TeX uses three significant digits to preserve near-zero bounds.
See PROTOCOL.md and the manifest for seeds, source mapping and limitations.
