# Display-only overlay — CHG-20261010-08

Current paper rows use these assets. Numeric evidence remains the v2 publication
bundle and completed original outputs. Only full/rank-only selector-CI rendering
changes: positive/negative nonzero endpoints retain + or - even when rounded
to 0.00. Exact zero remains 0.00. Differences and p-values are unchanged.

ESM captions explain per-comparison, unadjusted intervals and that significance
uses Holm-adjusted Wilcoxon tests. SROIE Tesseract versus NN-K (full) has upper
endpoint +0.004694 pp, so the interval includes zero; its display is now +0.00.
This is not a new experiment, changed metric, seed or CI calculation.

Renderer: `analysis/export_publication_table_format.py`; hashes in manifest.json.
Original bundle/table snapshots remain historical. No remote push occurred.
