# Oracle-Free Selector Results

This directory contains public, sanitized outputs for the oracle-free top-3 output-selection analyses. The source documents, reference transcriptions, OCR crops, OCR caches, and OCR output text are not distributed. Every `nested_selector_candidate_scores_sanitized.csv` file omits the original OCR transcript field while retaining numeric and derived features needed for aggregate checks.

- `strict/` is the canonical manuscript-facing result tree. It contains fully nested full Surrogate, rank-only Surrogate, NN-K and FixedTopK outputs for all three settings, including reconstructed strict SROIE NN-K controls.
- `strict/comparisons/strict_selector_policy_paired_comparison.csv` contains the 16-test primary family and separate 16-test rank-only sensitivity family.
- `strict/comparisons/strict_selector_static_paired_comparison.csv` contains four separate eight-test static-comparator families.
- `raw/`, `contour/`, `sroie/`, `policy_matched/`, `feature_matched/`, and `policy_comparisons/` outside `strict/` are historical pooled-selector artifacts. They are retained for traceability and must not be used as final selector evidence.

Each strict variant includes release metadata tying the sanitized table to the completed source artifact by SHA-256. Rerunning selector training requires source documents and OCR output text, which are deliberately excluded.

Run `code/analysis/recompute_strict_selector_public_tables.py` to reproduce both canonical comparison CSVs directly from this sanitized tree without OCR or model fitting.
