# Oracle-Free Selector Results

This directory contains public, sanitized outputs for the deployable top-3 selector analyses reported in the manuscript. The source documents, reference transcriptions, OCR crops, OCR caches, and OCR output text are not distributed. Every `nested_selector_candidate_scores_sanitized.csv` file omits the original OCR transcript field while retaining the numeric and derived features needed for aggregate checks.

- `raw/`, `contour/`, and `sroie/` contain the original Surrogate top-3 selector outputs.
- `policy_matched/` contains the same nested selector protocol applied separately to NN-K and FixedTopK top-3 shortlists. SROIE NN-K is omitted because the historical shortlist cannot be reconstructed exactly for three EasyOCR pages.
- `feature_matched/` contains the Surrogate sensitivity analysis where continuous prediction and margin inputs are replaced by within-shortlist ranks 1, 2, and 3.
- `policy_comparisons/` contains the canonical 28-row paired comparison table, including two-sided Wilcoxon tests, bootstrap confidence intervals, effective sample sizes after zero differences, and Holm/BH adjustments.

Run `code/analysis/compare_policy_matched_top3_selectors.py` with the three feature-matched candidate-score files to recompute the canonical comparison table from the sanitized release artifacts. Rerunning selector training requires source documents and OCR output text, which are deliberately excluded.
