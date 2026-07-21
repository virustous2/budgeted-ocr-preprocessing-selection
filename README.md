# Budgeted Page-Level Preprocessing Selection for OCR

This repository contains public code and derived result tables for the manuscript:

**Budgeted Page-Level Preprocessing Selection for Vietnamese Administrative OCR: A Controlled Comparison of Learned, Retrieval-Based, and Static Shortlists**

The original Vietnamese administrative document images and full reference transcriptions are **not included**. They were derived from public government documents, but redistribution permission has not been obtained. The SROIE dataset is also **not redistributed** here; obtain it from the official benchmark distribution.

## What Is Included

- `code/`: scripts for candidate sampling, OCR data collection, surrogate training, shortlist evaluation, nearest-neighbor baselines, bootstrap confidence intervals, ablations, Top1Pred, policy-matched and feature-matched oracle-free selector analyses, and Figure 2 generation.
- `data/candidate_pools/`: fixed 50-candidate preprocessing pools used in the experiments.
- `data/preprocessing_response_tables/`: OCR-free page features, preprocessing parameters, and observed CER values. These tables do not include page images, reference transcriptions, or OCR output text.
- `results/`: nested-CV metrics, per-page best-of-K results, nearest-neighbor baseline outputs, bootstrap summaries, feature-ablation summaries, default-parameter sensitivity and Top1Pred summaries, policy-agnostic headroom controls, sanitized surrogate, policy-matched, and feature-matched selector scores, plus recomputed two-sided statistical tests.
- `paper/`: current LaTeX manuscript and compiled manuscript PDF, Online Resource 1 source and PDF, bibliography, Springer template files, and non-data figures used by the manuscript.
- `environment-experiment.txt`: exact OCR software versions and crop-handling configuration for the reported experiments.

## What Is Excluded

- Raw page images.
- Derived page images.
- Full reference transcriptions.
- OCR output text caches.
- OCR crop images.
- Trained model pickle files.
- Private local paths and temporary run caches.

For the oracle-free selector, the original candidate-score files contained OCR output text. The public copies keep only numeric and derived metadata features; see files named `nested_selector_candidate_scores_sanitized.csv`.

## Reproducing Results

The included result tables are enough to recompute the paper's aggregate CER summaries, paired comparisons, bootstrap intervals, and policy-agnostic headroom controls without redistributing the source documents.

For the current manuscript, use:

```bash
python code/analysis/recompute_two_sided_wilcoxon_tables.py
```

This writes `results/statistical_tests/two_sided_wilcoxon_summary.csv`, the authoritative two-sided paired Wilcoxon summary for the public release. Manuscript-facing p-values should be read from this CSV, not from older experiment logs or regenerated legacy reports.

To reproduce the headroom analysis from the public per-page and NN-K caches, run:

```bash
python code/analysis/compute_oracle_gap_quartiles.py
```

This writes the coupled Surrogate/Random/NN-K summaries, quartile summaries, direct policy-to-policy controls, and headline report under `results/headroom_controls/`. The script verifies that the duplicated Surrogate, Random, and FixedTopK values in the NN-K cache match the authoritative per-page tables before computing the controls.

To recompute the policy-matched and feature-matched top-3 selector comparisons from the sanitized public candidate-score tables, run:

```bash
python code/analysis/compare_policy_matched_top3_selectors.py \
  --feature-matched-raw results/oracle_free_selector/feature_matched/raw/nested_selector_candidate_scores_sanitized.csv \
  --feature-matched-contour results/oracle_free_selector/feature_matched/contour/nested_selector_candidate_scores_sanitized.csv \
  --feature-matched-sroie results/oracle_free_selector/feature_matched/sroie/nested_selector_candidate_scores_sanitized.csv \
  --output-dir /tmp/ijdar_policy_selector_recomputed
```

This reproduces the 28 paired policy-comparison and feature-matched sensitivity rows in `results/oracle_free_selector/policy_comparisons/` without OCR runs. The selector-training scripts are also included, but rerunning them requires source documents and OCR output text that are intentionally excluded from the public release.

To regenerate Figure 2 from the public headroom-control tables, run:

```bash
python code/analysis/plot_oracle_gap_diagnostic.py
```

To rerun OCR data collection from scratch, you need your own local copies of:

- Vietnamese administrative document images and matching reference transcriptions, or an equivalent dataset.
- The SROIE images, ground-truth text files, and optional ground-truth boxes from the official benchmark distribution.
- OCR dependencies listed in `requirements.txt`. Consult `environment-experiment.txt` for the exact package versions and OCR configuration used in the reported experiments.

The default external-data layout is documented in `data/source_documents_not_included/README.md`. All locally generated outputs are written under `results/recomputed/` by default. Scripts that require an exact NoPre OCR table make that input explicit because those OCR outputs are intentionally excluded from the release.

## Suggested Citation

If you use this code or the derived result tables, cite the manuscript once published.

## License

Original source code in `code/` is licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`. The license does not grant rights to source documents, reference transcriptions, SROIE, third-party dependencies, or manuscript material in `paper/`; their scope is described in `NOTICE` and their own terms continue to apply.

The derived result tables are supplied for research transparency. Cite the associated manuscript when using them.
