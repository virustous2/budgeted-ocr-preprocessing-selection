# Public Release Manifest

## License

- `LICENSE`: Apache License, Version 2.0 for original source code in `code/`.
- `NOTICE`: copyright, license scope, and exclusions for data, dependencies, and manuscript material.

## Environment

- `environment-experiment.txt`: exact OCR software versions and crop-handling configuration for the reported experiments.

## Code

- `code/main/preprocessing_sampling.py`: fixed preprocessing candidate sampling.
- `code/main/apply_contour_removal.py`: contour-suppression preprocessing.
- `code/main/collect_multiengine_data.py`: Vietnamese multiengine OCR data collection.
- `code/main/train_surrogate_multiengine_kfold.py`: nested GroupKFold surrogate training and shortlist evaluation.
- `code/main/inference_surrogate.py`: inference helper for saved surrogate models.
- `code/main/run_default_multiengine.py`: Vietnamese NoPre data collection.
- `code/main/train_surrogate_multiengine_default.py`: default-parameter sensitivity evaluation.
- `code/main/analyze_shortlist_heterogeneity.py`: shortlist heterogeneity analysis.
- `code/analysis/*.py`: statistical summaries, NN baseline, feature ablation, Top1Pred, oracle-free top-3 selectors, policy-matched and feature-matched selector controls, policy comparisons, Figure 2 generation, and supporting checks.
- `code/sroie_scripts/*.py`: SROIE-specific data collection and checks.

## Data Tables

- `data/candidate_pools/*.csv`: 50 preprocessing configurations.
- `data/preprocessing_response_tables/*_preprocessing_data.csv`: page features, candidate parameters, and CER targets.

These tables intentionally do not include raw images, reference text, OCR output text, or source PDFs.

## Result Tables

- `results/main_metrics/*_metrics.json`: aggregate nested-CV metrics.
- `results/per_page_bestk/*/per_page_results_*.csv`: held-out page-level best-of-K values.
- `results/nn_baseline/*`: nearest-neighbor baseline metrics and per-image outputs.
- `results/oracle_free_selector/{raw,contour,sroie}/*`: sanitized surrogate top-3 selector candidate-score tables.
- `results/oracle_free_selector/policy_matched/*`: sanitized NN-K and FixedTopK top-3 selector controls, including their policy-specific candidate lists and nested selector summaries.
- `results/oracle_free_selector/feature_matched/*`: sanitized feature-matched Surrogate selector sensitivity outputs.
- `results/oracle_free_selector/policy_comparisons/*`: paired policy and feature-matched comparison table, report, and release metadata.
- `results/statistical_tests/two_sided_wilcoxon_summary.csv`: recomputed two-sided paired Wilcoxon tests from public per-page outputs.
- `results/bootstrap_ci/*`: percentile bootstrap summaries.
- `results/feature_ablation/*`: feature-ablation metric summaries.
- `results/default_parameter_sensitivity/*`: raw and contour supplementary default-regressor sensitivity metrics and reports; trained model files are excluded.
- `results/top1pred/*`: raw, contour, and SROIE Top1Pred metrics, reports, held-out per-page values, and selected preprocessing parameters.
- `results/headroom_controls/*`: authoritative coupled Surrogate/Random/NN-K headroom summaries, quartiles, direct policy-to-policy controls, report, and headline JSON.
- `results/oracle_gap/*`: legacy surrogate-only FixedTopK-to-oracle summaries retained for traceability; these are superseded by `results/headroom_controls/*`.
- `results/statistical_tests/two_sided_wilcoxon_summary.csv`: recomputed two-sided paired Wilcoxon tests from public per-page outputs.

## Paper

- `paper/main.tex`, `paper/main.pdf`, `paper/ESM_1.tex`, `paper/ESM_1.pdf`, `paper/reference.bib`, `paper/sn-jnl.cls`, `paper/sn-basic.bst`.
- `paper/figures/*`: result plots and schematic figures used by the manuscript.

LaTeX build artifacts such as `.aux`, `.log`, `.fls`, and `.fdb_latexmk` are excluded.
