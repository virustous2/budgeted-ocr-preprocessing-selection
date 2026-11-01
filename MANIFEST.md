# Public Release Manifest

## Current authority — CHG-20261019-06

- `results/primary_receipt612_20261019_v3/`: VN300/300 and corrected SROIE612
  response/NoPre/folds, per-page best-of-K/NN,12 text-free selector variants,
  primary statistics224/21 families, A384/36 and matched metric80/10. Generated
  tables, relative-path/source hashes, source-verification digest and limited
  same-receipt screening provenance are retained. Some comparisons overlap.
- `results/publication_figures_receipt612_20261019_v2/`: corrected192 headroom
  coefficients and regenerated panels; unchanged96 VN curve means/rasters and
  labeled offline-supervised pipeline with vector source/provenance.
- `code/analysis/replay_receipt612_publication.py`: portable arithmetic replay
  using the two included independent replay helper modules. No OCR/fitting.
- `paper/`: synchronized TeX/PDF,13 generated row tables, matched metric and
  A tables. Full-precision numeric CSVs remain in the bundle above.
- `code/analysis/publication_authority_receipt612.json`: private project-relative
  path registry; public users follow the README replay command instead.

Both members of two additional same-receipt pairs were excluded; all affected
SROIE training layers were rerun. VN inputs remain unchanged but shared Holm
adjustment can change a VN adjusted p-value. Old616 releases below are retained
historically, never mixed with612. No data images, references, OCR strings,
models or personal absolute paths in the new numeric bundle. No remote push.

## Historical authority — CHG-20261010-07 (superseded)

CHG-20261016-01: figure-only clarification at
`results/publication_figure_clarity_20261016_v2/`. The 96 curve means and PNG
rasters, and all 192 headroom correlations, retain their original bytes.
Updated headroom labels state subtraction direction; the vector pipeline adds
offline supervised selector training. Recovered draw.io and LaTeX overlay
sources accompany the retained vector base and hashed payload manifest.
`code/analysis/export_figure_clarity.py` prepares a fresh version from explicit
source paths. ESM Fig.1 uses two labeled full-width rows and a corrected caption.
Original numeric, table and experimental manifests are unchanged; no remote push.

CHG-20261014-07: current public bundle is portable metadata v3. Only six
provenance JSON files lose the original machine-specific project-root prefix;
all numeric CSVs/tables/folds remain byte-identical. Its manifest records the
source v2 manifest and original/released metadata hashes. Original v2 artifacts
are archived outside the public repository, not silently overwritten. The
historical contour report's CSV path is likewise relative; numbers unchanged.

- `code/analysis/sanitize_publication_paths.py`: archived metadata-only release
  export, requiring the original source bundle and explicit project root.
- `code/analysis/audit_portable_release.py`, `test_portable_release.py`: independent
  byte checks and synthetic export guards; source-to-release audit requires the
  retained source bundle. Portable numerical replay below needs only this release.

CHG-20261014-03: supplementary (not primary) standalone-feature A:
`results/sensitivity/noise_feature_20261014_v1/` includes paired numeric values,
64 direct comparisons, 160 policy comparisons per condition, all changed Holm
classifications, Full reproduction diagnostics, protocol and verification.
`code/analysis/replay_noise_feature_sensitivity.py` independently replays the
384 arithmetic comparisons/36 families using the numeric bundle alone.
`code/analysis/audit_noise_feature_statistics.py` supplies independent arithmetic
primitives; its private-job audit CLI additionally needs the original layout.
Training runner/protocol/config/tests are archived in `code/analysis/`; their
private project paths are not public fitting instructions. ESM source/PDF and
`paper/noise_feature_tables.tex` are synchronized. Primary numeric bundle and
strict selectors remain unchanged; no remote push.

CHG-20261011-07: exact-zero NN helper repair and current-runtime OpenCV
diagnostic/tests under `code/analysis/`. Primary numerical bundle unchanged.
`environment-experiment.txt` distinguishes runtime 4.10.0 from overlapping
wheel metadata. Historical experimental manifests are unchanged; future
reruns need fresh output roots when helper hashes change. No remote push.

CHG-20261011-05: current display overlay is
`results/publication_table_format_20261011_v4/`; nonzero near-zero Sel-Fixed
effects retain their sign. Numerical bundle unchanged. Manuscript descriptions
now identify Laplacian redundancy and color NoPre versus grayscale candidates.

CHG-20261011-04: post hoc projection-profile raw-page diagnostic under
`results/vn_projection_skew_raw_20261011_v1/`; no image/text data, no primary CER
changes. Main/ESM explicitly distinguish scope from unmeasured deskew benefits.

CHG-20261011-01 supersedes the display supplement below with
`results/publication_table_format_20261011_v3/`. SROIE CER uses nine columns
and two budget rows per engine; model quality stays in ESM. Three .orig backups
were moved outside this public repository; *.orig and *.rej are now ignored.

CHG-20261010-08 display-only supplement:
`results/publication_table_format_20261010_v1/` contains current row formatting
with signed near-zero CI endpoints and hashes. Original v2 numeric data are
unchanged; `paper/primary_tables/` uses the new display overlay. No inference rerun.

- `results/primary_v3_casefold_20261014_v3/`: current response/NoPre/folds,
  per-page best-of-K, NN-K, 12 text-free selector variants, statistics and tables.
  `manifest.json` records payload/input hashes. Its cohort manifest hashes the
  evaluated VN-v3 labels, not original recovery labels.
- `results/publication_figure_clarity_20261016_v2/`: current figure assets,
  unchanged plotted means/correlations, vector pipeline sources and provenance.
  `results/publication_figures_v3_casefold_20261010_v1/` is historical.
- `code/analysis/replay_publication_v3.py`: portable no-OCR/no-fit independent
  numerical replay of 224 comparisons and 21 Holm families.
- `paper/{main,ESM_1}.{tex,pdf}`, `paper/primary_tables/`, updated cited figures
  and bibliography: synchronized current publication.

All older result and response-table paths listed below are historical and are
superseded for primary publication. SROIE casefold is now primary; historical
case-sensitive evaluation remains sensitivity. No remote push has occurred.

## Historical package inventory below

## License

- `LICENSE`: Apache License, Version 2.0 for original source code in `code/`.
- `NOTICE`: copyright, license scope, and exclusions for data, dependencies, and manuscript material.

## Environment

- `environment-experiment.txt`: software metadata/crop configuration, OpenCV runtime distinction and historical-provenance limits (not a complete lockfile).

## Code

- `code/analysis/export_sroie_casefold_sensitivity.py`: no-OCR/no-fit primary-versus-casefold export with eight-test Holm families.
- `results/sensitivity/sroie_casefold_20260926_v3/`: seven input CSV snapshots, 80 statistical rows, generated ESM tables and reproduction instructions. Post hoc best-of-K only; primary results unchanged.
- `paper/sroie_casefold_tables.tex`: generated table included by ESM_1.tex.

- `code/main/preprocessing_sampling.py`: fixed preprocessing candidate sampling.
- `code/main/apply_contour_removal.py`: contour-suppression preprocessing.
- `code/main/collect_multiengine_data.py`: Vietnamese multiengine OCR data collection.
- `code/main/train_surrogate_multiengine_kfold.py`: nested GroupKFold surrogate training and shortlist evaluation.
- `code/main/inference_surrogate.py`: inference helper for saved surrogate models.
- `code/main/run_default_multiengine.py`: Vietnamese NoPre data collection.
- `code/main/train_surrogate_multiengine_default.py`: default-parameter sensitivity evaluation.
- `code/main/analyze_shortlist_heterogeneity.py`: shortlist heterogeneity analysis.
- `code/analysis/*.py`: statistical summaries, NN baseline, feature ablation, Top1Pred, strict nested oracle-free top-3 selectors, policy-matched controls, paired comparisons, Figure 2 generation, and supporting checks.
- `code/config/strict_selector/*`: strict selector configurations and protocol documents.
- `code/sroie_scripts/*.py`: SROIE-specific data collection and checks.

## Data Tables

- `data/candidate_pools/*.csv`: 50 preprocessing configurations.
- `data/cohort_manifests/vietnamese_cohort_manifest.csv`: final 300-page Vietnamese cohort identifiers and image/reference SHA-256 hashes.
- `data/cohort_manifests/vietnamese_cohort_metadata.json`: balanced year counts and the two audit replacements.
- `data/cohort_manifests/sroie_cohort_metadata.json`: final 616-receipt SROIE cohort rule and exclusion-manifest provenance.
- `data/preprocessing_response_tables/*_preprocessing_data.csv`: page features, candidate parameters, and CER targets.
- `data/source_documents_not_included/README.md`: required local layout for source documents that are intentionally excluded.

These tables intentionally do not include raw images, reference text, OCR output text, or source PDFs.

## Result Tables

- `results/main_metrics/*_metrics.json`: aggregate nested-CV metrics.
- `results/per_page_bestk/*/per_page_results_*.csv`: held-out page-level best-of-K values.
- `results/nn_baseline/*`: nearest-neighbor baseline metrics and per-image outputs.
- `results/oracle_free_selector/strict/full/*`: canonical sanitized strict Surrogate selector outputs.
- `results/oracle_free_selector/strict/rank_only/*`: canonical sanitized strict rank-only Surrogate sensitivity outputs.
- `results/oracle_free_selector/strict/policy_matched/*`: canonical sanitized strict NN-K and FixedTopK selector controls, including SROIE NN-K.
- `results/oracle_free_selector/strict/comparisons/*`: audited 32-row primary/rank-only policy table and 32-row static-baseline table with run metadata.
- `results/oracle_free_selector/{raw,contour,sroie,policy_matched,feature_matched,policy_comparisons}/*`: earlier pooled-selector artifacts retained only for historical traceability and superseded by `strict/`.
- `results/statistical_tests/two_sided_wilcoxon_summary.csv`: recomputed two-sided paired Wilcoxon tests with explicit Holm families and adjusted p-values from public per-page outputs.
- `results/bootstrap_ci/*`: percentile bootstrap summaries.
- `results/feature_ablation/*`: historical pre-NFC feature-ablation metric summaries; not current manuscript evidence.
- `results/default_parameter_sensitivity/*`: historical pre-NFC raw and contour default-regressor sensitivity metrics and reports; not current manuscript evidence; trained model files are excluded.
- `results/top1pred/*`: raw, contour, and SROIE Top1Pred metrics, reports, held-out per-page values, and selected preprocessing parameters.
- `results/headroom_controls/*`: authoritative coupled Surrogate/Random/NN-K headroom summaries, quartiles, direct policy-to-policy controls, Holm-adjusted correlation p-values, report, and headline JSON.
- `results/preprocessing_benefit/*`: corrected-cohort within-page candidate variation and sampled-pool-oracle benefit summaries.
- `results/contour_robustness/*`: exploratory VietOCR-only 30-page contour-threshold summary and CSV from before the reference-NFC correction, reported in ESM Section17/Table17. Thresholds were varied one at a time; only the upper aspect-ratio bound was varied. Mean-CER changes are descriptive, not significance tests or a post-NFC robustness rerun.
- `results/oracle_gap/*`: legacy surrogate-only FixedTopK-to-oracle summaries retained for traceability; these are superseded by `results/headroom_controls/*`.
- `results/statistical_tests/wilcoxon_zero_sensitivity_raw_fixedtopk_20260924.csv`: optional post hoc raw-Vietnamese Surrogate--FixedTopK sensitivity to Wilcoxon zero handling, regenerated by the included public script; it does not replace the primary `wilcox` table.

## Paper

- `paper/main.tex`, `paper/main.pdf`, `paper/ESM_1.tex`, `paper/ESM_1.pdf`, `paper/reference.bib`, `paper/sn-jnl.cls`, `paper/sn-basic.bst`.
- `paper/figures/*`: result plots and schematic figures used by the manuscript.

LaTeX build artifacts such as `.aux`, `.log`, `.fls`, and `.fdb_latexmk` are excluded.
