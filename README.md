# Budgeted Page-Level Preprocessing Selection for OCR

## Current release authority — SROIE612 (CHG-20261019-06)

Use `results/primary_receipt612_20261019_v3/` for the current numerical results
and `results/publication_figures_receipt612_20261019_v2/` for figures. Earlier
616-page bundles and their commands below are historical, not current evidence.
VN300 raw/contour retain frozen-v3 case-sensitive targets; SROIE612 casefolds
both sides. Both members of two further same-receipt pairs were excluded and
all affected training layers rerun on new page-group folds. Old/new differences
include the changed partitions; the image-identity screen does not certify
exhaustive duplicate recall. No OCR was needed for this correction.

Portable replay, requiring NumPy, pandas and SciPy, without OCR or fitting:

```bash
python code/analysis/replay_receipt612_publication.py \
  --bundle results/primary_receipt612_20261019_v3
```

This checks primary224 comparisons/21 Holm families, A384/36 and matched
letter-case80/10, plus192 headroom coefficients. Counts overlap across analyses,
not688 unique hypotheses. All seven A classification changes are retained;
none of64 direct Drop-minus-Full tests survives Holm. No equivalence or
strict-selector feature-removal robustness claim. CIs are paired-page
percentile95%,10000 resamples,per-comparison and conditional on realized fits.

The new bundle contains response/NoPre/fold snapshots, original numeric
per-page outputs,12 text-stripped selector variants, generated tables and
ID/hash provenance. It excludes images, full transcriptions, OCR strings,
caches, model files and visual transaction-content notes. `paper/` is the
current synchronized manuscript/ESM, with signed rounded-zero CI endpoints.
The pipeline and96 VN curve means/rasters are unchanged; headroom uses the
corrected SROIE evidence. Source exporters need private inputs; only the replay
command above is a self-contained public numerical reproduction command.

Local update only: commit/push and online verification remain owner-deferred.
An arithmetic/provenance audit is not proof of scientific validity or acceptance.

## Historical release authority (CHG-20261010-07; superseded)

CHG-20261016-01 updates figure presentation at
`results/publication_figure_clarity_20261016_v2/`. All 96 curve means/rasters and
192 headroom coefficients remain unchanged. Main Fig.2 labels the actual
subtractions; Main Fig.1 identifies offline selector training with labels;
ESM Fig.1 has stacked full-width Raw/Contour rows, separate-scale disclosure
and complete numerical-source references. The bundle includes the historical
vector base, reproducible LaTeX annotation, matching editable draw.io source
and source/output hashes. The old figure bundle remains historical. Numeric
portable v3, fitted policies, statistics and table values are unchanged.

CHG-20261014-07 publishes metadata-only portable bundle v3. Six provenance
JSON files now use paths relative to the original project root, not personal
absolute paths. Those upstream inputs are not necessarily included here.
All numeric CSVs and tables are unchanged; the new manifest records the original
v2 manifest hash and before/after metadata hashes. The original v2 source and
former public copy are retained outside this public repository. Figure/display
provenance continues to identify the original v2 numeric source. The historical
contour report also has only its upstream CSV-path prefix removed.

CHG-20261014-03 adds a separate, post hoc best-of-K feature sensitivity at
`results/sensitivity/noise_feature_20261014_v1/`, reported in Online Resource 1.
The primary 22/12-column results remain unchanged. Removing ONLY `orig_noise`
(21/11 columns; interaction retained) changes 10/160 policy significance
classifications; do not infer equivalence or strict-selector robustness.
All 64 direct and 160 policy comparisons per condition are retained. Replay
the 384 comparisons and 36 Holm families without OCR or model fitting:

```bash
python code/analysis/replay_noise_feature_sensitivity.py \
  --bundle results/sensitivity/noise_feature_20261014_v1
```

The supplement README explains the fixed post hoc protocol, conditional and
unadjusted CIs, all changed comparisons and archived private-layout commands.
No source images/references, OCR text or fitted model files are included.

CHG-20261011-07 repairs only the NN summary helper's exact-zero handling;
current stored helper p-values and primary result CSVs are unchanged. The
environment record now distinguishes imported cv2 from overlapping wheel
metadata; it is not a complete lockfile. Run
`python code/analysis/check_opencv_environment.py` to inspect your interpreter;
add `--require-single-distribution` when validating a prospective clean
environment. No experimental environment was reinstalled and no OCR/refit ran.
New helper/test hashes do not authorize resume into old hash-locked roots.

CHG-20261011-05 clarifies feature redundancy and color NoPre/grayscale candidates
without changing experiment outputs. Use `results/publication_table_format_20261011_v4/`
for current row display: the near-zero raw EasyOCR Sel-Fixed effect is `+0.00`.
Older display overlays below are historical. Primary numerical CSVs are unchanged.

CHG-20261011-04 adds `results/vn_projection_skew_raw_20261011_v1/` (post hoc
raw-page diagnostic, not deskew OCR evaluation). Script/test/protocol are in
`code/analysis/`; source images are still excluded. Primary CER results unchanged.

The current manuscript uses **Vietnamese reference v3 with case-sensitive CER**
and **SROIE casefolded CER**. The SROIE change was post hoc after original and
best-of-K sensitivity results; main/ESM disclose that chronology. Original
case-sensitive SROIE is retained as sensitivity, not current primary evidence.

Use only `results/primary_v3_casefold_20261014_v3/` for current primary numerical results.
CHG-20261011-01 supersedes the display overlay below with
`results/publication_table_format_20261011_v3/`: SROIE table uses separate
K=3/15 rows; model-quality metrics remain in ESM. Numerical CSVs are unchanged.
CHG-20261010-08 adds `results/publication_table_format_20261010_v1/` for display
only: near-zero selector-CI endpoints retain their signs, with explanatory
captions in ESM. Original numeric CSVs and the v2 manifest are unchanged; current
paper row assets use this formatting overlay rather than the older bundle rows.
Its README describes the response/fold tables, original per-page outputs,
12 sanitized selector variants, 224 comparisons, generated tables and hashes.
All other result/data subtrees below are historical unless explicitly cited by
this versioned bundle. In particular the top-level old response tables and
`results/oracle_free_selector/strict/` are NOT current v3/casefold evidence.

From the release root, independently replay all 224 comparisons without OCR or
training:

```bash
python code/analysis/replay_publication_v3.py \
  --bundle results/primary_v3_casefold_20261014_v3
```

Current `paper/` contains synchronized main/ESM TeX/PDF, bibliography, 13 row
tables in `paper/primary_tables/`, and updated figures. Current figure provenance
and plotted means are under the versioned bundle's sibling
`results/publication_figure_clarity_20261016_v2/`. The original
`results/publication_figures_v3_casefold_20261010_v1/` remains historical.
This local release is not yet pushed online. CHG-20261010-09 shortens the main
paper to 20 pages through prose-only edits; numerical evidence is unchanged.
No OCR/model fitting occurred during synchronization or page shortening.

## Historical release description and commands (not current reproduction)

Everything below describes prior checkpoints. In particular phrases such as
"primary unchanged", "canonical strict" or "current manuscript" below refer to
the historical September release, not the promoted authority above. Retained
for traceability; use the versioned reproduction command above now.

This repository contains public code and derived result tables for the manuscript:

**Budgeted Page-Level Preprocessing Selection for Vietnamese Administrative OCR: A Controlled Comparison of Learned, Retrieval-Based, and Static Shortlists**

The original Vietnamese administrative document images and full reference transcriptions are **not included**. They were derived from public government documents, but redistribution permission has not been obtained. The SROIE dataset is also **not redistributed** here; obtain it from the official benchmark distribution.

## What Is Included

- `code/`: scripts for candidate sampling, OCR data collection, surrogate training, shortlist evaluation, nearest-neighbor baselines, bootstrap confidence intervals, ablations, Top1Pred, policy-matched and feature-matched oracle-free selector analyses, and Figure 2 generation.
- `data/candidate_pools/`: fixed 50-candidate preprocessing pools used in the experiments.
- `data/cohort_manifests/`: final Vietnamese and SROIE cohort metadata, identifiers, exclusion records, and content hashes; source documents remain excluded.
- `data/preprocessing_response_tables/`: OCR-free page features, preprocessing parameters, and observed CER values. These tables do not include page images, reference transcriptions, or OCR output text.
- `results/`: nested-CV metrics, per-page best-of-K results, nearest-neighbor baseline outputs, bootstrap summaries, policy-agnostic headroom controls, and sanitized strict nested selector scores and comparisons. Historical feature-ablation and default-parameter holdout outputs remain for traceability but predate the corrected NFC response labels and are not current manuscript evidence. The canonical selector artifacts are under `results/oracle_free_selector/strict/`.
- `paper/`: current LaTeX manuscript and compiled manuscript PDF, Online Resource 1 source and PDF, bibliography, Springer template files, and non-data figures used by the manuscript.
- `environment-experiment.txt`: retained OCR software metadata, crop-handling configuration, and explicit OpenCV runtime/metadata provenance limits; not a complete lockfile.

The final Vietnamese cohort contains 300 first pages (30 per year, 2015--2024). After an exact-content audit, two duplicated selections were replaced within their 2018 and 2024 strata, and every Vietnamese manuscript-facing result was recomputed from the corrected cohort.

## What Is Excluded

- Raw page images.
- Derived page images.
- Full reference transcriptions.
- OCR output text caches.
- OCR crop images.
- Trained model pickle files.
- Private local paths and temporary run caches.

For the oracle-free selector, the original candidate-score files contained OCR output text. The public copies keep only numeric and derived metadata features; see files named `nested_selector_candidate_scores_sanitized.csv`. The `strict/` subtree supersedes the earlier pooled-selector artifacts retained outside it for historical traceability.

## SROIE casefold sensitivity (post hoc)

`results/sensitivity/sroie_casefold_20260926_v3/` contains paired primary/alternative
best-of-K results, two-sided/Holm tables and per-comparison bootstrap CIs. Its
README gives the no-OCR/no-fitting reproduction command. Primary results are
unchanged. NN-K EasyOCR K=3/5 advantages over Surrogate lose significance under
casefold; Surrogate Tesseract significance extends to K=5. This does not evaluate
casefolded one-output selectors or Vietnamese results.

## Reproducing Results

The included result tables are enough to recompute the paper's aggregate CER summaries, paired comparisons, bootstrap intervals, and policy-agnostic headroom controls without redistributing the source documents.

For the current manuscript, use:

```bash
python code/analysis/recompute_two_sided_wilcoxon_tables.py
```

This writes `results/statistical_tests/two_sided_wilcoxon_summary.csv`, the authoritative two-sided paired Wilcoxon summary for the public release. It uses SciPy `zero_method="wilcox"` (zero paired differences excluded from signed ranks) and includes raw p-values, explicit Holm-family identifiers, Holm-adjusted p-values, and adjusted-significance indicators. Manuscript-facing p-values should be read from this CSV, not from older experiment logs or regenerated legacy reports.

To reproduce the raw-Vietnamese Surrogate-versus-FixedTopK zero-method sensitivity without changing the primary table, run:

```bash
python code/analysis/recompute_wilcoxon_zero_sensitivity.py
```

This writes 36 rows (12 engine/budget comparisons for each of `wilcox`, `pratt`, and `zsplit`) to `results/statistical_tests/wilcoxon_zero_sensitivity_raw_fixedtopk_20260924.csv`.

To rebuild the percentile paired-bootstrap summaries from the same current per-page tables, without OCR or model fitting, run:

```bash
python code/analysis/rebuild_public_bootstrap_from_perpage.py
```

To reproduce the headroom analysis from the public per-page and NN-K caches, run:

```bash
python code/analysis/compute_oracle_gap_quartiles.py
```

This writes the coupled Surrogate/Random/NN-K summaries, quartile summaries, direct policy-to-policy controls, Holm-adjusted correlation p-values, and headline report under `results/headroom_controls/`. The script verifies that the duplicated Surrogate, Random, and FixedTopK values in the NN-K cache match the authoritative per-page tables before computing the controls.

The canonical strict selector result tree contains full Surrogate, rank-only Surrogate, NN-K and FixedTopK candidate-score tables for raw Vietnamese, contour-suppressed Vietnamese and SROIE, plus the two audited 32-row paired-comparison tables:

- `results/oracle_free_selector/strict/comparisons/strict_selector_policy_paired_comparison.csv`: one 16-test primary family and one separate 16-test rank-only sensitivity family.
- `results/oracle_free_selector/strict/comparisons/strict_selector_static_paired_comparison.csv`: four separate eight-test families for Top1Pred, agreement-only, FixedBest and NoPre.

These comparisons require no OCR rerun. The strict selector-training scripts and experiment configurations are included under `code/analysis/` and `code/config/strict_selector/`; rerunning training requires source documents and OCR output text that are intentionally excluded from the release.

To recompute both strict comparison tables from the sanitized release files:

```bash
python code/analysis/recompute_strict_selector_public_tables.py \
  --output-dir /tmp/ijdar_strict_selector_recomputed \
  --bootstrap-replicates 10000 \
  --seed 42
```

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
