# Strict nested NN-K / FixedTopK selector controls

Implementation date: 2026-09-12. Status as of 2026-09-23: raw v2, contour
and SROIE strict policy controls and paired comparisons are complete.

Runner: `analysis/evaluate_policy_selectors_strict_nested.py`.
Configs: raw uses `analysis/strict_policy_raw_20260912_v2.json`; contour and
SROIE use `analysis/strict_policy_{contour,sroie}_20260912.json`.
Tests: `analysis/test_strict_policy_selectors.py` (synthetic/mocked only).

This runner supplements, without modifying, the completed strict surrogate
selector runs. It reuses their fold-exclusion and selector-family selection
functions from `evaluate_top3_selector_strict_nested.py`.

## Isolation and policy definitions

For outer test fold A, its shortlist is built with A excluded. Selector training
pages in fold B use upstream policies trained without A and B. During selector
family selection, validation fold C uses policies trained without A and C;
selector-training fold B then uses policies trained without A, C and B.
Thus each engine needs 25 exclusion sets (5 singletons, 10 pairs, 10 triples),
for each of the two policies. Pages, not candidate rows, define the folds.

NN-K follows the existing `evaluate_nn_shortlist_baseline.py` definition:
standardize page features on allowed training pages, order ALL allowed neighbors
by Euclidean distance, transfer each neighbor's training-label-best configuration,
retain unique configurations in order, and fall back to training FixedTopK if
needed. Keep the first three. The old output's first 15 neighbor records are not
sufficient to reconstruct this in every case. Candidate provenance now records
the source neighbor, distance and neighbor rank, or the training-fixed fallback.

FixedTopK uses the three lowest mean-CER configurations on allowed training pages.
FixedBest is independently derived from the allowed training means. No held-out
CER is used to choose a shortlist; held-out CER is attached afterward for targets
and evaluation. Parameter keys, feature definitions, page ordering, nearest-
neighbor ordering and training-mean sorting follow the original baseline.

Both policy controls use shortlist ranks 1/2/3 in place of predicted CER, as in
the previous rank-based controls. They use the same 36-feature selector builder,
five selector families, four inner validation folds per outer fold, and seed 42
as the strict surrogate comparison. There is no duplicate rank-only variant:
these controls already use ranks. The surrogate rank-only result is the matched
feature-input sensitivity comparison, not a claim that full surrogate CER inputs
and policy ranks are identical information.

## Stages, outputs and interruption

- `plan`: read-only validation and job counts; no output folder creation.
- `prepare`: builds both policies, fits only the NN feature scaler, saves jobs
  with input/code hashes, source page IDs and scaler statistics. No surrogate
  or selector estimator fitting, and no OCR. Completed jobs are reused.
- `cache`: initializes from the completed strict surrogate OCR cache; runs OCR
  only for required page/config/engine entries missing valid text/crop metadata.
  New results go to the new output folder's shared `ocr_cache.csv`. The seed
  cache is never overwritten. Atomic saves every 25 missing keys and on exit;
  an interrupted in-flight OCR call may need repeating.
- `evaluate`: builds selector features from cached OCR and performs nested
  selector fitting. Completed policy/engine/outer-fold checkpoints are reused;
  there is no checkpoint inside an estimator fit. Outputs are saved separately
  under `nn_k/` and `fixed_topk/`.

Output roots are `outputs/strict_nested_policy_controls_raw_cohort300_v2_20260912_v2`,
`outputs_contour/strict_nested_policy_controls_contour_cohort300_v2_20260912`, and
`SROIRE_COMPARISON/output_sroire/strict_nested_policy_controls_sroie_all616_20260912`.
Each contains `input_manifest.json`, `fold_assignments.csv`, a shared OCR cache,
and policy-specific `policy_jobs/`, `folds/`, candidate scores, inner validation,
summary CSV/JSON, report, and completion markers.

Do not change hashed code, inputs, seed caches or environments between stages.
Use a new output directory if they change. The runner refuses mismatched hashes.
The user runs the experimental stages; synthetic tests and read-only audits are
not experimental results.

Cache validity additionally requires every requested engine to have an integer
`n_chars_<engine>` equal to the normalized length of `text_<engine>`. This is
necessary because a rectangular CSV represents an engine that was not run on a
key as empty cells; after loading, its text cell alone is indistinguishable from
a genuine empty OCR string. A genuine empty result remains valid when its saved
character count is zero. The first raw cache attempt exposed 1,007 such legacy
missing-engine cells after adding policy requirements; it must not be evaluated.
Raw therefore continues with `strict_policy_raw_20260912_v2.json`, whose new
output directory seeds from that attempt and reuses all actually complete OCR.

## Verification and submission status

Before experimental handoff, a read-only reconstruction of singleton-fold
shortlists was compared with saved `nn_per_image.csv` in the existing paired
NFC/exact-Random runs. NN-K and FixedTopK best-of-3 CER matched exactly, with
the same fold assignments, for all 300 pages in each Vietnamese setting/engine
and all 616 SROIE pages/engine (6,064 policy/page/engine comparisons total).
This checks baseline compatibility, not learned-selector performance.

Summary-helper one-sided p-values are NOT the final two-sided/Holm submission
analysis. The completed strict SROIE NN-K control restores two SROIE engine
comparisons that the historical incomplete neighbor output could not support.
`analysis/compare_strict_policy_selectors.py` fixes the new comparison sets:
eight setting/engine blocks times two policy comparators (NN-K and FixedTopK),
for 16 tests in the primary full-surrogate-selector family; the same 16 tests
form a separate rank-only sensitivity family. Each family uses two-sided paired
Wilcoxon tests with Holm adjustment across its 16 rows. Bootstrap percentile
intervals describe the mean paired CER difference; they are pointwise and not
Holm-adjusted. These definitions precede
the new script's p-value computation. The script validates all 12 completed
score sources and exact full/rank-only shortlist identity before calculation.
The paired comparisons and manuscript/public-release artifacts have since been
synchronized. The post-edit read-only artifact audit reported 36,670 passed
checks and zero failures; it does not establish statistical or external validity.
