# Strict nested selector correction — 2026-09-06

Status as of 2026-09-23: strict raw, contour and SROIE surrogate-selector
experiments are complete. Strict NN-K and FixedTopK selector controls are also
complete; the resulting two-sided/Holm comparisons are synchronized with the
manuscript and public release. Earlier pooled-OOF learned-selector results
remain historical artifacts, not submission evidence. The artifact audit
checks internal consistency and provenance, not methodological or external
validity. Keep old outputs for traceability.

## Defect and required separation

The old Top3 script fits a surrogate outside each page's own outer fold. The
old selector then cross-validates on the single pooled table of these outputs.
When the selector is tested on fold A, its training candidates in fold B came
from a surrogate that included A. This is an indirect test-label dependency.
It does not imply that independent best-of-K or agreement-only evaluation has
the same defect; those do not fit a second supervised learner on pooled OOF rows.

The new `evaluate_top3_selector_strict_nested.py` uses the following scopes:

| Purpose, outer test A | Predicted page fold | Surrogate excludes |
|---|---|---|
| Final selector test | A | A |
| Final selector training | B, B != A | A and B |
| Selector family validation on C | C | A and C |
| Training for that validation | B, B != A,C | A, C and B |

Every surrogate chooses its family by the existing five-fold page-grouped MAE
CV restricted to its own allowed training subset, then refits on that subset.
Selector families and their hyperparameters remain unchanged; family choice
uses mean inner-fold selected-output CER, ties resolved in fixed family order.
Five original outer folds are preserved. No outer test CER is passed to the
scoring function. Inner validation CER is used only for family choice.

This is an implementation of the separation principle in nested CV, not a claim
that scikit-learn supplies this custom OCR pipeline:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html

## Computation and outputs

There are 5 + 10 + 10 = 25 distinct excluded-fold subsets per recognizer. Their
predictions can be reused across contexts with exactly the same training set.
Each new job uses 20 model-family CV fits plus one final fit. Train subsets have
80%, 60%, or 40% of pages (approximately for SROIE). No new hyperparameters are
selected based on the corrected outer test results.

The `prepare` stage refits the surrogate jobs, not OCR. `cache` reuses the supplied
OCR-text cache and collects only missing page/configuration/recognizer outputs.
`evaluate` trains full-feature and rank-only selectors on the same regenerated
shortlists. Prediction ties keep input candidate order. RF prediction is serial
to avoid threaded prediction-summation noise; training model definitions are
unchanged. New outer candidate identities must still be compared with prior
outputs, not assumed identical.

Run from project root, replacing `raw` by `contour` or `sroie` as needed:

```bash
./venv_py310/bin/python analysis/evaluate_top3_selector_strict_nested.py --config analysis/strict_selector_raw_20260906.json --stage plan
./venv_py310/bin/python analysis/evaluate_top3_selector_strict_nested.py --config analysis/strict_selector_raw_20260906.json --stage prepare
CUDA_VISIBLE_DEVICES=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True ./venv_py310/bin/python analysis/evaluate_top3_selector_strict_nested.py --config analysis/strict_selector_raw_20260906.json --stage cache
./venv_py310/bin/python analysis/evaluate_top3_selector_strict_nested.py --config analysis/strict_selector_raw_20260906.json --stage evaluate
```

`plan` is read-only. Other stages write only to the configured new directory.
For the first real execution, append `--max-new-jobs 1` to `--stage prepare`.
This intentionally pauses after one completed job, allowing an artifact check
before the remaining fits. Repeat without the limit to resume and finish.
Completed surrogate jobs and selector outer folds are checkpointed with hashes;
Ctrl+C loses the active unfinished fit, not completed jobs. OCR cache saves every
25 missing entries and on exit from the OCR loop. No mid-tree or mid-family-CV
resume is claimed. Do not run two writers on one output directory concurrently.
Changing tracked code, inputs, image bytes, seed cache, or recorded ML package
versions requires a new output directory. Original OCR caches are reused as
existing project evidence; this does not independently certify their model/
weight provenance or prove cached OCR text matches newly executed OCR.

Within the new output directory:

- `input_manifest.json`, `fold_assignments.csv`: frozen sources and page folds.
- `surrogate_jobs/<engine>_exclude_<folds>/`: predictions, baselines, explicit
  training/predicted page provenance and hash-checked completion marker.
- `ocr_cache.csv`: separate mutable cache; original seed cache is not overwritten.
- `full/`, `rank_only/`: per-fold checkpoints and merged selector scores,
  summaries and family-validation scores, with completion markers.

The helper summary still exposes historical one-sided p-values. They are NOT
the manuscript's authoritative two-sided/Holm results. For submission
comparisons use the strict, recomputed paired tables under
`ijdar_public_release/results/oracle_free_selector/strict/comparisons/`.

## Completed correction and remaining scope

Strict surrogate-selector runs on raw/contour/SROIE, strict NN-K/FixedTopK
policy controls, final paired comparisons and the manuscript/public-release
sync have completed. The post-edit read-only artifact audit reported 36,670
passed checks and zero failures. This is an internal consistency/provenance result,
not proof of statistical validity or generalization. Overall submission
readiness still requires normal manuscript and venue checks.

The legacy full-feature, feature-matched and policy-matched selector CLIs now
refuse to run by default. `--allow-legacy-pooled-oof` is an explicit opt-in for
historical reproduction only, not an alternate corrected submission protocol.
The public-release CLI copies have the same guard. The strict runners and
portable configurations are included in the public release; source images,
full references and OCR-text caches are intentionally excluded.
