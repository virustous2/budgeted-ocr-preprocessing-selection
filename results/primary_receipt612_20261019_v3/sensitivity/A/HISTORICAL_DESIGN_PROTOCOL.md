# A: remove the standalone Laplacian-standard-deviation column

CHG-20261012-01. Protocol fixed before any real ablation fits. This is a post hoc
supplementary best-of-K analysis, not replacement of the primary results and not
a preregistration predating the original experiments.

## Question and scope

Does removing `orig_noise` change the evaluated best-of-K policies under the
existing training/evaluation protocol? It equals sqrt(`orig_blur`), not an
independent noise estimate. The intervention includes any induced change in
model-family selection, column subsampling and NN distance weighting; it is not
an isolated causal measurement of information content.

| Condition | Surrogate input | NN-K input |
| --- | --- | --- |
| `full` | existing ordered 22 columns | existing ordered 12 page columns |
| `drop_orig_noise` | same order minus only `orig_noise`: 21 columns | same order minus only `orig_noise`: 11 columns |

Keep `inter_noise_bilateral_d` unchanged in values and historical field name;
interpret it explicitly as sqrt(Laplacian variance) times bilateral diameter.
Do not remove this interaction or insert a new noise estimator in this A.

Inputs: frozen VN v3 case-sensitive raw/contour response and NoPre tables, SROIE
casefold response and NoPre tables, and their frozen five page-group folds, all
explicit in `noise_feature_sensitivity_20261012_v1.json`. Keep row/candidate
order, original model definitions/seeds (42), five-fold inner group selection
by MAE, scaler fit scope, K=3/5/10/15, exact Random and static train-only policies.
Use the existing production implementations by import; do not edit their code
or historical manifests. Input CSV columns/values are not deleted or changed.

No OCR, new predictions from OCR, reference edits, metric changes, strict
selector fitting, headroom redesign, additional seeds or primary promotion.
In particular A cannot establish selector robustness. CIs are conditional on
the realized out-of-fold page outcomes, not refit/seed/split uncertainty.

## Execution and reproducibility guards

Dedicated entry point: `run_noise_feature_sensitivity.py`; it supplies feature
lists directly to the surrogate and NN-K production functions. This avoids
editing hash-locked runners or using the legacy ablation defaults. Required
preflight: 300/300/616 pages, 50 distinct common candidates/page, finite inputs,
invariant page features, exact NoPre cohort, and both production GroupKFold
constructions matching the saved folds. Check exact NN distance ties in every
outer fold for both conditions; fail if ties are found rather than silently
introducing a new tie policy. This check is specific to these input tables.

`plan` is read-only. `prepare` refuses an existing output root and freezes
input/baseline/code/config/protocol hashes and fitting-library versions.
`run` requires that manifest unchanged. One writer per root. Checkpoints are
after a completed block/condition/engine job, not inside a fold/family fit.
Interrupted/failed attempts are retained, and a resumed job starts a new attempt.
Completed job files are hash checked before reuse. Do not edit hashed files or
change the environment while a run is in progress. Use a new versioned root for
a changed protocol. Ctrl+C loses the active job's fits, not completed jobs.

There are 16 jobs (two conditions times eight engine/setting blocks). Each uses
the unchanged production routine: 105 outer/nested fits plus 21 final-refit fits,
126 estimator fits/job, 2,016 in total. The final refit is not used in held-out
evaluation; no model pickle is exported. NN-K is evaluated in the same job.
Measure one completed job before estimating elapsed time; its timing is only
representative of that engine/setting/condition, not all jobs. Do not launch
parallel writers using the same root; the fitting models already use n_jobs=-1.

## Comparisons fixed before fitting

1. Reproduction diagnostic: compare rerun `full` with the frozen primary
   per-page outputs, model-family selections and 160 paired/Holm comparisons.
   Record exact changed-page counts and magnitudes; do not assume bit equality.
   Invariant Random/FixedTopK/NoPre/oracle/FixedBest must match within 1e-12 CER.
   Material replay drift must be investigated before interpreting A.
2. Direct intervention: `drop_orig_noise` minus rerun `full` for Surrogate and
   NN separately, at every engine/K. Negative means lower CER after removal.
   Two-sided paired Wilcoxon, zero_method=wilcox, exact all-zero p=1; six Holm
   families (policy times evaluation block), 12 tests for each VN family and
   eight for each SROIE family: 64 comparisons in total. Alpha .05.
3. Recompute the existing five policy contrasts for each condition separately:
   Surrogate vs Random/FixedTopK; NN vs Random/FixedTopK/Surrogate. Each contrast
   times evaluation block is a separate family, 12/12/8 tests as in the primary
   protocol: 160 comparisons and 15 families per condition. Compare significance
   and direction classifications, not just aggregate counts. Also compare the
   full rerun with the original primary statistics.
4. Mean differences are in percentage points. Paired-page percentile 95% CIs,
   10,000 resamples, not multiplicity-adjusted. Page IDs lexicographically
   sorted. Existing policy contrasts retain the existing checksum seed rule.
   Direct comparisons use that same rule with an explicit block/engine namespace
   and tokens `policy`, `drop_orig_noise_vs_full`; save each seed. No seed tuning.

These families are a declared supplementary analysis, not simultaneous error
control over every hypothesis in the paper. Wilcoxon and mean-CER CIs test
different quantities. Non-significance or unchanged Holm classification does
not prove equivalence, absence of impact or stability under other seeds.

## Reporting and decision

Report A in ESM regardless of favorable/unfavorable outcomes. Keep primary
22/12-dimensional results unchanged. Do not automatically promote B if results
improve, or hide A if they worsen. A separate owner-approved design decision and
new protocol would be required for B and all strict-selector downstream work.
Any changed conclusion and any full-replay drift must be stated explicitly.
No predefined CER equivalence margin or equivalence claim is made.

No ESM numeric sentence/table is added before completed outputs are independently
verified. Future public export contains numeric per-page/provenance artifacts,
not OCR text, references or model pickle files. New private preparation artifacts
are not silently inserted into the already-authoritative public bundle.

## Commands (project root)

Cheap checks, without fitting:

```bash
./venv_py310/bin/python analysis/test_noise_feature_sensitivity.py
./venv_py310/bin/python analysis/run_noise_feature_sensitivity.py --config analysis/noise_feature_sensitivity_20261012_v1.json --stage plan
./venv_py310/bin/python analysis/run_noise_feature_sensitivity.py --config analysis/noise_feature_sensitivity_20261012_v1.json --stage prepare
```

Owner-run pilot, one full raw VietOCR job, CPU only:

```bash
./venv_py310/bin/python analysis/run_noise_feature_sensitivity.py --config analysis/noise_feature_sensitivity_20261012_v1.json --stage run --block raw --engine vietocr --condition full --max-new-jobs 1
```

After inspecting the pilot, owner runs remaining jobs (same command resumes):

```bash
./venv_py310/bin/python analysis/run_noise_feature_sensitivity.py --config analysis/noise_feature_sensitivity_20261012_v1.json --stage run
```

Then generate statistics only, once all 16 jobs are complete:

```bash
./venv_py310/bin/python analysis/run_noise_feature_sensitivity.py --config analysis/noise_feature_sensitivity_20261012_v1.json --stage analyze
```

The analysis stage refuses an existing analysis directory. Independent numeric
verification and ESM/public export follow completion; they are not claimed as
done by this protocol/design checkpoint.
