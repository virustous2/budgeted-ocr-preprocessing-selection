# Post hoc standalone-feature sensitivity A

Supplementary best-of-K only; NOT replacement primary results. Full uses 22
surrogate/12 NN columns; Drop removes ONLY orig_noise (21/11) and retains
inter_noise_bilateral_d = sqrt(Laplacian variance) times bilateral diameter.
The protocol was fixed before these sensitivity fits, not before the original
study. Model definitions/seed42/folds/row order/targets were retained; changes
also include model-family choice, column subsampling and NN distance weighting.
No OCR, strict selector refit, additional seeds or equivalence claim in A.

From the public release root (NumPy, pandas and SciPy needed):

```bash
python code/analysis/replay_noise_feature_sensitivity.py \
  --bundle results/sensitivity/noise_feature_20261014_v1
```

This verifies 384 means/two-sided Wilcoxon p-values/paired percentile CIs and
36 Holm families from the numeric paired pages alone. CIs use 10,000 draws with
recorded seeds and are per comparison, not multiplicity-adjusted or refit/seed/
split uncertainty. It does not prove coverage or scientific validity.

Full exactly reproduced the primary per-page CER/family selections and numerical
statistical rows. Drop changes 10/160 significance classifications; all are in
holm_changes_drop_vs_full.csv and the ESM table. This does not prove equivalence
or test differences between contrasts. Strong SROIE Surrogate/NN advantages over
FixedTopK remain 8/8, but secondary policy distinctions are feature-sensitive.
Only one direct test is Holm-significant (SROIE Tesseract Surrogate K3); its
unadjusted mean-CER CI includes zero, so no significant mean-reduction claim.

CSV copies preserve source bytes/full precision; TeX alone rounds effects/CI to
two decimals and p-values to three significant digits. No images, labels, OCR
text, model pickle, private paths or raw runtime logs are included. Source hash
identifiers and the sanitized independent verification accompany this bundle.
The archived protocol's training commands refer to the original private project
layout; do not treat them as runnable public fitting commands. Training needs
appropriately laid-out inputs; strict selectors additionally need excluded OCR
text. The public replay above needs none of these private inputs.
