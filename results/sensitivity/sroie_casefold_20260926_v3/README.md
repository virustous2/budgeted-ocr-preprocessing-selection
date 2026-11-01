# SROIE casefold sensitivity, v3

Post hoc best-of-K only. CHG-20260926-14 changes display/interpretation, not
the primary results or any of the five numerical CSVs from v2.

From the public-release root, using the versions in the manifests:

```bash
python code/analysis/export_sroie_casefold_sensitivity.py \
  --inputs results/sensitivity/sroie_casefold_20260926_v3/inputs \
  --output-dir /tmp/sroie_casefold_v3_reproduced/tables
OPENBLAS_NUM_THREADS=1 python code/analysis/check_sroie_ci_sensitivity.py \
  --tables /tmp/sroie_casefold_v3_reproduced/tables \
  --output-dir /tmp/sroie_casefold_v3_reproduced/ci_diagnostics
```

Use fresh output directories. No OCR, training, documents or labels needed.
Compare tables/ and ci_diagnostics/ byte-for-byte with this bundle. The diagnostic
runner refuses an existing output directory; it does not resume incomplete runs.

Intervals remain paired percentile 95%, B=10,000, stored seeds, unadjusted for
multiplicity. Significance statements use the specified two-sided Wilcoxon/Holm
families, not mean-CI zero exclusion. TeX pp values use two decimals; rounded
0.00 is not an assertion that the underlying endpoint equals zero. Full precision
is in CSV. All p-values remain unchanged; display retains three significant digits.

Diagnostics compare BCa with percentile using identical resamples, and repeat
K=3/K=5 casefold EasyOCR NN--Surrogate with seeds 0..199. BCa is a diagnostic,
not a replacement chosen after results. MCSE uses the asymptotic quantile formula
and Gaussian KDE with Scott bandwidth; it estimates a standard deviation, not an
absolute error bound. Agreement does not validate CI coverage or account for
model-refit uncertainty. No formal metric-by-policy interaction is tested.

Both Surrogate and NN-K retain 8/8 Holm-significant advantages over Random and
FixedTopK. NN's primary EasyOCR K=3/5 advantages over Surrogate lose significance;
Surrogate's Tesseract advantage extends to K=5. No selector or VN sensitivity
claim is made. Main-text length reduction is deliberately deferred.

Sources supporting the distinction between resampling precision and inference:
- DiCiccio & Efron (1996), Bootstrap Confidence Intervals, Statistical Science
  11(3):189--228, DOI 10.1214/ss/1032280214 (including Rejoinder p.226).
- SciPy bootstrap documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Harrington et al. (2019), New Guidelines for Statistical Reporting in the
  Journal, NEJM 381:285--286, DOI 10.1056/NEJMe1906559. This is reporting context,
  not an endorsement of this project's exact Wilcoxon/bootstrap design.
