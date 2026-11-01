# M2 diagnostic — CHG-20261011-02

Measure raw-page skew independently of the stored Hough feature. This is a
diagnostic estimate, not an OCR experiment or evidence that deskew cannot help.
Do not change features, references, candidate pools, models or CER outputs.

- Use the hashed 300-page cohort in the current numeric publication bundle.
- Grayscale input, downsize only when width exceeds1000pixels, Otsu inverted
  ink mask. Zero-pad to diagonal-sized square before rotation to avoid clipping.
- Search correction angles -5 to+5 degrees in0.05degree increments using
  bilinear rotation. Score=sum of squared adjacent differences in horizontal
  ink projection. Pick maximum, ties by absolute angle then signed angle.
- Retain per-page estimates, full angle score curves, hashes, software/config,
  elapsed time, flat objectives and range-boundary flags. Fresh output directory
  only. No resume: if interrupted, retain partial directory and use a new name.
- Geometry sign is the correction angle; reported magnitude is absolute value.
  Blank/line-dominated/multicolumn pages can mislead this estimator. Inspect
  extremes and flagged objectives against images before making manuscript claims.
- The earlier chat's median0.15/max1.4 has no provided script/results and is not
  assumed correct. This implementation may legitimately give different values.
- Synthetic straight/tilted-line tests validate implementation, not accuracy on
  actual documents. No uncertainty/ground-truth guarantee from a0.05degree grid.

Owner-run full command (CPU only; no OCR or model fitting):

```bash
./venv_py310/bin/python analysis/measure_vn_projection_skew.py \
  --output-dir analysis/outputs/vn_projection_skew_raw_20261011_v1
```

Run complete before writing quantitative main/ESM text. Agent verifies300IDs,
hashes, finite score curves, summary recomputation and image examples. Keep the
intensity-only scope rationale independently of this diagnostic. No feature
removal/refit is required solely to correct manuscript wording.
