# Corrected SROIE612 publication evidence

VN300 raw/contour use frozen v3 case-sensitive references; SROIE612 casefolds
both sides. Both members of two additional same-receipt pairs are excluded.
All affected SROIE training layers were refit on new GroupKFold partitions;
old/new differences are not a duplicate-only causal effect. No OCR was rerun.
The 2027-candidate review is not a guarantee of exhaustive duplicate recall.

Primary:224 paired comparisons/21 Holm families. A:384/36. Matched metric
sensitivity:80/10. Some comparisons are deliberately repeated across analyses;
these are not 688 unique hypotheses. All means/CI endpoints/p-values remain
full precision in CSVs. CIs:10000 paired-page percentile draws,unadjusted and
conditional on these fitted folds,not refit/seed/split uncertainty.

Replay from the public repository root (NumPy,pandas,SciPy;no OCR/training):

    python code/analysis/replay_receipt612_publication.py --bundle results/primary_receipt612_20261019_v3

All seven A classification changes and adverse primary outcomes are released.
No equivalence or selector feature-removal robustness claim. Historical616
BCa/seed diagnostics do not validate new612 CIs. Saved model-quality summaries
are not independent reconstructions of unavailable raw OOF predictions.

No images,crops,full transcriptions,OCR strings,caches or model pickle files
are included. Provenance paths are project-relative identifiers;upstream private
inputs are not all included. Old616 releases are historical,not current results.
