# External Source Documents

The original Vietnamese images and reference transcriptions, and the SROIE benchmark files, are intentionally excluded from this release. To use the default paths in the OCR collection and validation scripts, create the following local layout after obtaining the source material under the applicable terms:

```text
data/source_documents_not_included/
  vietnamese/
    images/
    references/
  sroie/
    images/
    ground_truth/
    gt_boxes/
```

All scripts also accept explicit input-path arguments. Outputs generated during a local rerun are written under `results/recomputed/` by default.

The reported SROIE cohort contains 616 receipts. After initial filename-level filtering yielded 626 candidates, a SHA-256 audit found five pairs of byte-identical images with inconsistent benchmark-provided reference transcriptions. Both records in each ambiguous pair were excluded before fold construction. The identifiers and hashes are listed in `../cohort_manifests/sroie_excluded_exact_duplicate_clusters.csv`.
