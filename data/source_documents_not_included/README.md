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
