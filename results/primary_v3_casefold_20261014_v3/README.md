# Current primary numerical bundle — CHG-20261010-07

VN raw/contour: corrected frozen references v3, case-sensitive CER. SROIE:
Unicode-casefolded CER (both predictions and references). Both normalize NFC
and whitespace symmetrically, retaining accents/punctuation. SROIE metric
transition was post hoc after initial and best-of-K sensitivity results; see
main/ESM. Original case-sensitive SROIE remains in the historical sensitivity.

- `response/`: frozen targets/NoPre/folds, no source text or images.
- `per_page_bestk/`, `nn_baseline/`: original page-level outputs, not refitted.
- `selectors/`: 12 variants; only OCR text removed, other CSV fields unchanged.
  Selector features retain original OCR case; SROIE targets are casefolded.
- `evidence/`: 160 best-of-K tests (12/12/8-test families per contrast), 64
  selector comparisons (16 tests per full/rank family, 8 per static baseline),
  headroom, oracle benefit, saved model-quality aggregates and zero sensitivity.
- `tables/`: 13 row-only TeX assets matching current main/ESM.
- `vietnamese_cohort_manifest.csv`: 300 identifiers, image hashes and evaluated
  v3 reference hashes. References themselves are not redistributed.
- `manifest.json`: source and payload hashes. Metadata may identify upstream
  filesystem paths for provenance; no private images/text/caches/models included.

Portable arithmetic reproduction from the public repository root:

```bash
python code/analysis/replay_publication_v3.py \
  --bundle results/primary_v3_casefold_20261014_v3
```

This independently replays 224 means/Wilcoxon tests/paired percentile CIs and
21 Holm families from numeric records, without project statistical helpers,
OCR or fitting. Requires NumPy, pandas and SciPy. Uses two-sided wilcox,
10,000 percentile resamples, saved per-row best-of-K seeds and selector seed42
in recorded row order. CIs are pointwise, not multiplicity-adjusted. Historical
SROIE CS/CF sensitivity uses its own documented CI seed stream; do not assert
bit-identical CIs across exports with different streams. Replay establishes
implemented consistency, not nominal coverage, model correctness or external
validity. Further fitting requires the excluded source data/OCR text.

V1 bundle remains a non-authoritative preview because selector numeric fields
were reserialized. V2 preserves those fields lexically and supersedes V1.
No experiment result was altered to force an audit pass.

Other released export/plot source snapshots document the full project workflow
and may expect its original private directory layout. They are not substitutes
for the portable replay command above. Bundle preparation/refitting cannot be
reproduced from an image/text-free release alone.

## Portable metadata release — CHG-20261014-07

Only six provenance JSON files have their original project-root prefix removed.
These relative paths identify upstream project artifacts; they are not promises
that excluded private inputs exist in this public repository. All numeric CSVs,
tables, folds and model-output records are byte-identical to the source v2 bundle.
The release manifest retains the source manifest hash, original input hashes,
and before/after hashes for each sanitized metadata file. Figure/display
provenance still refers to that original numeric-source manifest.
