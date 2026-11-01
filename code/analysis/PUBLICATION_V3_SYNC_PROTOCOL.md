# Publication synchronization — CHG-20261010-07

No OCR, fitting, seed selection or metric redesign in this step. Promote the
completed VN-v3 case-sensitive and SROIE-casefold runs listed in the two source
configs and in `publication_authority_v3.json`. Historical results remain intact.

1. Export into a fresh directory using `export_publication_v3.py`. This collects
   frozen response/per-page/statistical outputs, removes only `ocr_text` from
   selector scores, and preserves all retained CSV field strings. V1 was a
   preview using pandas serialization; V2 supersedes it with lexical preservation.
2. Replay 160 best-of-K and 64 selector comparisons using
   `replay_publication_v3.py --bundle <directory>`. It is portable and needs only
   the numeric bundle, NumPy, pandas and SciPy. It checks 21 Holm families and
   the recorded percentile CI streams. It does not establish nominal coverage.
3. Render 13 row-only TeX assets; keep captions/header structure in the paper.
   Plot two VN best-of-K curves and headroom using `plot_publication_v3.py`.
   Retain plotted means and figure/source hashes. Stage in fresh directories.
4. Update main/ESM, including adverse results, zero sensitivity, post hoc SROIE
   metric-transition chronology and original-text selector feature scope. Retain
   the historical case-sensitive comparison; its CI stream differs from the new
   consolidated export and is explicitly described as historical.
5. Build both PDFs, synchronize local public paper/figures/tables and a versioned
   numeric bundle. Update source map and release instructions. Never silently
   replace historical fitting outputs or strict manifests.
6. Run focused tests and `audit_submission_artifacts.py`. The explicit authority
   configuration switches that entry point to `audit_publication_v3.py`, rather
   than changing September expected numbers until an old audit passes.

The old audit functions remain available for historical snapshots; they are not
applicable to the current TeX. Audit means consistency/provenance and implemented
arithmetic replay, not scientific validity. No remote push in this protocol.
Page shortening is a separate final editorial step.

CHG-20261010-08: preserve signs for nonzero selector-CI endpoints that round to
zero. `export_publication_table_format.py` writes a fresh display-only overlay;
do not rewrite the numeric bundle or its original table snapshot. Current audit
checks historical bundle rendering and current signed-endpoint rendering
separately. Update ESM captions, build and resynchronize; no OCR/fit/CI resampling.

CHG-20261011-01: current overlay is publication_table_format_20261011_v3.
Use --compact-sroie to omit duplicated model quality from the main SROIE table
and render separate K=3/15 rows. Historical numeric-bundle rendering keeps its
original 15 columns; signed CI endpoints remain unchanged. V2 is a local-only
12-column layout trial, superseded by the more readable nine-column V3.

CHG-20261011-05: current display overlay is publication_table_format_20261011_v4.
Run the display exporter with --compact-sroie --preserve-effect-sign. The latter
retains the sign of nonzero Sel-Fixed differences even when rounded to 0.00.
Historical renderer defaults remain unchanged for numeric-bundle replay. No
feature, OCR, CER, model, test-family or CI stream is changed by this display and
documentation checkpoint. Methods explicitly describe Laplacian redundancy and
color NoPre versus the grayscale pool; the grayscale-only baseline was not
evaluated. Remote publication is still separate.

CHG-20261014-07: retain original numeric bundle v2 and its manifest untouched.
Create metadata-only portable release v3 with `sanitize_publication_paths.py`:
explicit --source, fresh --output-dir and --project-root. Only the six listed
provenance JSON files lose the project-root prefix; all other payload bytes stay
unchanged. Independently check `audit_portable_release.py`, retain before/after
hashes in the new manifest, and replay 224 comparisons from public v3. Move the
former public v2 copy to the private rollback archive rather than leave personal
paths in public. Historical contour summary gets the same prefix-only change;
its original report/CSV remain intact. Figure/display manifests continue to
identify original numerical v2. Switch only public/portable paths in authority;
no OCR/model/statistical rerun or retrospective experimental-manifest rewrite.

CHG-20261016-01: figure-only clarification uses a new figure/source bundle
publication_figure_clarity_20261016_v2 and a separate public_figures authority
path. The original numerical/portable/table/figure manifests remain unchanged.
export_figure_clarity.py copies both curve rasters and their 96 means, copies
the 192 source correlations, redraws only explicit signed headroom labels and
adds supervised offline selector training to the existing vector pipeline.
The matching draw.io source is recovered from the base PDF; the older unrelated
draw.io file remains historical. A retained v1 attempt failed a PDF-text reader
check after rendering; v2 uses raw text-stream extraction and preserves that
failure/snapshot rather than replacing its output.

ESM Fig.1 stacks the two unchanged curve rasters at full text width, labels Raw
and Contour, states separate y-axis scales and cites both partial main tables
plus the complete public results. Build both PDFs, synchronize paper/source
copies, independently verify all 96 means and 192 signed correlations, replay
the public numeric release and run the entry-point audit. No CER, folds,
shortlists, model fitting, primary test family or manuscript aggregate changes.
