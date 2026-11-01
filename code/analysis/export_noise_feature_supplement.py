#!/usr/bin/env python3
"""Export verified A numbers/tables to a fresh text-free supplementary bundle."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd
from audit_noise_feature_statistics import sha, require

ROOT = Path(__file__).resolve().parents[1]


def signed(value):
    return "0.00" if value == 0 else f"{value:+.2f}"


def tables(direct, changes):
    ranges = []
    for (block,policy),g in direct.groupby(["block","policy"],sort=True):
        sig = g.p_holm < .05
        ranges.append(" & ".join([{"raw":"VN raw","contour":"VN contour","sroie":"SROIE"}[block],
            "NN-K" if policy == "NN" else policy,signed(g.difference_pp.min()),signed(g.difference_pp.max()),
            str(len(g)),f"{int((sig&(g.difference_pp<0)).sum())}/{int((sig&(g.difference_pp>0)).sum())}"])+r" \\")
    rows = []
    for r in changes[changes.changed].itertuples():
        rows.append(" & ".join([{"raw":"VN raw","contour":"VN contour","sroie":"SROIE"}[r.block],
            {"vietocr":"VietOCR","tesseract":"Tesseract","easyocr":"EasyOCR"}[r.engine],str(r.k),
            {"Surrogate":"Surr.","NN":"NN-K"}[r.method_a]+"--"+
            {"FixedTopK":"FixedTopK","Surrogate":"Surr."}[r.method_b],
            signed(r.difference_pp_a),signed(r.difference_pp_b),f"{r.p_holm_a:#.3g}",f"{r.p_holm_b:#.3g}"])+r" \\")
    return "\n".join([
        r"\begin{table}[htbp]",r"\centering\small",
        r"\caption{Standalone Laplacian-standard-deviation removal: range of Drop-minus-Full mean best-of-$K$ CER differences across engines and budgets (pp). Negative is lower CER after removal. The final column counts Holm-significant lower/higher results in each separate direct-comparison family. All 64 comparisons, including CIs, are released.}",
        r"\label{tab:noise_direct_ranges}",r"\begin{tabular}{llrrrr}\toprule",
        r"Setting & Policy & Minimum & Maximum & Tests & Lower/Higher \\\midrule",
        *ranges,r"\bottomrule\end{tabular}\end{table}","",
        r"\begin{table}[htbp]",r"\centering\small",
        r"\caption{All ten policy comparisons whose Holm-adjusted significance classification changes after removing only the standalone column. Differences are first policy minus second (pp); negative favors the first. $p_H$ uses the same prespecified block/contrast family in each condition (12 tests for each Vietnamese block, eight for SROIE). These threshold crossings are not formal tests of a difference between contrasts. Full-precision results for all 160 comparisons per condition are released.}",
        r"\label{tab:noise_changed_conclusions}",r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll l rrrr}\toprule",
        r"Setting & Engine & $K$ & Contrast & Full $\Delta$ & Drop $\Delta$ & Full $p_H$ & Drop $p_H$ \\\midrule",
        *rows,r"\bottomrule\end{tabular}}",r"\end{table}",""])


def generate(config_path, verification_path, out):
    require(not out.exists(),"Use a fresh release directory")
    config=json.loads(config_path.read_text()); source=ROOT/config["output_dir"]/"analysis_v1"
    verification=json.loads(verification_path.read_text())
    require(verification["comparisons_replayed"]==384 and verification["holm_families_replayed"]==36,"Incomplete audit")
    require(verification["hashes"]["analysis_manifest"]==sha(source/"manifest.json"),"Audited analysis mismatch")
    require(verification["hashes"]["verifier"]==sha(ROOT/"analysis/audit_noise_feature_statistics.py"),"Verifier changed after audit")
    meta=json.loads((source/"manifest.json").read_text())
    for name,digest in meta["files"].items(): require(sha(source/name)==digest,f"Source changed: {name}")
    copies={name:source/name for name in (
        "paired_conditions.csv","direct_drop_minus_full.csv","full/statistics.csv","drop_orig_noise/statistics.csv",
        "holm_changes_drop_vs_full.csv","holm_changes_full_vs_primary.csv",
        "full_vs_primary_perpage_drift.csv","full_vs_primary_model_families.csv")}
    copies["primary_statistics.csv"]=ROOT/config["primary_statistics"]
    copies["verification.json"]=verification_path
    copies["NOISE_FEATURE_SENSITIVITY_PROTOCOL.md"]=ROOT/"analysis/NOISE_FEATURE_SENSITIVITY_PROTOCOL.md"
    # Check CSV field whitelist before copying. No text cache/model/response export.
    forbidden={"ocr_text","reference_text","prediction","text","label","label_text"}
    for name,path in copies.items():
        if name.endswith(".csv"):
            with path.open(newline="") as handle:
                fields=next(csv.reader(handle))
            require(not (set(fields)&forbidden),f"Unexpected text field: {name}")
    out.mkdir(parents=True,exist_ok=False)
    for name,path in copies.items():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    direct=pd.read_csv(out/"direct_drop_minus_full.csv")
    changes=pd.read_csv(out/"holm_changes_drop_vs_full.csv")
    (out/"noise_feature_tables.tex").write_text(tables(direct,changes))
    (out/"README.md").write_text("""# Post hoc standalone-feature sensitivity A

Supplementary best-of-K only; NOT replacement primary results. Full uses 22
surrogate/12 NN columns; Drop removes ONLY orig_noise (21/11) and retains
inter_noise_bilateral_d = sqrt(Laplacian variance) times bilateral diameter.
The protocol was fixed before these sensitivity fits, not before the original
study. Model definitions/seed42/folds/row order/targets were retained; changes
also include model-family choice, column subsampling and NN distance weighting.
No OCR, strict selector refit, additional seeds or equivalence claim in A.

From the public release root (NumPy, pandas and SciPy needed):

```bash
python code/analysis/replay_noise_feature_sensitivity.py \\
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
""")
    manifest=dict(kind="supplementary_best_of_K_not_primary",change="CHG-20261014-03",
        created_utc=datetime.now(timezone.utc).isoformat(),source_analysis_manifest_sha256=sha(source/"manifest.json"),
        source_verification_sha256=sha(verification_path),full_vs_primary_changes=verification["full_vs_primary_holm_changes"],
        drop_vs_full_changes=verification["drop_vs_full_holm_changes"],
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        exporter_sha256=sha(__file__),scope="Post hoc best-of-K; primary/strict selectors unchanged; no equivalence/coverage claim")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Supplement written to {out}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--verification",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();generate(args.config,args.verification,args.output_dir)
