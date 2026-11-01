"""Prepare a new figure-only publication version; preserve all numeric inputs.

The two curve rasters and their 96 means are copied byte for byte. Headroom
uses the same distributions with explicit subtraction labels. The existing
vector pipeline is retained as a base and receives a labeled-selector training
annotation, with the matching embedded draw.io source recovered and updated.
No OCR, estimator fitting, CER scoring or inference tests are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_oracle_gap_diagnostic import (
    COMPARISON_ORDER, POLICY_LABELS, POLICY_ORDER, draw_distribution,
)

CHANGE = "CHG-20261016-01"
GAIN_LABELS = {
    "Surrogate_vs_Random": "Random \u2212 Surr.",
    "NN-K_vs_Random": "Random \u2212 NN-K",
    "Surrogate_vs_NN-K": "NN-K \u2212 Surr.",
}
PAPER_ASSETS = (
    "raw_pareto_multiengine.png", "contour_pareto_multiengine.png",
    "headroom_policy_controls.pdf", "headroom_policy_controls.png",
    "pipeline_flowchart_2_5.pdf",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_annotation(source, output):
    shutil.copy2(source, output / "pipeline_base.pdf")
    bbox = subprocess.run(
        ["pdftotext", "-bbox-layout", str(source), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    metadata = ET.fromstring(bbox)
    subject = next(e for e in metadata.iter()
                   if e.tag.endswith("meta") and e.get("name") == "Subject")
    original = unquote(subject.get("content")).replace(r"\(", "(").replace(r"\)", ")")
    diagram = ET.fromstring(original)
    cells = {c.get("id"): c for c in diagram.iter("mxCell")}
    artifact = next(c for c in cells.values()
                    if "Training artifacts:" in c.get("value", ""))
    geometry = artifact.find("mxGeometry")
    if any(geometry.get(k) != v for k, v in
           {"x": "205", "y": "374", "width": "270", "height": "38"}.items()):
        raise ValueError("Unexpected base diagram geometry")
    artifact.set("value",
        '<font style="font-size:12px;">Training artifacts:</font>'
        '<div><font style="font-size:10px;">'
        'Surrogate model + FixedTopK list + NN training pages</font></div>'
        '<div><font style="font-size:10px;">'
        'Output selector trained offline on labeled training shortlists'
        '</font></div>')
    geometry.attrib.update(x="60", y="363", width="560", height="60")
    for cell in cells.values():
        if cell.get("target") == artifact.get("id"):
            cell.set("style", cell.get("style", "") +
                     "entryX=0.5;entryY=0;entryDx=0;entryDy=0;")
    ET.indent(diagram)
    (output / "pipeline_flowchart.drawio").write_text(
        ET.tostring(diagram, encoding="unicode") + "\n")

    # Original draw.io-to-PDF transform, recorded in the exported vector paths.
    # Keep the canvas and every other node unchanged. The wider replacement
    # box also covers the old training-artifact node and incoming arrow tip.
    scale, dx, dy = .69699, -4.347728, -.553983
    def point(x, y):
        return f"({scale*x+dx:.6f},{scale*y+dy:.6f})"
    tex = r"""\pdfminorversion=7
\documentclass[border=0pt]{standalone}
\usepackage{graphicx,tikz,helvet}
\usetikzlibrary{arrows.meta}
\pagestyle{empty}
\begin{document}
\begin{tikzpicture}[x=1bp,y=-1bp]
\useasboundingbox (0,0) rectangle (464.879970,679.919980);
\node[anchor=north west,inner sep=0] at (0,0)
{\includegraphics[width=464.879970bp,height=679.919980bp]{pipeline_base.pdf}};
\definecolor{artifactfill}{HTML}{FAECE7}
\definecolor{artifactstroke}{HTML}{993C1D}
\definecolor{artifacttext}{HTML}{712B13}
"""
    tex += (r"\path[fill=artifactfill,draw=artifactstroke,line width=.697bp,"
            r"rounded corners=2.8bp] " + point(60,363) + " rectangle " +
            point(620,423) + ";\n")
    tex += (r"\draw[black,line width=.697bp,-{Stealth[length=4.4bp,width=3.4bp]}] "
            + point(340,326) + " -- " + point(340,363) + ";\n")
    for y, size, label in (
        (375, 8.36, "Training artifacts:"),
        (392, 6.97, "Surrogate model + FixedTopK list + NN training pages"),
        (408, 6.97, "Output selector trained offline on labeled training shortlists"),
    ):
        tex += (r"\node[text=artifacttext,inner sep=0,font=\sffamily\bfseries"
                rf"\fontsize{{{size}}}{{{size+1}}}\selectfont] at "
                + point(340,y) + " {" + label + "};\n")
    tex += "\\end{tikzpicture}\n\\end{document}\n"
    (output / "pipeline_flowchart_2_5.tex").write_text(tex)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "pipeline_flowchart_2_5.tex"], cwd=output, capture_output=True, text=True)
    (output / "pipeline_build_stdout.txt").write_text(result.stdout + result.stderr)
    result.check_returncode()
    text = subprocess.run(
        ["pdftotext", "-raw", str(output / "pipeline_flowchart_2_5.pdf"), "-"],
        check=True, capture_output=True, text=True).stdout
    if "Output selector trained offline on labeled training shortlists" not in text:
        raise ValueError("Selector training annotation missing from PDF text")


def export(bundle, prior, pipeline, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("curve_points.csv", "raw_pareto_multiengine.png",
                 "contour_pareto_multiengine.png"):
        shutil.copy2(prior / name, output / name)
    for name in ("headroom_policy_summary_allK.csv",
                 "headroom_nonshared_controls_allK.csv"):
        shutil.copy2(bundle / "evidence/headroom" / name, output / name)
    coupled = pd.read_csv(output / "headroom_policy_summary_allK.csv")
    controls = pd.read_csv(output / "headroom_nonshared_controls_allK.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), constrained_layout=True)
    draw_distribution(
        axes[0], coupled, "policy", POLICY_ORDER,
        "spearman_headroom_vs_advantage", POLICY_LABELS,
        "(a) Coupled FixedTopK-based statistic")
    draw_distribution(
        axes[1], controls, "comparison", COMPARISON_ORDER,
        "spearman_headroom_vs_gain", GAIN_LABELS,
        "(b) Non-shared policy-gain controls")
    fig.savefig(output / "headroom_policy_controls.pdf", bbox_inches="tight")
    fig.savefig(output / "headroom_policy_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pipeline_annotation(pipeline, output)
    (output / "README.md").write_text(
        "# Figure clarity update — CHG-20261016-01\n\n"
        "Figure-only version of current VN-v3/SROIE-casefold results. "
        "All 96 curve means and both curve PNGs are unchanged. The two copied "
        "headroom CSVs retain all 192 correlations. Headroom labels state the "
        "signed subtraction; the pipeline explicitly identifies offline "
        "selector training using labeled training shortlists.\n\n"
        "pipeline_base.pdf is the historical vector source; the .tex overlay "
        "is reproducible with pdflatex in this directory. pipeline_flowchart.drawio "
        "is recovered from that PDF's embedded source and updated consistently. "
        "The old unrelated draw.io file is historical, not this PDF's source.\n\n"
        "The ESM caption and stacked raw/contour layout live in paper/ESM_1.tex. "
        "Primary CER, models, folds, tests and numeric release are unchanged. "
        "These are figure-consistency checks, not scientific-validity evidence.\n")
    names = [*PAPER_ASSETS, "curve_points.csv",
             "headroom_policy_summary_allK.csv", "headroom_nonshared_controls_allK.csv",
             "pipeline_base.pdf", "pipeline_flowchart_2_5.tex",
             "pipeline_flowchart.drawio", "README.md"]
    metadata = {
        "change": CHANGE,
        "scope": "figure labels, pipeline supervision annotation, ESM layout/caption",
        "bundle_manifest_sha256": sha(bundle / "manifest.json"),
        "prior_figures_manifest_sha256": sha(prior / "manifest.json"),
        "pipeline_base_sha256": sha(pipeline),
        "script_path": "analysis/export_figure_clarity.py",
        "script_sha256": sha(__file__),
        "plot_helper_path": "analysis/plot_oracle_gap_diagnostic.py",
        "plot_helper_sha256": sha(Path(__file__).with_name("plot_oracle_gap_diagnostic.py")),
        "paper_assets": list(PAPER_ASSETS),
        "headroom_gain_labels": GAIN_LABELS,
        "files": {name: sha(output / name) for name in names},
        "numerical_impact": "none; copied curve means and headroom distributions",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "payload_files": len(names),
                      "curve_points": 96, "headroom_correlations": len(coupled)+len(controls)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prior-figures", type=Path, required=True)
    parser.add_argument("--pipeline-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.bundle.resolve(), args.prior_figures.resolve(),
           args.pipeline_source.resolve(), args.output_dir.resolve())
