#!/usr/bin/env python3
"""Compare completed strict Surrogate selector with four one-output baselines.

Reads completed full-selector candidate scores only. Each contrast is a separate
eight-test Holm family (raw/contour: three engines each; SROIE: two engines).
No OCR or model fitting is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_policy_matched_top3_selectors import as_bool, bh_adjust, compare, holm_adjust
from compare_strict_policy_selectors import ROOT, EXPECTED_ENGINES, completed_source, sha, load_settings


BASELINES = {
    "top1": ("Top1Pred", "actual_cer", "rank"),
    "agreement": ("Agreement", "actual_cer", "agreement_selected"),
    "fixedbest": ("FixedBest", "fixedbest_actual_cer", None),
    "nopre": ("NoPre", "nopre_cer", None),
}


def selected_series(rows: pd.DataFrame, flag: pd.Series, column: str, label: str) -> pd.Series:
    if flag.sum() != rows.groupby(["engine", "image_name"]).ngroups:
        raise ValueError(f"{label}: expected one chosen row per engine/page")
    chosen = rows.loc[flag, ["engine", "image_name", column]].copy()
    series = chosen.set_index(["engine", "image_name"], verify_integrity=True)[column].sort_index()
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: nonfinite CER")
    return series


def load_methods(path: Path, engines: tuple[str, ...], pages: int) -> dict[str, pd.DataFrame]:
    cols = ["engine", "image_name", "rank", "actual_cer",
            "selected_by_nested_model", "agreement_selected",
            "fixedbest_actual_cer", "nopre_cer"]
    rows = pd.read_csv(path, usecols=cols)
    rows["selected_by_nested_model"] = as_bool(rows["selected_by_nested_model"])
    rows["agreement_selected"] = as_bool(rows["agreement_selected"])
    counts = rows.groupby(["engine", "image_name"]).size()
    if not counts.eq(3).all() or set(rows.engine) != set(engines):
        raise ValueError(f"Wrong candidate or engine coverage: {path}")
    if counts.groupby(level="engine").size().to_dict() != {e: pages for e in engines}:
        raise ValueError(f"Wrong page coverage: {path}")
    if not rows.groupby(["engine", "image_name"])["rank"].apply(
            lambda x: set(x) == {1, 2, 3}).all():
        raise ValueError(f"Invalid top-3 ranks: {path}")
    for column in ["fixedbest_actual_cer", "nopre_cer"]:
        if rows.groupby(["engine", "image_name"])[column].nunique(dropna=False).ne(1).any():
            raise ValueError(f"{column} differs within a shortlist: {path}")
    flags = {
        "surrogate": rows.selected_by_nested_model,
        "top1": rows["rank"].eq(1),
        "agreement": rows.agreement_selected,
        "fixedbest": rows["rank"].eq(1),
        "nopre": rows["rank"].eq(1),
    }
    methods = {}
    for name, flag in flags.items():
        column = "actual_cer" if name in {"surrogate", "top1", "agreement"} else BASELINES[name][1]
        series = selected_series(rows, flag, column, name)
        methods[name] = series.to_frame(name=f"cer_{name}")
    if any(not methods["surrogate"].index.equals(frame.index) for frame in methods.values()):
        raise ValueError(f"Method page alignment differs: {path}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON path overrides for all three settings")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/strict_selector_static_comparison_20260923")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a fresh directory")

    rng = np.random.default_rng(args.seed)
    sources = {}
    rows = []
    for key, setting in load_settings(args.config).items():
        path = completed_source(setting["surrogate"], "full",
                                "strict_nested_surrogate_selector_v1")
        methods = load_methods(path, EXPECTED_ENGINES[key], setting["pages"])
        sources[key] = {"path": str(path), "sha256": sha(path)}
        for baseline, (label, _, _) in BASELINES.items():
            rows.extend(compare(setting["dataset"], setting["setting"],
                                "primary_static_comparison", f"surrogate_vs_{baseline}",
                                methods["surrogate"], methods[baseline], "Surrogate",
                                label, args.bootstrap_replicates, rng))

    table = pd.DataFrame(rows)
    if len(table) != 32 or table.groupby("comparison_family").size().ne(8).any():
        raise ValueError("Expected four separate eight-test comparison families")
    table["holm_family_size"] = 8
    table["p_holm_manuscript"] = np.nan
    table["p_bh_sensitivity"] = np.nan
    for _, index in table.groupby("comparison_family", sort=False).groups.items():
        p_values = table.loc[index, "p_two_sided"].tolist()
        table.loc[index, "p_holm_manuscript"] = holm_adjust(p_values)
        table.loc[index, "p_bh_sensitivity"] = bh_adjust(p_values)

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "strict_selector_static_paired_comparison.csv"
    table.to_csv(output, index=False)
    metadata = {
        "protocol": "strict_selector_static_comparison_v1",
        "families": table.groupby("comparison_family").size().to_dict(),
        "holm_rule": "Four separate eight-test families, one per static comparator",
        "test": "paired two-sided Wilcoxon signed-rank, scipy default zero_method=wilcox",
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed,
                      "interval": "percentile CI for mean paired CER difference"},
        "sources": sources,
        "script_sha256": sha(Path(__file__)),
        "config": ({"path": str(args.config.resolve()), "sha256": sha(args.config)}
                   if args.config else None),
        "dependency_sha256": {
            "compare_strict_policy_selectors.py": sha(ROOT / "analysis/compare_strict_policy_selectors.py"),
            "compare_policy_matched_top3_selectors.py": sha(ROOT / "analysis/compare_policy_matched_top3_selectors.py"),
        },
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Validated three completed full-selector sources; wrote {len(table)} comparisons")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
