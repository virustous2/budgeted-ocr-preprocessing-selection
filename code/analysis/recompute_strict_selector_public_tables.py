#!/usr/bin/env python3
"""Recompute strict selector comparison tables from sanitized public CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_policy_matched_top3_selectors import (
    as_bool, bh_adjust, compare, holm_adjust, load_selected,
)


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
DEFAULT_RELEASE = (
    HERE.parents[2] / "results/oracle_free_selector/strict"
    if HERE.parent.parent.name == "code"
    else PROJECT_ROOT / "ijdar_public_release/results/oracle_free_selector/strict"
)
SETTINGS = {
    "raw": ("Vietnamese", "raw", 300, ("vietocr", "tesseract", "easyocr")),
    "contour": ("Vietnamese", "contour", 300, ("vietocr", "tesseract", "easyocr")),
    "sroie": ("SROIE", "receipt", 616, ("tesseract", "easyocr")),
}
STATIC = {
    "top1": ("Top1Pred", "actual_cer"),
    "agreement": ("Agreement", "actual_cer"),
    "fixedbest": ("FixedBest", "fixedbest_actual_cer"),
    "nopre": ("NoPre", "nopre_cer"),
}


def validated_selected(path: Path, label: str, engines: tuple[str, ...], pages: int) -> pd.DataFrame:
    selected = load_selected(path, label)
    if set(selected.index.get_level_values("engine")) != set(engines):
        raise ValueError(f"{label}: wrong engine coverage")
    if selected.groupby(level="engine").size().to_dict() != {engine: pages for engine in engines}:
        raise ValueError(f"{label}: wrong page coverage")
    return selected


def static_methods(path: Path, engines: tuple[str, ...], pages: int) -> dict[str, pd.DataFrame]:
    columns = ["engine", "image_name", "rank", "actual_cer",
               "selected_by_nested_model", "agreement_selected",
               "fixedbest_actual_cer", "nopre_cer"]
    rows = pd.read_csv(path, usecols=columns)
    rows["selected_by_nested_model"] = as_bool(rows["selected_by_nested_model"])
    rows["agreement_selected"] = as_bool(rows["agreement_selected"])
    groups = rows.groupby(["engine", "image_name"])
    if groups.size().ne(3).any() or groups.ngroups != pages * len(engines):
        raise ValueError(f"Wrong candidate/page coverage: {path}")
    flags = {
        "surrogate": rows.selected_by_nested_model,
        "top1": rows["rank"].eq(1),
        "agreement": rows.agreement_selected,
        "fixedbest": rows["rank"].eq(1),
        "nopre": rows["rank"].eq(1),
    }
    methods = {}
    for name, flag in flags.items():
        if rows.assign(chosen=flag).groupby(["engine", "image_name"]).chosen.sum().ne(1).any():
            raise ValueError(f"{name}: expected one choice per page")
        column = "actual_cer" if name in {"surrogate", "top1", "agreement"} else STATIC[name][1]
        series = rows.loc[flag].set_index(["engine", "image_name"])[column].sort_index()
        methods[name] = series.to_frame(name=f"cer_{name}")
    return methods


def policy_table(release: Path, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = []
    for key, (dataset, setting, pages, engines) in SETTINGS.items():
        paths = {
            "surrogate_full": release / "full" / key / "nested_selector_candidate_scores_sanitized.csv",
            "surrogate_rank": release / "rank_only" / key / "nested_selector_candidate_scores_sanitized.csv",
            "nn_k": release / "policy_matched" / key / "nn_k/nested_selector_candidate_scores_sanitized.csv",
            "fixed_topk": release / "policy_matched" / key / "fixed_topk/nested_selector_candidate_scores_sanitized.csv",
        }
        values = {name: validated_selected(path, f"{key}_{name}", engines, pages)
                  for name, path in paths.items()}
        if any(not values["surrogate_full"].index.equals(frame.index) for frame in values.values()):
            raise ValueError(f"{key}: page alignment differs")
        for analysis, surrogate_key, label in [
            ("primary_policy_comparison", "surrogate_full", "Surrogate"),
            ("rank_only_sensitivity", "surrogate_rank", "Rank-only Surrogate"),
        ]:
            for baseline, baseline_label in [("nn_k", "NN-K"), ("fixed_topk", "FixedTopK")]:
                output.extend(compare(dataset, setting, analysis, f"surrogate_vs_{baseline}",
                                      values[surrogate_key], values[baseline], label,
                                      baseline_label, replicates, rng))
    table = pd.DataFrame(output)
    table["p_holm_within_comparator"] = np.nan
    table["p_bh_within_comparator"] = np.nan
    table["p_holm_within_analysis_set"] = np.nan
    table["p_bh_within_analysis_set"] = np.nan
    for _, index in table.groupby(["analysis_set", "comparison_family"], sort=False).groups.items():
        p = table.loc[index, "p_two_sided"].tolist()
        table.loc[index, "p_holm_within_comparator"] = holm_adjust(p)
        table.loc[index, "p_bh_within_comparator"] = bh_adjust(p)
    for _, index in table.groupby("analysis_set", sort=False).groups.items():
        p = table.loc[index, "p_two_sided"].tolist()
        table.loc[index, "p_holm_within_analysis_set"] = holm_adjust(p)
        table.loc[index, "p_bh_within_analysis_set"] = bh_adjust(p)
    table["manuscript_holm_family_size"] = 16
    table["p_holm_manuscript"] = table["p_holm_within_analysis_set"]
    return table


def static_table(release: Path, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = []
    for key, (dataset, setting, pages, engines) in SETTINGS.items():
        path = release / "full" / key / "nested_selector_candidate_scores_sanitized.csv"
        methods = static_methods(path, engines, pages)
        for baseline, (label, _) in STATIC.items():
            output.extend(compare(dataset, setting, "primary_static_comparison",
                                  f"surrogate_vs_{baseline}", methods["surrogate"],
                                  methods[baseline], "Surrogate", label, replicates, rng))
    table = pd.DataFrame(output)
    table["holm_family_size"] = 8
    table["p_holm_manuscript"] = np.nan
    table["p_bh_sensitivity"] = np.nan
    for _, index in table.groupby("comparison_family", sort=False).groups.items():
        p = table.loc[index, "p_two_sided"].tolist()
        table.loc[index, "p_holm_manuscript"] = holm_adjust(p)
        table.loc[index, "p_bh_sensitivity"] = bh_adjust(p)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty")
    args.output_dir.mkdir(parents=True)
    policy = policy_table(args.release_root, args.bootstrap_replicates, args.seed)
    static = static_table(args.release_root, args.bootstrap_replicates, args.seed)
    policy.to_csv(args.output_dir / "strict_selector_policy_paired_comparison.csv", index=False)
    static.to_csv(args.output_dir / "strict_selector_static_paired_comparison.csv", index=False)
    metadata = {"protocol": "strict_selector_public_recompute_v1",
                "release_root": str(args.release_root), "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed, "policy_rows": len(policy), "static_rows": len(static)}
    (args.output_dir / "recompute_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(policy)} policy and {len(static)} static comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
