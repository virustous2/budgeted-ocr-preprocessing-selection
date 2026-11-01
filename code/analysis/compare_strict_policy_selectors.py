#!/usr/bin/env python3
"""Paired comparisons of completed strict nested top-3 output selectors.

The primary analysis compares the full Surrogate selector with NN-K and
FixedTopK selectors. The separate sensitivity analysis uses the same Surrogate
shortlists with rank-only inputs. Each analysis has exactly 16 two-sided paired
tests: eight setting/engine blocks times two comparators. Holm adjustment
across all 16 tests in each analysis is the manuscript-facing adjustment.

Reads completed outputs only. No OCR or model fitting is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_policy_matched_top3_selectors import (
    add_adjustments, bh_adjust, compare, holm_adjust, load_selected,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "raw": {
        "dataset": "Vietnamese", "setting": "raw", "pages": 300,
        "surrogate": ROOT / "outputs/strict_nested_selector_raw_cohort300_v2_20260906",
        "policy": ROOT / "outputs/strict_nested_policy_controls_raw_cohort300_v2_20260912_v2",
    },
    "contour": {
        "dataset": "Vietnamese", "setting": "contour", "pages": 300,
        "surrogate": ROOT / "outputs_contour/strict_nested_selector_contour_cohort300_v2_20260906",
        "policy": ROOT / "outputs_contour/strict_nested_policy_controls_contour_cohort300_v2_20260912",
    },
    "sroie": {
        "dataset": "SROIE", "setting": "receipt", "pages": 616,
        "surrogate": ROOT / "SROIRE_COMPARISON/output_sroire/strict_nested_selector_sroie_all616_20260906",
        "policy": ROOT / "SROIRE_COMPARISON/output_sroire/strict_nested_policy_controls_sroie_all616_20260912",
    },
}
EXPECTED_ROWS = {"primary_policy_comparison": 16, "rank_only_sensitivity": 16}
EXPECTED_ENGINES = {"raw": ("vietocr", "tesseract", "easyocr"),
                    "contour": ("vietocr", "tesseract", "easyocr"),
                    "sroie": ("tesseract", "easyocr")}
SHORTLIST_PARAMS = ("gaussian_ksize", "bilateral_d", "clahe_clip",
                    "gamma", "sharpen")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_settings(config: Path | None) -> dict:
    """Override paths only; preserve cohort sizes, order and test families."""
    settings = {key: dict(value) for key, value in SETTINGS.items()}
    if config is None:
        return settings
    paths = json.loads(config.read_text())
    if set(paths) != set(SETTINGS):
        raise ValueError("Config must contain exactly raw, contour and sroie")
    for key, value in paths.items():
        if set(value) != {"surrogate", "policy"}:
            raise ValueError(f"{key}: config may override surrogate/policy paths only")
        for name, path in value.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{key}/{name}: expected a nonempty path")
            candidate = Path(path)
            settings[key][name] = candidate if candidate.is_absolute() else ROOT / candidate
    return settings


def completed_source(root: Path, variant: str, protocol: str) -> Path:
    folder = root / variant
    marker = json.loads((folder / "complete.json").read_text())
    if marker.get("protocol") != protocol:
        raise ValueError(f"Unexpected protocol: {folder}")
    manifest = sha(root / "input_manifest.json")
    cache = sha(root / "ocr_cache.csv")
    expected_id = hashlib.sha256((manifest + cache).encode()).hexdigest()
    if marker.get("manifest_sha256") != expected_id or marker.get("cache_sha256") != cache:
        raise ValueError(f"Inputs/cache changed after completion: {folder}")
    for name, digest in marker["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Completed file hash mismatch: {folder / name}")
    return folder / "nested_selector_candidate_scores.csv"


def validated_selected(path: Path, label: str, engines: tuple[str, ...], pages: int) -> pd.DataFrame:
    selected = load_selected(path, label)
    if set(selected.index.get_level_values("engine")) != set(engines):
        raise ValueError(f"{label}: wrong engine coverage")
    counts = selected.groupby(level="engine").size().to_dict()
    if counts != {engine: pages for engine in engines}:
        raise ValueError(f"{label}: wrong page coverage: {counts}")
    if not np.isfinite(selected.iloc[:, 0].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: nonfinite selected CER")
    return selected


def assert_same_surrogate_shortlists(full: Path, rank_only: Path) -> None:
    columns = ["engine", "image_name", "fold", "rank", "actual_cer",
               *SHORTLIST_PARAMS]
    a = pd.read_csv(full, usecols=columns).sort_values(
        ["engine", "image_name", "rank"]).reset_index(drop=True)
    b = pd.read_csv(rank_only, usecols=columns).sort_values(
        ["engine", "image_name", "rank"]).reset_index(drop=True)
    if not a[["engine", "image_name", "fold", "rank"]].equals(
            b[["engine", "image_name", "fold", "rank"]]):
        raise ValueError("Full and rank-only surrogate page/fold/rank rows differ")
    np.testing.assert_allclose(a[["actual_cer", *SHORTLIST_PARAMS]],
                               b[["actual_cer", *SHORTLIST_PARAMS]],
                               rtol=0, atol=1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON path overrides for all three settings")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs/strict_selector_policy_comparison_20260923")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a fresh directory")

    rng = np.random.default_rng(args.seed)
    rows = []
    sources = {}
    for key, setting in load_settings(args.config).items():
        engines = EXPECTED_ENGINES[key]
        source_paths = {
            "surrogate_full": completed_source(setting["surrogate"], "full",
                                               "strict_nested_surrogate_selector_v1"),
            "surrogate_rank": completed_source(setting["surrogate"], "rank_only",
                                               "strict_nested_surrogate_selector_v1"),
            "nn_k": completed_source(setting["policy"], "nn_k",
                                     "strict_nested_policy_selectors_v2"),
            "fixed_topk": completed_source(setting["policy"], "fixed_topk",
                                           "strict_nested_policy_selectors_v2"),
        }
        assert_same_surrogate_shortlists(source_paths["surrogate_full"],
                                         source_paths["surrogate_rank"])
        values = {name: validated_selected(path, f"{key}_{name}", engines,
                                            setting["pages"])
                  for name, path in source_paths.items()}
        if any(not values["surrogate_full"].index.equals(v.index)
               for v in values.values()):
            raise ValueError(f"{key}: selected page/engine indices differ")
        sources[key] = {name: {"path": str(path), "sha256": sha(path)}
                        for name, path in source_paths.items()}
        for analysis, surrogate_key, label in [
            ("primary_policy_comparison", "surrogate_full", "Surrogate"),
            ("rank_only_sensitivity", "surrogate_rank", "Rank-only Surrogate"),
        ]:
            for baseline_key, baseline_label in [("nn_k", "NN-K"),
                                                  ("fixed_topk", "FixedTopK")]:
                rows.extend(compare(setting["dataset"], setting["setting"],
                    analysis, f"surrogate_vs_{baseline_key}",
                    values[surrogate_key], values[baseline_key], label,
                    baseline_label, args.bootstrap_replicates, rng))

    table = add_adjustments(pd.DataFrame(rows))
    # The shared helper groups comparator names across all analysis sets.
    # Here each sensitivity analysis is a separate family, so the descriptive
    # comparator-level columns must also be partitioned by analysis_set.
    for _, index in table.groupby(["analysis_set", "comparison_family"],
                                  sort=False).groups.items():
        p_values = table.loc[index, "p_two_sided"].tolist()
        table.loc[index, "p_holm_within_comparator"] = holm_adjust(p_values)
        table.loc[index, "p_bh_within_comparator"] = bh_adjust(p_values)
    counts = table.groupby("analysis_set").size().to_dict()
    if counts != EXPECTED_ROWS:
        raise ValueError(f"Wrong comparison-family sizes: {counts}")
    if table.groupby(["analysis_set", "comparison_family"]).size().ne(8).any():
        raise ValueError("Expected eight rows for each comparator")
    table["manuscript_holm_family_size"] = table["analysis_set"].map(EXPECTED_ROWS)
    table["p_holm_manuscript"] = table["p_holm_within_analysis_set"]

    args.output_dir.mkdir(parents=True)
    result = args.output_dir / "strict_selector_policy_paired_comparison.csv"
    table.to_csv(result, index=False)
    metadata = {
        "protocol": "strict_selector_policy_comparison_v1",
        "tests": counts,
        "holm_rule": "Holm across all 16 two-sided tests separately within each analysis_set",
        "sensitivity": "Surrogate rank-only inputs retain the same candidate shortlists",
        "test": "paired two-sided Wilcoxon signed-rank, scipy default zero_method=wilcox",
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed,
                      "interval": "percentile CI for mean paired CER difference"},
        "sources": sources,
        "script_sha256": sha(Path(__file__)),
        "helper_sha256": sha(ROOT / "analysis/compare_policy_matched_top3_selectors.py"),
        "config": ({"path": str(args.config.resolve()), "sha256": sha(args.config)}
                   if args.config else None),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(sources) * 4} completed score files; wrote {len(table)} paired comparisons")
    print(f"Primary: {counts['primary_policy_comparison']}; rank-only sensitivity: {counts['rank_only_sensitivity']}")
    print(f"CSV: {result}")


if __name__ == "__main__":
    main()
