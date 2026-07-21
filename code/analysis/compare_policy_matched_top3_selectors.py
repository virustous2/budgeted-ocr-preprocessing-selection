#!/usr/bin/env python3
"""Compare nested oracle-free selectors across matched top-3 policies.

The comparison is between complete deployable policies: policy-specific
preprocessing top-3 shortlist construction followed by the nested oracle-free
output selector. It reads saved candidate-score CSVs only; it never runs OCR,
trains a surrogate, or modifies an OCR cache.

Each input is validated to contain exactly three candidates and one selector
choice per engine/page. The output reports paired CER differences, bootstrap
confidence intervals, two-sided Wilcoxon tests, and Holm/BH adjustments.

An optional feature-matched Surrogate ablation replaces the Surrogate's
predicted-CER features with within-shortlist ranks 1, 2, and 3. Its tests are
reported as a separate sensitivity-analysis set, so they do not alter the
multiplicity adjustments for the primary policy comparison.
SROIE NN-K is intentionally omitted because exact historical top-3 provenance
is incomplete for three EasyOCR pages.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_RESULTS = RELEASE_ROOT / "results" / "oracle_free_selector"
K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SELECTOR_RESULTS / "policy_comparisons" / "recomputed",
    )
    parser.add_argument("--surrogate-raw", type=Path, default=SELECTOR_RESULTS / "raw/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--surrogate-contour", type=Path, default=SELECTOR_RESULTS / "contour/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--surrogate-sroie", type=Path, default=SELECTOR_RESULTS / "sroie/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--nn-raw", type=Path, default=SELECTOR_RESULTS / "policy_matched/raw/nn_k/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--nn-contour", type=Path, default=SELECTOR_RESULTS / "policy_matched/contour/nn_k/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--fixed-raw", type=Path, default=SELECTOR_RESULTS / "policy_matched/raw/fixed_topk/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--fixed-contour", type=Path, default=SELECTOR_RESULTS / "policy_matched/contour/fixed_topk/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument("--fixed-sroie", type=Path, default=SELECTOR_RESULTS / "policy_matched/sroie/fixed_topk/nested_selector_candidate_scores_sanitized.csv")
    parser.add_argument(
        "--feature-matched-raw",
        type=Path,
        help="Optional raw feature-matched Surrogate candidate-score CSV.",
    )
    parser.add_argument(
        "--feature-matched-contour",
        type=Path,
        help="Optional contour feature-matched Surrogate candidate-score CSV.",
    )
    parser.add_argument(
        "--feature-matched-sroie",
        type=Path,
        help="Optional SROIE feature-matched Surrogate candidate-score CSV.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes", "y"})


def load_selected(path: Path, label: str) -> pd.DataFrame:
    required = {"engine", "image_name", "actual_cer", "selected_by_nested_model"}
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing candidate-score file: {path}")
    rows = pd.read_csv(path)
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(sorted(missing))}")
    if rows["actual_cer"].isna().any():
        raise ValueError(f"{label}: actual_cer contains missing values")

    rows["engine"] = rows["engine"].astype(str)
    rows["image_name"] = rows["image_name"].astype(str)
    rows["selected_by_nested_model"] = as_bool(rows["selected_by_nested_model"])
    group_sizes = rows.groupby(["engine", "image_name"], sort=False).size()
    if not (group_sizes == K).all():
        bad = group_sizes[group_sizes != K].head(5).to_dict()
        raise ValueError(f"{label}: expected {K} candidate rows per engine/page; examples: {bad}")
    selected_counts = rows.groupby(["engine", "image_name"], sort=False)["selected_by_nested_model"].sum()
    if not (selected_counts == 1).all():
        bad = selected_counts[selected_counts != 1].head(5).to_dict()
        raise ValueError(f"{label}: expected one selected row per engine/page; examples: {bad}")

    selected = rows.loc[rows["selected_by_nested_model"], ["engine", "image_name", "actual_cer"]].copy()
    selected = selected.rename(columns={"actual_cer": f"cer_{label}"})
    return selected.set_index(["engine", "image_name"], verify_integrity=True).sort_index()


def wilcoxon_two_sided(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if np.all(a - b == 0.0):
        return math.nan, 1.0
    statistic, p_value = wilcoxon(
        a,
        b,
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(statistic), float(p_value)


def bootstrap_ci(diff: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    if replicates <= 0 or not len(diff):
        return math.nan, math.nan
    samples = diff[rng.integers(0, len(diff), size=(replicates, len(diff)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    ordered = valid[np.argsort(values[valid], kind="mergesort")]
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    ordered = valid[np.argsort(values[valid], kind="mergesort")]
    raw = np.asarray([min(1.0, values[index] * len(ordered) / (rank + 1)) for rank, index in enumerate(ordered)])
    adjusted[ordered] = np.minimum.accumulate(raw[::-1])[::-1]
    return adjusted.tolist()


def add_adjustments(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["p_holm_within_comparator"] = math.nan
    results["p_bh_within_comparator"] = math.nan
    for _, index in results.groupby("comparison_family", sort=False).groups.items():
        values = results.loc[index, "p_two_sided"].tolist()
        results.loc[index, "p_holm_within_comparator"] = holm_adjust(values)
        results.loc[index, "p_bh_within_comparator"] = bh_adjust(values)
    results["p_holm_within_analysis_set"] = math.nan
    results["p_bh_within_analysis_set"] = math.nan
    for _, index in results.groupby("analysis_set", sort=False).groups.items():
        values = results.loc[index, "p_two_sided"].tolist()
        results.loc[index, "p_holm_within_analysis_set"] = holm_adjust(values)
        results.loc[index, "p_bh_within_analysis_set"] = bh_adjust(values)
    return results


def compare(
    dataset: str,
    setting: str,
    analysis_set: str,
    family: str,
    surrogate: pd.DataFrame,
    baseline: pd.DataFrame,
    method_a_name: str,
    baseline_name: str,
    replicates: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    if not surrogate.index.equals(baseline.index):
        raise ValueError(
            f"{dataset}/{setting}/{baseline_name}: page alignment mismatch; "
            f"only Surrogate={len(surrogate.index.difference(baseline.index))}, "
            f"only baseline={len(baseline.index.difference(surrogate.index))}"
        )
    surrogate_column, baseline_column = surrogate.columns[0], baseline.columns[0]
    rows: List[Dict[str, object]] = []
    for engine in surrogate.index.get_level_values("engine").unique():
        mask = surrogate.index.get_level_values("engine") == engine
        a = surrogate.loc[mask, surrogate_column].to_numpy(dtype=float)
        b = baseline.loc[mask, baseline_column].to_numpy(dtype=float)
        diff = a - b
        statistic, p_value = wilcoxon_two_sided(a, b)
        ci_low, ci_high = bootstrap_ci(diff, replicates, rng)
        n_zero_diff = int((diff == 0).sum())
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "engine": engine,
                "k": K,
                "analysis_set": analysis_set,
                "comparison_family": family,
                "method_a": method_a_name,
                "method_b": baseline_name,
                "n_pages": int(len(diff)),
                "n_zero_diff": n_zero_diff,
                "n_effective_wilcoxon": int(len(diff)) - n_zero_diff,
                "surrogate_mean_cer_pct": float(100 * a.mean()),
                "baseline_mean_cer_pct": float(100 * b.mean()),
                "mean_diff_surrogate_minus_baseline_pp": float(100 * diff.mean()),
                "median_diff_surrogate_minus_baseline_pp": float(100 * np.median(diff)),
                "bootstrap_95ci_low_pp": float(100 * ci_low),
                "bootstrap_95ci_high_pp": float(100 * ci_high),
                "surrogate_wins": int((diff < 0).sum()),
                "ties": n_zero_diff,
                "baseline_wins": int((diff > 0).sum()),
                "wilcoxon_statistic": statistic,
                "p_two_sided": p_value,
                "direction_from_mean": "surrogate_lower" if diff.mean() < 0 else "baseline_lower" if diff.mean() > 0 else "tie",
            }
        )
    return rows


def write_report(results: pd.DataFrame, path: Path, replicates: int, seed: int) -> None:
    lines = [
        "POLICY-MATCHED TOP-3 NESTED SELECTOR COMPARISON",
        "=" * 72,
        "Difference is Surrogate minus baseline in percentage-point CER; negative values favor Surrogate.",
        "Tests are paired, two-sided Wilcoxon signed-rank tests.",
        f"Bootstrap CIs use {replicates} percentile resamples with seed {seed}.",
        "Holm/BH are reported within comparator family and within the analysis set.",
        "",
    ]
    for family, group in results.groupby("comparison_family", sort=False):
        lines.extend([family, "-" * len(family)])
        for _, row in group.iterrows():
            lines.append(
                f"{row['dataset']} / {row['setting']} / {row['engine']}: "
                f"{row['method_a']}={row['surrogate_mean_cer_pct']:.3f}%, {row['method_b']}={row['baseline_mean_cer_pct']:.3f}%, "
                f"diff={row['mean_diff_surrogate_minus_baseline_pp']:+.3f} pp, "
                f"95% CI [{row['bootstrap_95ci_low_pp']:+.3f}, {row['bootstrap_95ci_high_pp']:+.3f}], "
                f"n={row['n_pages']} (eff={row['n_effective_wilcoxon']}, ties={row['n_zero_diff']}), "
                f"p={row['p_two_sided']:.3g}, Holm(family)={row['p_holm_within_comparator']:.3g}, "
                f"BH(family)={row['p_bh_within_comparator']:.3g}, "
                f"Holm(analysis)={row['p_holm_within_analysis_set']:.3g}, "
                f"BH(analysis)={row['p_bh_within_analysis_set']:.3g}"
            )
        lines.append("")
    lines.append("SROIE NN-K is unavailable because exact historical shortlist provenance is incomplete for three EasyOCR pages.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    feature_paths = (args.feature_matched_raw, args.feature_matched_contour)
    if any(path is not None for path in feature_paths) and not all(path is not None for path in feature_paths):
        raise ValueError("Provide both --feature-matched-raw and --feature-matched-contour, or neither.")

    inputs = {
        "surrogate_raw": args.surrogate_raw,
        "surrogate_contour": args.surrogate_contour,
        "surrogate_sroie": args.surrogate_sroie,
        "nn_raw": args.nn_raw,
        "nn_contour": args.nn_contour,
        "fixed_raw": args.fixed_raw,
        "fixed_contour": args.fixed_contour,
        "fixed_sroie": args.fixed_sroie,
    }
    selected = {label: load_selected(path, label) for label, path in inputs.items()}
    feature_selected: Dict[str, pd.DataFrame] = {}
    if all(path is not None for path in feature_paths):
        feature_selected.update(
            {
                "feature_matched_raw": load_selected(args.feature_matched_raw, "feature_matched_raw"),
                "feature_matched_contour": load_selected(args.feature_matched_contour, "feature_matched_contour"),
            }
        )
    if args.feature_matched_sroie is not None:
        feature_selected["feature_matched_sroie"] = load_selected(
            args.feature_matched_sroie, "feature_matched_sroie"
        )

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    primary = "primary_policy_comparison"
    rows.extend(compare("Vietnamese", "raw", primary, "surrogate_vs_nn_k", selected["surrogate_raw"], selected["nn_raw"], "Surrogate", "NN-K", args.bootstrap_replicates, rng))
    rows.extend(compare("Vietnamese", "contour", primary, "surrogate_vs_nn_k", selected["surrogate_contour"], selected["nn_contour"], "Surrogate", "NN-K", args.bootstrap_replicates, rng))
    rows.extend(compare("Vietnamese", "raw", primary, "surrogate_vs_fixed_topk", selected["surrogate_raw"], selected["fixed_raw"], "Surrogate", "FixedTopK", args.bootstrap_replicates, rng))
    rows.extend(compare("Vietnamese", "contour", primary, "surrogate_vs_fixed_topk", selected["surrogate_contour"], selected["fixed_contour"], "Surrogate", "FixedTopK", args.bootstrap_replicates, rng))
    rows.extend(compare("SROIE", "receipt", primary, "surrogate_vs_fixed_topk", selected["surrogate_sroie"], selected["fixed_sroie"], "Surrogate", "FixedTopK", args.bootstrap_replicates, rng))

    if feature_selected:
        sensitivity = "feature_matched_sensitivity"
        if "feature_matched_raw" in feature_selected:
            rows.extend(compare("Vietnamese", "raw", sensitivity, "feature_matched_surrogate_vs_nn_k", feature_selected["feature_matched_raw"], selected["nn_raw"], "Feature-matched Surrogate", "NN-K", args.bootstrap_replicates, rng))
            rows.extend(compare("Vietnamese", "raw", sensitivity, "feature_matched_surrogate_vs_fixed_topk", feature_selected["feature_matched_raw"], selected["fixed_raw"], "Feature-matched Surrogate", "FixedTopK", args.bootstrap_replicates, rng))
        if "feature_matched_contour" in feature_selected:
            rows.extend(compare("Vietnamese", "contour", sensitivity, "feature_matched_surrogate_vs_nn_k", feature_selected["feature_matched_contour"], selected["nn_contour"], "Feature-matched Surrogate", "NN-K", args.bootstrap_replicates, rng))
            rows.extend(compare("Vietnamese", "contour", sensitivity, "feature_matched_surrogate_vs_fixed_topk", feature_selected["feature_matched_contour"], selected["fixed_contour"], "Feature-matched Surrogate", "FixedTopK", args.bootstrap_replicates, rng))
        if "feature_matched_sroie" in feature_selected:
            rows.extend(compare("SROIE", "receipt", sensitivity, "feature_matched_surrogate_vs_fixed_topk", feature_selected["feature_matched_sroie"], selected["fixed_sroie"], "Feature-matched Surrogate", "FixedTopK", args.bootstrap_replicates, rng))

    results = add_adjustments(pd.DataFrame(rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "policy_selector_paired_comparison.csv"
    report_path = args.output_dir / "policy_selector_paired_report.txt"
    metadata_path = args.output_dir / "run_metadata.json"
    results.to_csv(csv_path, index=False)
    write_report(results, report_path, args.bootstrap_replicates, args.seed)
    metadata_path.write_text(
        json.dumps(
            {
                "k": K,
                "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "inputs": {label: str(path) for label, path in inputs.items()},
                "feature_matched_inputs": {
                    label: str(path)
                    for label, path in {"raw": args.feature_matched_raw, "contour": args.feature_matched_contour, "sroie": args.feature_matched_sroie}.items()
                    if path is not None
                },
                "comparisons": int(len(results)),
                "note": "SROIE NN-K is omitted because exact historical shortlist provenance is incomplete.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Validated and compared {len(results)} policy/engine rows.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
