#!/usr/bin/env python3
"""Recompute two-sided paired Wilcoxon tests from public per-page outputs.

The original experimental scripts evolved during manuscript revision. This helper
is the authoritative public-release script for the two-sided p-values reported in
the current manuscript. It uses only sanitized, per-page result tables included in
this release and writes a single CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


K_VALUES = (3, 5, 10, 15)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def as_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilcoxon_two_sided(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float, int]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return math.nan, math.nan, 0
    if np.allclose(x, y):
        return math.nan, 1.0, int(len(x))
    stat, p_value = wilcoxon(x, y, alternative="two-sided")
    return float(stat), float(p_value), int(len(x))


def add_test(
    rows: List[Dict[str, object]],
    dataset: str,
    setting: str,
    family: str,
    engine: str,
    k: int | str,
    method_a: str,
    values_a: Sequence[float],
    method_b: str,
    values_b: Sequence[float],
) -> None:
    stat, p_value, n = wilcoxon_two_sided(values_a, values_b)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b
    mean_diff = float(np.mean(diff)) if len(diff) else math.nan
    rows.append(
        {
            "dataset": dataset,
            "setting": setting,
            "comparison_family": family,
            "engine": engine,
            "k": k,
            "method_a": method_a,
            "method_b": method_b,
            "n": n,
            "method_a_mean_cer_pct": float(100 * np.mean(a)) if len(a) else math.nan,
            "method_b_mean_cer_pct": float(100 * np.mean(b)) if len(b) else math.nan,
            "mean_diff_a_minus_b_pp": float(100 * mean_diff) if len(diff) else math.nan,
            "direction_from_mean": "a_lower" if mean_diff < 0 else "b_lower" if mean_diff > 0 else "tie",
            "wilcoxon_statistic": stat,
            "p_two_sided": p_value,
            "alternative": "two-sided",
        }
    )


def collect_main_bestk(root: Path, out_rows: List[Dict[str, object]]) -> None:
    specs = [
        ("Vietnamese", "raw", root / "results/per_page_bestk/raw", ("vietocr", "tesseract", "easyocr")),
        ("Vietnamese", "contour", root / "results/per_page_bestk/contour", ("vietocr", "tesseract", "easyocr")),
        ("SROIE", "raw", root / "results/per_page_bestk/sroie", ("tesseract", "easyocr")),
    ]
    for dataset, setting, folder, engines in specs:
        for engine in engines:
            rows = read_rows(folder / f"per_page_results_{engine}.csv")
            for k in K_VALUES:
                surr = [as_float(r[f"surr_k{k}"]) for r in rows]
                rand = [as_float(r[f"rand_k{k}"]) for r in rows]
                fixed_topk = [as_float(r[f"fixed_topk_k{k}"]) for r in rows]
                add_test(out_rows, dataset, setting, "best_of_k", engine, k, "Surrogate", surr, "Random", rand)
                add_test(out_rows, dataset, setting, "best_of_k", engine, k, "Surrogate", surr, "FixedTopK", fixed_topk)


def collect_nn(root: Path, out_rows: List[Dict[str, object]]) -> None:
    specs = [
        ("Vietnamese", "raw", root / "results/nn_baseline/raw"),
        ("Vietnamese", "contour", root / "results/nn_baseline/contour"),
        ("SROIE", "raw", root / "results/nn_baseline/sroie"),
    ]
    for dataset, setting, folder in specs:
        rows = read_rows(folder / "nn_per_image.csv")
        engines = sorted({r["engine"] for r in rows})
        for engine in engines:
            erows = [r for r in rows if r["engine"] == engine]
            for k in K_VALUES:
                nn = [as_float(r[f"nn_k{k}_cer"]) for r in erows]
                rand = [as_float(r[f"random_k{k}_cer"]) for r in erows]
                fixed_topk = [as_float(r[f"fixed_topk_k{k}_cer"]) for r in erows]
                add_test(out_rows, dataset, setting, "nn_best_of_k", engine, k, "NN", nn, "Random", rand)
                add_test(out_rows, dataset, setting, "nn_best_of_k", engine, k, "NN", nn, "FixedTopK", fixed_topk)
                surr_col = f"surr_k{k}_cer"
                if surr_col in erows[0]:
                    surr = [as_float(r[surr_col]) for r in erows]
                    add_test(out_rows, dataset, setting, "nn_best_of_k", engine, k, "NN", nn, "Surrogate", surr)


def selector_page_values(rows: List[Dict[str, str]], engine: str) -> Dict[str, List[float]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row["engine"] == engine:
            groups.setdefault(row["image_name"], []).append(row)

    values = {
        "Selector": [],
        "Top1Pred": [],
        "Agreement": [],
        "FixedBest": [],
        "NoPre": [],
        "BestOf3": [],
    }
    for image_name in sorted(groups):
        grows = groups[image_name]
        selected = next((r for r in grows if as_bool(r.get("selected_by_nested_model"))), None)
        top1 = next((r for r in grows if int(float(r["rank"])) == 1), None)
        agreement = next((r for r in grows if as_bool(r.get("agreement_selected"))), None)
        if selected is None or top1 is None:
            continue
        if agreement is None:
            agreement = top1
        values["Selector"].append(as_float(selected["actual_cer"]))
        values["Top1Pred"].append(as_float(top1["actual_cer"]))
        values["Agreement"].append(as_float(agreement["actual_cer"]))
        values["FixedBest"].append(as_float(grows[0]["fixedbest_actual_cer"]))
        values["NoPre"].append(as_float(grows[0]["nopre_cer"]))
        values["BestOf3"].append(min(as_float(r["actual_cer"]) for r in grows))
    return values


def collect_selector(root: Path, out_rows: List[Dict[str, object]]) -> None:
    specs = [
        ("Vietnamese", "raw", root / "results/oracle_free_selector/raw", ("vietocr", "tesseract", "easyocr")),
        ("Vietnamese", "contour", root / "results/oracle_free_selector/contour", ("vietocr", "tesseract", "easyocr")),
        ("SROIE", "raw", root / "results/oracle_free_selector/sroie", ("tesseract", "easyocr")),
    ]
    for dataset, setting, folder, engines in specs:
        rows = read_rows(folder / "nested_selector_candidate_scores_sanitized.csv")
        for engine in engines:
            vals = selector_page_values(rows, engine)
            for baseline in ("Top1Pred", "Agreement", "FixedBest", "NoPre", "BestOf3"):
                add_test(
                    out_rows,
                    dataset,
                    setting,
                    "oracle_free_top3_selector",
                    engine,
                    3,
                    "Selector",
                    vals["Selector"],
                    baseline,
                    vals[baseline],
                )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "setting",
        "comparison_family",
        "engine",
        "k",
        "method_a",
        "method_b",
        "n",
        "method_a_mean_cer_pct",
        "method_b_mean_cer_pct",
        "mean_diff_a_minus_b_pp",
        "direction_from_mean",
        "wilcoxon_statistic",
        "p_two_sided",
        "alternative",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to results/statistical_tests/two_sided_wilcoxon_summary.csv.",
    )
    args = parser.parse_args()

    root = args.release_root.resolve()
    output = args.output or root / "results/statistical_tests/two_sided_wilcoxon_summary.csv"

    rows: List[Dict[str, object]] = []
    collect_main_bestk(root, rows)
    collect_nn(root, rows)
    collect_selector(root, rows)
    write_csv(output, rows)
    print(f"Wrote {len(rows)} two-sided Wilcoxon tests to {output}")


if __name__ == "__main__":
    main()
