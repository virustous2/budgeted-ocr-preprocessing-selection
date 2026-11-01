#!/usr/bin/env python3
"""Recompute raw-Vietnamese Surrogate--FixedTopK zero-method sensitivity.

Uses public held-out per-page tables only; no OCR or model fitting. Each zero
method is Holm-adjusted across the same 12 engine-by-budget comparisons.
The primary manuscript convention remains wilcox.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from recompute_two_sided_wilcoxon_tables import holm_adjust


ENGINES = ("vietocr", "tesseract", "easyocr")
BUDGETS = (3, 5, 10, 15)
ZERO_METHODS = ("wilcox", "pratt", "zsplit")


def calculate(release_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for engine in ENGINES:
        source = release_root / "results/per_page_bestk/raw" / f"per_page_results_{engine}.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            pages = list(csv.DictReader(handle))
        if len(pages) != 300 or len({page["image_name"] for page in pages}) != 300:
            raise ValueError(f"Expected 300 unique raw Vietnamese pages in {source}")
        for budget in BUDGETS:
            surrogate = np.asarray([float(page[f"surr_k{budget}"]) for page in pages])
            fixed = np.asarray([float(page[f"fixed_topk_k{budget}"]) for page in pages])
            difference = surrogate - fixed
            if not np.isfinite(difference).all():
                raise ValueError(f"Non-finite CER in {source}, K={budget}")
            for zero_method in ZERO_METHODS:
                p_value = float(
                    wilcoxon(surrogate, fixed, alternative="two-sided", zero_method=zero_method).pvalue
                )
                rows.append(
                    {
                        "setting": "raw",
                        "engine": engine,
                        "k": budget,
                        "zero_method": zero_method,
                        "n_pages": len(pages),
                        "n_zero_diff": int(np.count_nonzero(difference == 0)),
                        "mean_diff_surrogate_minus_fixedtopk_pp": float(100 * np.mean(difference)),
                        "p_two_sided": p_value,
                    }
                )

    if len(rows) != 36:
        raise AssertionError(f"Expected 36 sensitivity rows, got {len(rows)}")
    for zero_method in ZERO_METHODS:
        subset = [row for row in rows if row["zero_method"] == zero_method]
        adjusted = holm_adjust([float(row["p_two_sided"]) for row in subset])
        for row, p_holm in zip(subset, adjusted):
            row["p_holm_12"] = p_holm
            row["significant_holm_0_05"] = p_holm < 0.05
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release_root = args.release_root.resolve()
    output = args.output or release_root / (
        "results/statistical_tests/wilcoxon_zero_sensitivity_raw_fixedtopk_20260924.csv"
    )
    rows = calculate(release_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sensitivity rows to {output}")


if __name__ == "__main__":
    main()
