#!/usr/bin/env python3
"""Report-label checks for the NN-K baseline (no OCR, no model fitting).

Builds a synthetic per-image table, runs the real summarize_engine/write_report
path with and without paired Surrogate columns, and checks that every reported
Wilcoxon p-value is two-sided and labelled non-directionally.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

import evaluate_nn_shortlist_baseline as nn  # noqa: E402

K_VALUES = [3, 5]
DIRECTIONAL = re.compile(r"\b(NN|Surrogate|FixedTopK|Random)\s*[<>]\s*(NN|Surrogate|FixedTopK|Random)\b")


def synthetic_per_image(with_surrogate: bool) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 40
    frame = pd.DataFrame({"image_name": [f"p{i:03d}" for i in range(n)],
                          "samporacle_cer": rng.uniform(0.05, 0.2, n)})
    for k in K_VALUES:
        base = rng.uniform(0.1, 0.5, n)
        frame[f"nn_k{k}_cer"] = base
        frame[f"fixed_topk_k{k}_cer"] = base + rng.normal(0.01, 0.02, n)
        frame[f"random_k{k}_cer"] = base + rng.normal(0.03, 0.02, n)
        if with_surrogate:
            frame[f"surr_k{k}_cer"] = base + rng.normal(-0.005, 0.02, n)
    return frame


def render(with_surrogate: bool) -> tuple[str, dict]:
    per_image = synthetic_per_image(with_surrogate)
    summary = nn.summarize_engine(per_image, "tesseract", K_VALUES)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        nn.write_report({"tesseract": summary}, out, K_VALUES, None, Path("synthetic.csv"),
                        ["orig_blur"], "euclidean", has_compare_per_page=with_surrogate)
        return (out / "nn_report.txt").read_text(encoding="utf-8"), summary


def check(with_surrogate: bool) -> int:
    text, summary = render(with_surrogate)
    if "Wilcoxon two-sided paired tests" not in text:
        raise AssertionError("Missing two-sided report header")
    if re.search(r"one-sided|first method has lower", text):
        raise AssertionError("Stale one-sided wording in report")
    bad = DIRECTIONAL.findall(text)
    if bad:
        raise AssertionError(f"Directional p-value labels remain: {bad}")
    for name, test in summary["wilcoxon"].items():
        if test.get("alternative") != "two-sided":
            raise AssertionError(f"{name} is not two-sided: {test}")
    per_image = synthetic_per_image(with_surrogate)
    pairs = {"nn_vs_fixed_topk": ("nn", "fixed_topk"), "nn_vs_random": ("nn", "random")}
    if with_surrogate:
        pairs.update({"nn_vs_surrogate": ("nn", "surr"), "surrogate_vs_nn": ("surr", "nn")})
        for label in ("NN vs Surrogate p=", "Surrogate vs NN p="):
            if label not in text:
                raise AssertionError(f"Missing Surrogate-branch label: {label}")
    elif "Surrogate p=" in text:
        raise AssertionError("Surrogate p-values printed without paired Surrogate data")
    checked = 0
    for k in K_VALUES:
        for key, (a, b) in pairs.items():
            expected = wilcoxon(per_image[f"{a}_k{k}_cer"], per_image[f"{b}_k{k}_cer"],
                                alternative="two-sided").pvalue
            got = summary["wilcoxon"][f"{key}_k{k}"]["p"]
            if not np.isclose(got, expected, rtol=0.0, atol=1e-15):
                raise AssertionError(f"{key}_k{k}: p={got}, scipy two-sided={expected}")
            checked += 1
    return checked


def main() -> None:
    with_surr = check(with_surrogate=True)
    without_surr = check(with_surrogate=False)
    print(f"PASS: report labels two-sided/non-directional; "
          f"{with_surr} p-values with Surrogate branch, {without_surr} without")


if __name__ == "__main__":
    main()
