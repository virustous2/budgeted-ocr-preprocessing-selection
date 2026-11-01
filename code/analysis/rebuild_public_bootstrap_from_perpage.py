#!/usr/bin/env python3
"""Rebuild public paired-bootstrap summaries from current per-page CSVs only.

No OCR, model training, or access to private reference text is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_shortlist_ci import summarize  # noqa: E402

K_VALUES = (3, 5, 10, 15)
SETTINGS = {
    "raw": ("vietocr", "tesseract", "easyocr"),
    "contour": ("vietocr", "tesseract", "easyocr"),
    "sroie": ("tesseract", "easyocr"),
}


def load_per_image(release_root: Path, setting: str, engines: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for engine in engines:
        source = release_root / "results" / "per_page_bestk" / setting / f"per_page_results_{engine}.csv"
        frame = pd.read_csv(source)
        expected_pages = 616 if setting == "sroie" else 300
        if len(frame) != expected_pages or frame["image_name"].nunique() != expected_pages:
            raise ValueError(f"{source}: expected {expected_pages} unique pages")
        for k in K_VALUES:
            cols = [f"surr_k{k}", f"rand_k{k}", f"fixed_topk_k{k}"]
            if frame[cols].isna().any().any():
                raise ValueError(f"{source}: missing paired CER value at K={k}")
            for _, row in frame.iterrows():
                surr, rand, fixed = (float(row[col]) for col in cols)
                rows.append({
                    "engine": engine,
                    "image_name": row["image_name"],
                    "k": k,
                    "surr_cer": surr,
                    "rand_cer": rand,
                    "fixed_topk_cer": fixed,
                    "diff_vs_random": surr - rand,
                    "diff_vs_fixed_topk": surr - fixed,
                })
    return pd.DataFrame(rows)


def main() -> None:
    release_root = PROJECT_ROOT.parent if (PROJECT_ROOT.parent / "results").exists() else PROJECT_ROOT / "ijdar_public_release"
    target_dir = release_root / "results" / "bootstrap_ci"
    for setting, engines in SETTINGS.items():
        per_image = load_per_image(release_root, setting, engines)
        summary = summarize(per_image, n_bootstrap=10000, seed=12345, alpha=0.05, ci_method="percentile")
        target = target_dir / f"{setting}_fixedtopk_bootstrap_summary.csv"
        summary.to_csv(target, index=False)
        print(f"{setting}: {len(per_image)} page-engine-budget rows, {len(summary)} summary rows -> {target}")


if __name__ == "__main__":
    main()
