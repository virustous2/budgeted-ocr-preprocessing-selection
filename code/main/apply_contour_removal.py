#!/usr/bin/env python3
"""
Apply contour removal (full-page noise suppression) to all 300 images.
Output mirrors the same year-based folder structure.

Contour removal logic from OCR_backend/preprocessing.py:
  - GaussianBlur(5,5) + Otsu threshold → find large rectangular blobs
  - Fill white on original color image (does NOT binarize output)
  - Filter: area > 500, 1.0 < aspect < 5.0, w > 80, h > 30

Usage:
    python main/apply_contour_removal.py
    python main/apply_contour_removal.py \
        --input-dir data/source_documents_not_included/vietnamese/images \
        --output-dir results/recomputed/contour_suppressed_images
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent

DEFAULT_INPUT = str(RELEASE_ROOT / 'data/source_documents_not_included/vietnamese/images')
DEFAULT_OUTPUT = str(RELEASE_ROOT / 'results/recomputed/contour_suppressed_images')


# ── contour removal ───────────────────────────────────────────────────────────

def contour_removal(image: np.ndarray) -> np.ndarray:
    """
    Fill white on large rectangular blobs in full-page image.
    Targets noise regions (stamps, watermarks) that cause false detections.
    Does NOT binarize — output is original color image with some regions whited out.

    Filter criteria (updated: area=500 based on visualization):
        area > 500, 1.0 < aspect_ratio < 5.0, w > 80, h > 30
    """
    out  = image.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    n_removed = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area   = cv2.contourArea(cnt)
        aspect = w / h if h != 0 else 0
        if area > 500 and 1.0 < aspect < 5.0 and w > 80 and h > 30:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), -1)
            n_removed += 1
    return out, n_removed


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_dir: str, output_dir: str):
    input_root  = Path(input_dir)
    output_root = Path(output_dir)

    # Collect all images preserving year structure
    image_pairs = []
    for year_dir in sorted(input_root.iterdir()):
        if not year_dir.is_dir():
            continue
        out_year = output_root / year_dir.name
        out_year.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(year_dir.glob('*.png')):
            image_pairs.append((img_path, out_year / img_path.name))

    print(f"Found {len(image_pairs)} images")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}\n")

    stats = {'processed': 0, 'failed': 0, 'total_removed': 0}

    for src, dst in tqdm(image_pairs, desc="Applying contour removal"):
        img = cv2.imread(str(src))
        if img is None:
            print(f"  WARNING: cannot read {src}")
            stats['failed'] += 1
            continue

        result, n_removed = contour_removal(img)
        cv2.imwrite(str(dst), result)
        stats['processed'] += 1
        stats['total_removed'] += n_removed

    # Summary
    summary = [
        "=" * 55,
        "CONTOUR REMOVAL — DONE",
        f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Processed: {stats['processed']} images",
        f"Failed   : {stats['failed']} images",
        f"Total contours removed: {stats['total_removed']}",
        f"Avg per image: {stats['total_removed']/max(stats['processed'],1):.1f}",
        "=" * 55,
        f"Output: {output_root}",
    ]
    print("\n" + "\n".join(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply contour removal to all images, preserving folder structure')
    parser.add_argument('--input-dir',  default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
