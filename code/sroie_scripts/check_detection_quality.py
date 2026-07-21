#!/usr/bin/env python3
"""
Detection quality check: PaddleOCR (lang=en) vs SROIE GT bounding boxes.

For each image, computes precision and recall at IoU >= 0.5 between
PaddleOCR detected polygons and SROIE GT quadrilaterals.

GT format (gt_boxes/*.txt): x1,y1,x2,y2,x3,y3,x4,y4,text  (one line per box)
PaddleOCR output: dt_polys — list of quadrilateral arrays

Output: SROIRE_COMPARISON/output_sroire/detection_quality/
    - detection_quality_TIMESTAMP.csv  (per-image precision, recall, F1)
    - summary_TIMESTAMP.txt

Usage:
    python sroie_scripts/check_detection_quality.py
    python sroie_scripts/check_detection_quality.py --device cuda
"""

import os
import sys
import csv
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
SROIE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SROIE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGES   = str(SROIE_ROOT / 'sroie_dataset/images')
DEFAULT_GT_BOXES = str(SROIE_ROOT / 'sroie_dataset/gt_boxes')
DEFAULT_OUTPUT   = str(SROIE_ROOT / 'output_sroire/detection_quality')
IOU_THRESHOLD    = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_gt_boxes(gt_path: str) -> List[np.ndarray]:
    """Parse SROIE gt_boxes format: x1,y1,x2,y2,x3,y3,x4,y4,text"""
    boxes = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 8)
            if len(parts) < 8:
                continue
            try:
                coords = list(map(float, parts[:8]))
                poly = np.array(coords).reshape(4, 2)
                boxes.append(poly)
            except ValueError:
                continue
    return boxes


def poly_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute IoU between two quadrilaterals using Shapely."""
    try:
        s1 = Polygon(p1).buffer(0)
        s2 = Polygon(p2).buffer(0)
        if not s1.is_valid or not s2.is_valid:
            return 0.0
        inter = s1.intersection(s2).area
        union = s1.union(s2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_precision_recall(
    det_polys: List[np.ndarray],
    gt_polys: List[np.ndarray],
    iou_thr: float = 0.5
) -> Tuple[float, float, float]:
    """
    Greedy matching: each GT matched to at most one detection.
    Returns (precision, recall, f1).
    """
    if not gt_polys:
        return (1.0, 1.0, 1.0) if not det_polys else (0.0, 1.0, 0.0)
    if not det_polys:
        return (0.0, 0.0, 0.0)

    matched_gt  = set()
    matched_det = set()

    # Build IoU matrix
    iou_matrix = np.zeros((len(det_polys), len(gt_polys)))
    for i, d in enumerate(det_polys):
        for j, g in enumerate(gt_polys):
            iou_matrix[i, j] = poly_iou(d, g)

    # Greedy match: highest IoU first
    pairs = sorted(
        [(iou_matrix[i, j], i, j)
         for i in range(len(det_polys))
         for j in range(len(gt_polys))],
        reverse=True
    )
    for iou_val, i, j in pairs:
        if iou_val < iou_thr:
            break
        if i not in matched_det and j not in matched_gt:
            matched_det.add(i)
            matched_gt.add(j)

    tp        = len(matched_det)
    precision = tp / len(det_polys) if det_polys else 0.0
    recall    = tp / len(gt_polys)  if gt_polys  else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ── main ──────────────────────────────────────────────────────────────────────

def run(images_folder: str, gt_boxes_folder: str, output_folder: str, device: str,
        n_images: int = None, seed: int = 42):
    os.makedirs(output_folder, exist_ok=True)

    print("Loading PaddleOCR (detection only, lang=en)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',
        device='gpu' if device == 'cuda' else 'cpu',
    )

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and '(' not in f
    ])

    if n_images is not None and n_images < len(image_files):
        import random
        rng = random.Random(seed)
        image_files = rng.sample(image_files, n_images)
        image_files = sorted(image_files)
        print(f"Subsampled to {len(image_files)} images (seed={seed})")

    print(f"Processing {len(image_files)} images...\n")

    rows = []
    for fname in tqdm(image_files, desc="Detection quality"):
        img_path = os.path.join(images_folder, fname)
        gt_path  = os.path.join(gt_boxes_folder, fname.replace('.jpg', '.txt')
                                                       .replace('.jpeg', '.txt')
                                                       .replace('.png', '.txt'))
        if not os.path.exists(gt_path):
            continue

        gt_polys = parse_gt_boxes(gt_path)
        if not gt_polys:
            continue

        results  = ocr.predict(input=img_path)
        det_polys = []
        if results and results[0]:
            for poly in results[0].get('dt_polys', []):
                arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
                if arr.shape[0] >= 3:
                    det_polys.append(arr)

        precision, recall, f1 = compute_precision_recall(
            det_polys, gt_polys, IOU_THRESHOLD)

        rows.append({
            'image':     fname,
            'n_det':     len(det_polys),
            'n_gt':      len(gt_polys),
            'ratio':     round(len(det_polys) / len(gt_polys), 3) if gt_polys else 0,
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
        })

    if not rows:
        print("No results.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path     = os.path.join(output_folder, f'detection_quality_{timestamp}.csv')
    summary_path = os.path.join(output_folder, f'summary_{timestamp}.txt')

    df.to_csv(csv_path, index=False)

    mean_p  = df['precision'].mean()
    mean_r  = df['recall'].mean()
    mean_f1 = df['f1'].mean()
    mean_ratio = df['ratio'].mean()

    lines = [
        "=" * 60,
        "DETECTION QUALITY — PaddleOCR (lang=en) vs SROIE GT Boxes",
        f"Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Images:        {len(df)}",
        f"IoU threshold: {IOU_THRESHOLD}",
        "=" * 60,
        f"  Mean Precision : {mean_p*100:.2f}%",
        f"  Mean Recall    : {mean_r*100:.2f}%",
        f"  Mean F1        : {mean_f1*100:.2f}%",
        f"  Mean Ratio     : {mean_ratio:.3f}  (n_det / n_gt)",
        "",
        f"  Precision >= 0.8 : {(df['precision']>=0.8).sum()} / {len(df)} images",
        f"  Recall    >= 0.8 : {(df['recall']>=0.8).sum()} / {len(df)} images",
        f"  F1        >= 0.8 : {(df['f1']>=0.8).sum()} / {len(df)} images",
        "",
        f"  Worst 5 images by F1:",
    ]
    worst = df.nsmallest(5, 'f1')[['image', 'n_det', 'n_gt', 'precision', 'recall', 'f1']]
    for _, r in worst.iterrows():
        lines.append(
            f"    {r['image']:<35} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
            f"(det={int(r['n_det'])}, gt={int(r['n_gt'])})"
        )
    lines += ["", f"CSV: {csv_path}", "=" * 60]

    summary = "\n".join(lines)
    with open(summary_path, 'w') as f:
        f.write(summary + "\n")
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(
        description='Detection quality: PaddleOCR vs SROIE GT boxes (IoU@0.5)')
    parser.add_argument('--images-folder',   default=DEFAULT_IMAGES)
    parser.add_argument('--gt-boxes-folder', default=DEFAULT_GT_BOXES)
    parser.add_argument('--output-folder',   default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--n-images', type=int, default=None,
                        help='Limit to N images using same selection as collect_lhs_sroie.py (seed-based)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run(args.images_folder, args.gt_boxes_folder, args.output_folder, args.device,
        n_images=args.n_images, seed=args.seed)


if __name__ == '__main__':
    main()
