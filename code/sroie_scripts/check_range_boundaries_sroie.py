#!/usr/bin/env python3
"""
Boundary check for SROIE dataset: test in-range vs out-of-range preprocessing values.

Adapted from analysis/check_range_boundaries.py for SROIE:
  - GT format: plain .txt (not .docx)
  - Flat image structure (no year subfolders)
  - Engines: tesseract (lang=eng) + easyocr (lang=en) only
  - PaddleOCR: lang=en
  - Image selection: random (no noprec_csv filtering)

Output: results/recomputed/boundary_check_sroie/
    - boundary_check_TIMESTAMP.csv
    - summary_TIMESTAMP.txt

Usage:
    python sroie_scripts/check_range_boundaries_sroie.py
    python sroie_scripts/check_range_boundaries_sroie.py --n-images 20 --device cuda
"""

import os
import sys
import csv
import argparse
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

DEFAULT_IMAGES = str(RELEASE_ROOT / 'data/source_documents_not_included/sroie/images')
DEFAULT_LABELS = str(RELEASE_ROOT / 'data/source_documents_not_included/sroie/ground_truth')
DEFAULT_OUTPUT = str(RELEASE_ROOT / 'results/recomputed/boundary_check_sroie')

# ── parameter test configs (same ranges as Vietnamese pipeline) ──────────────
PARAM_TESTS = {
    'gaussian_ksize': [
        (0,  'noop'),
        (3,  'in-range'),
        (5,  'in-range'),
        (7,  'boundary'),
        (9,  'out-of-range'),
        (11, 'out-of-range'),
    ],
    'bilateral_d': [
        (0,  'noop'),
        (5,  'in-range'),
        (7,  'in-range'),
        (9,  'boundary'),
        (11, 'out-of-range'),
        (13, 'out-of-range'),
    ],
    'clahe_clip': [
        (0.0,  'noop'),
        (1.0,  'in-range'),
        (2.0,  'in-range'),
        (4.0,  'boundary'),
        (5.5,  'out-of-range'),
        (7.0,  'out-of-range'),
        (8.0,  'out-of-range'),
        (10.0, 'out-of-range'),
    ],
    'gamma': [
        (1.0, 'noop'),
        (0.8, 'in-range'),
        (1.2, 'in-range'),
        (0.5, 'boundary-low'),
        (1.5, 'boundary-high'),
        (0.3, 'out-of-range'),
        (1.8, 'out-of-range'),
    ],
    'sharpen': [
        (0.0, 'noop'),
        (0.5, 'in-range'),
        (1.0, 'in-range'),
        (2.0, 'boundary'),
        (3.0, 'out-of-range'),
        (4.0, 'out-of-range'),
    ],
}

NOOP_PARAMS = {
    'gaussian_ksize': 0,
    'bilateral_d':    0,
    'clahe_clip':     0.0,
    'gamma':          1.0,
    'sharpen':        0.0,
}

CHOSEN_BOUNDS = {
    'gaussian_ksize': (0, 7),
    'bilateral_d':    (0, 9),
    'clahe_clip':     (0, 4.0),
    'gamma':          (0.5, 1.5),
    'sharpen':        (0, 2.0),
}


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image: np.ndarray, params: Dict) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    ksize = int(params['gaussian_ksize'])
    if ksize > 0:
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    bilateral_d = int(params['bilateral_d'])
    if bilateral_d > 0:
        gray = cv2.bilateralFilter(gray, bilateral_d, 75, 75)

    clahe_clip = params['clahe_clip']
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    gamma = params['gamma']
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")
        gray = cv2.LUT(gray, table)

    sharpen = params['sharpen']
    if sharpen > 0:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.float32) * sharpen
        kernel[1, 1] = 1 + 4 * sharpen
        gray = cv2.filter2D(gray, -1, kernel)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── data helpers ──────────────────────────────────────────────────────────────

def read_txt_gt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return " ".join(lines)


def calc_cer(gt: str, pred: str) -> float:
    from jiwer import cer as jiwer_cer
    gt_n   = " ".join(gt.split())
    pred_n = " ".join(pred.split())
    if len(gt_n) == 0:
        return 1.0 if len(pred_n) > 0 else 0.0
    return float(jiwer_cer(gt_n, pred_n))


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    polygon = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    cropped      = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    return cv2.bitwise_and(cropped, cropped, mask=mask_cropped)


def collect_all_images(images_folder: str) -> List[str]:
    """Flat structure — all images in one folder."""
    items = []
    for fname in sorted(os.listdir(images_folder)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            items.append(os.path.join(images_folder, fname))
    return items


def select_images(all_images: List[str], n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    return rng.sample(all_images, min(n, len(all_images)))


# ── model init ────────────────────────────────────────────────────────────────

def load_models(engines: List[str], device: str) -> Dict:
    models = {}

    print("Loading PaddleOCR (detection, lang=en)...")
    from paddleocr import PaddleOCR
    models['paddle'] = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',
        device='gpu' if 'cuda' in device else 'cpu',
    )

    if 'tesseract' in engines:
        print("Loading Tesseract (lang=eng)...")
        import pytesseract
        langs = pytesseract.get_languages()
        if 'eng' not in langs:
            print("  WARNING: 'eng' not found in Tesseract languages.")
        models['tesseract'] = pytesseract

    if 'easyocr' in engines:
        print("Loading EasyOCR (lang=en)...")
        import easyocr
        models['easyocr'] = easyocr.Reader(['en'], gpu=('cuda' in device), verbose=False)

    return models


# ── recognition per engine ────────────────────────────────────────────────────

def recognize_crops(engine: str, model, crops: List[np.ndarray], params: Dict) -> str:
    texts = []

    if engine == 'tesseract':
        for crop in crops:
            proc = preprocess(crop, params)
            try:
                pil  = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                text = model.image_to_string(pil, lang='eng').strip()
                if text:
                    texts.append(" ".join(text.split()))
            except Exception:
                continue

    elif engine == 'easyocr':
        for crop in crops:
            proc = preprocess(crop, params)
            try:
                for text in model.readtext(proc, detail=0):
                    text = text.strip()
                    if text:
                        texts.append(text)
            except Exception:
                continue

    return "\t".join(texts)


# ── main ──────────────────────────────────────────────────────────────────────

def run(images_folder: str, labels_folder: str, output_folder: str,
        engines: List[str], n_images: int, device: str, seed: int):

    os.makedirs(output_folder, exist_ok=True)

    all_images = collect_all_images(images_folder)
    images = select_images(all_images, n_images, seed)
    print(f"\nSelected {len(images)} images. Engines: {engines}\n")

    models = load_models(engines, device)
    paddle = models['paddle']

    rows = []
    for img_path in tqdm(images, desc="Images"):
        image_name = Path(img_path).stem
        label_path = os.path.join(labels_folder, f"{image_name}.txt")
        if not os.path.exists(label_path):
            print(f"  SKIP (no label): {image_name}")
            continue

        gt_text = read_txt_gt(label_path)
        if not gt_text.strip():
            continue

        results = paddle.predict(input=img_path)
        if not results or results[0] is None:
            print(f"  SKIP (no detections): {image_name}")
            continue
        polygons = results[0].get('dt_polys', [])
        raw_image = cv2.imread(img_path)
        crops_raw = []
        for poly in polygons:
            crop = crop_polygon(raw_image, poly)
            if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                crops_raw.append(crop)

        if not crops_raw:
            continue

        for param_name, test_values in PARAM_TESTS.items():
            for value, status in test_values:
                params   = {**NOOP_PARAMS, param_name: value}
                lo, hi   = CHOSEN_BOUNDS[param_name]
                in_bound = (lo <= value <= hi)

                for eng in engines:
                    pred    = recognize_crops(eng, models[eng], crops_raw, params)
                    cer_val = calc_cer(gt_text, pred)
                    rows.append({
                        'image':     image_name,
                        'engine':    eng,
                        'parameter': param_name,
                        'value':     value,
                        'status':    status,
                        'in_bound':  in_bound,
                        'cer':       round(cer_val, 6),
                    })

    if not rows:
        print("No results collected.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(output_folder, f'boundary_check_{timestamp}.csv')
    fieldnames = ['image', 'engine', 'parameter', 'value', 'status', 'in_bound', 'cer']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)

    summary_lines = [
        "=" * 76,
        "BOUNDARY CHECK (SROIE) — Mean CER by engine, parameter, and value status",
        f"Images: {len(images)}  Engines: {engines}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Chosen bounds: {CHOSEN_BOUNDS}",
        "=" * 76, "",
    ]

    for eng in engines:
        summary_lines.append(f"{'='*30}  ENGINE: {eng.upper()}  {'='*30}")
        df_eng = df[df['engine'] == eng]
        for param in PARAM_TESTS:
            summary_lines.append(f"\n  Parameter: {param}  (bound: {CHOSEN_BOUNDS[param]})")
            summary_lines.append(f"  {'Value':>8}  {'Status':<16}  {'Mean CER':>10}")
            summary_lines.append(f"  {'-'*8}  {'-'*16}  {'-'*10}")
            sub = df_eng[df_eng['parameter'] == param].groupby(
                ['value', 'status'])['cer'].mean().reset_index()
            sub = sub.sort_values('value')
            for _, r in sub.iterrows():
                mark = ' <-- bound' if r['status'] in (
                    'boundary', 'boundary-low', 'boundary-high') else ''
                summary_lines.append(
                    f"  {r['value']:>8.1f}  {r['status']:<16}  {r['cer']*100:>9.2f}%{mark}"
                )
        summary_lines.append("")

    summary_lines.append("=" * 76)
    summary_lines.append("Aggregate (all engines): in-range vs out-of-range")
    summary_lines.append(f"  {'Engine':<12}  {'Status':<16}  {'Mean CER':>10}  {'N':>6}")
    summary_lines.append(f"  {'-'*12}  {'-'*16}  {'-'*10}  {'-'*6}")
    df_nonnoop = df[df['status'] != 'noop']
    agg = df_nonnoop.groupby(['engine', 'in_bound'])['cer'].agg(['mean', 'count']).reset_index()
    for _, r in agg.iterrows():
        label = 'in-range' if r['in_bound'] else 'out-of-range'
        summary_lines.append(
            f"  {r['engine']:<12}  {label:<16}  {r['mean']*100:>9.2f}%  {int(r['count']):>6}"
        )

    summary_lines += ["", f"CSV saved: {csv_path}", "=" * 76]

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_folder, f'summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(
        description='Boundary check on SROIE: verify parameter ranges (tesseract + easyocr)')
    parser.add_argument('--engines', nargs='+',
                        default=['tesseract', 'easyocr'],
                        choices=['tesseract', 'easyocr'])
    parser.add_argument('--images-folder', default=DEFAULT_IMAGES)
    parser.add_argument('--labels-folder', default=DEFAULT_LABELS)
    parser.add_argument('--output-folder', default=DEFAULT_OUTPUT)
    parser.add_argument('--n-images', type=int, default=20)
    parser.add_argument('--device',   default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    run(args.images_folder, args.labels_folder, args.output_folder,
        args.engines, args.n_images, args.device, args.seed)


if __name__ == '__main__':
    main()
