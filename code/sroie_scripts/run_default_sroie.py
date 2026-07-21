#!/usr/bin/env python3
"""
Run exact no-preprocessing OCR on SROIE images for use as the NoPre baseline.

This mirrors the image-selection logic of collect_lhs_sroie.py so that
--n-images/--seed choose the same subset for baseline and surrogate evaluation.
"""

import os
import sys
import time
import csv
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List

os.environ['FLAGS_pir_apply_inplace_pass'] = '1'

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from jiwer import cer
from paddleocr import PaddleOCR

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

DEFAULT_IMAGES = str(RELEASE_ROOT / 'data/source_documents_not_included/sroie/images')
DEFAULT_LABELS = str(RELEASE_ROOT / 'data/source_documents_not_included/sroie/ground_truth')
DEFAULT_OUTPUT = str(RELEASE_ROOT / 'results/recomputed/default_output_sroie')


def read_txt_gt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return " ".join(lines)


def compute_cer(prediction: str, ground_truth: str) -> float:
    pred = " ".join(prediction.split())
    gt = " ".join(ground_truth.split())
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return cer(gt, pred)


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return np.array([])
    return image[y:y + h, x:x + w]


def collect_all_images(images_folder: str) -> List[str]:
    items = []
    for fname in sorted(os.listdir(images_folder)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and '(' not in fname:
            items.append(os.path.join(images_folder, fname))
    return items


def load_ground_truth(labels_folder: str) -> Dict[str, str]:
    gt_dict = {}
    for fname in os.listdir(labels_folder):
        if fname.endswith('.txt') and '(' not in fname:
            image_name = Path(fname).stem
            label_path = os.path.join(labels_folder, fname)
            try:
                gt_dict[image_name] = read_txt_gt(label_path)
            except Exception:
                pass
    return gt_dict


def load_models(engines: List[str], device: str):
    models = {}

    print("Loading PaddleOCR detector (lang=en)...")
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
            print(f"  WARNING: 'eng' not found in Tesseract languages: {langs}")
        models['tesseract'] = pytesseract

    if 'easyocr' in engines:
        print("Loading EasyOCR (lang=en)...")
        import easyocr
        models['easyocr'] = easyocr.Reader(['en'], gpu=('cuda' in device), verbose=False)

    return models


def detect(paddle_ocr, image: np.ndarray) -> List[np.ndarray]:
    temp_path = f"/tmp/temp_sroie_default_{os.getpid()}_{time.time()}.png"
    try:
        cv2.imwrite(temp_path, image)
        results = paddle_ocr.predict(input=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not results or results[0] is None:
        return []

    polygons = results[0].get('dt_polys', [])
    crops = []
    for polygon in polygons:
        crop = crop_polygon(image, polygon)
        if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            crops.append(crop)
    return crops


def recognize(engine_name: str, model, crops: List[np.ndarray]) -> str:
    texts = []

    if engine_name == 'tesseract':
        for crop in crops:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text = model.image_to_string(pil_img, lang='eng').strip()
                if text:
                    texts.append(" ".join(text.split()))
            except Exception:
                continue

    elif engine_name == 'easyocr':
        for crop in crops:
            try:
                for text in model.readtext(crop, detail=0):
                    text = text.strip()
                    if text:
                        texts.append(text)
            except Exception:
                continue

    return "\t".join(texts)


def run(images_folder: str,
        labels_folder: str,
        output_folder: str,
        engines: List[str],
        device: str,
        n_images: int | None = None,
        seed: int = 42):
    os.makedirs(output_folder, exist_ok=True)

    all_images = collect_all_images(images_folder)
    gt_dict = load_ground_truth(labels_folder)
    valid_images = [path for path in all_images if Path(path).stem in gt_dict]

    if n_images is not None and n_images < len(valid_images):
        rng = random.Random(seed)
        valid_images = rng.sample(valid_images, n_images)

    print(f"Found {len(valid_images)} valid images. Engines: {engines}")
    models = load_models(engines, device)
    paddle = models['paddle']

    rows = []
    for img_path in tqdm(valid_images, desc="Processing"):
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_name = Path(img_path).stem
        gt_text = gt_dict[image_name]

        t0 = time.time()
        crops = detect(paddle, image)
        detect_time = time.time() - t0

        row = {'file_name': Path(img_path).name}
        for engine in engines:
            t1 = time.time()
            pred = recognize(engine, models[engine], crops)
            row[f'time_{engine}'] = round(time.time() - t1 + detect_time / len(engines), 3)
            row[f'cer_{engine}'] = compute_cer(pred, gt_text)
        rows.append(row)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_folder, f'default_cer_{timestamp}.csv')
    fieldnames = ['file_name'] + [f for eng in engines for f in (f'cer_{eng}', f'time_{eng}')]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(output_folder, f'summary_{timestamp}.txt')
    lines = [
        "=" * 60,
        "SROIE DEFAULT (NO PREPROCESSING) OCR — SUMMARY",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Images: {len(rows)}",
        "=" * 60,
        "",
    ]
    for engine in engines:
        vals = [row[f'cer_{engine}'] for row in rows if row[f'cer_{engine}'] >= 0]
        avg = sum(vals) / len(vals) if vals else float('nan')
        lines.append(f"  {engine:<12}: {avg * 100:.2f}%  (n={len(vals)})")
    lines += ["", f"CSV: {csv_path}", "=" * 60]

    summary_text = "\n".join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Run exact no-preprocessing OCR on SROIE for multiple engines')
    parser.add_argument('--engines', nargs='+',
                        default=['tesseract', 'easyocr'],
                        choices=['tesseract', 'easyocr'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--images-folder', default=DEFAULT_IMAGES)
    parser.add_argument('--labels-folder', default=DEFAULT_LABELS)
    parser.add_argument('--output-folder', default=DEFAULT_OUTPUT)
    parser.add_argument('--n-images', type=int, default=None,
                        help='Limit to N randomly selected images using the same seed-based sampling as collect_lhs_sroie.py')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run(args.images_folder, args.labels_folder, args.output_folder,
        args.engines, args.device, args.n_images, args.seed)


if __name__ == '__main__':
    main()
