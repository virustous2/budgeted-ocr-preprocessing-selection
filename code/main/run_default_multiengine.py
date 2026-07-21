#!/usr/bin/env python3
"""
Run default (no preprocessing) OCR on 300 raw images for all 3 engines.
Outputs per-engine average CER for use as NoPre baseline in paper.

Usage:
    python modules/run_default_multiengine.py
    python modules/run_default_multiengine.py --device cuda
    python modules/run_default_multiengine.py --engines tesseract easyocr
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

os.environ['FLAGS_pir_apply_inplace_pass'] = '1'

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from docx import Document
from jiwer import cer

SCRIPT_DIR = Path(__file__).parent.resolve()
RELEASE_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_ROOT))

DEFAULT_IMAGES   = str(RELEASE_ROOT / 'data/source_documents_not_included/vietnamese/images')
DEFAULT_LABELS   = str(RELEASE_ROOT / 'data/source_documents_not_included/vietnamese/references')
DEFAULT_OUTPUT   = str(RELEASE_ROOT / 'results/recomputed/default_output_multiengine')
LABEL_SUFFIX     = '_label'


# ── helpers ───────────────────────────────────────────────────────────────────

def read_docx_text(path: str) -> str:
    doc = Document(path)
    return " ".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def read_txt_file(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return " ".join(lines)


def calc_cer(gt: str, pred: str) -> Optional[float]:
    try:
        gt_n   = " ".join(gt.split())
        pred_n = " ".join(pred.split())
        if len(gt_n) == 0:
            return 1.0 if len(pred_n) > 0 else 0.0
        return cer(gt_n, pred_n)
    except Exception:
        return None


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
    return image[y:y+h, x:x+w]


def collect_images(images_folder: str) -> List[Tuple[str, str]]:
    """Return list of (image_path, year)."""
    items = []
    for year in sorted(os.listdir(images_folder)):
        year_path = os.path.join(images_folder, year)
        if not os.path.isdir(year_path):
            continue
        for fname in sorted(os.listdir(year_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                items.append((os.path.join(year_path, fname), year))
    return items


# ── model init ────────────────────────────────────────────────────────────────

def load_models(engines: List[str], device: str):
    models = {}

    print("Loading PaddleOCR (detection)...")
    from paddleocr import PaddleOCR
    models['paddle'] = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='vi',
        device='gpu' if 'cuda' in device else 'cpu',
    )

    if 'vietocr' in engines:
        print("Loading VietOCR...")
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['cnn']['pretrained'] = False
        cfg['device'] = device
        models['vietocr'] = Predictor(cfg)

    if 'tesseract' in engines:
        print("Loading Tesseract...")
        import pytesseract
        langs = pytesseract.get_languages()
        if 'vie' not in langs:
            print("  WARNING: 'vie' not found. Install: sudo apt install tesseract-ocr-vie")
        models['tesseract'] = pytesseract

    if 'easyocr' in engines:
        print("Loading EasyOCR...")
        import easyocr
        models['easyocr'] = easyocr.Reader(['vi'], gpu=('cuda' in device), verbose=False)

    return models


# ── OCR pipeline ──────────────────────────────────────────────────────────────

def detect(paddle_ocr, image_path: str) -> List[np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        return []
    temp_path = f"/tmp/temp_detect_{os.getpid()}_{time.time()}.png"
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
    for poly in polygons:
        crop = crop_polygon(image, poly)
        if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            crops.append(crop)
    return crops


def recognize(engine_name: str, model, crops: List[np.ndarray],
              confidence_threshold: float = 0.8) -> str:
    texts = []
    if engine_name == 'vietocr':
        for crop in crops:
            try:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text, prob = model.predict(pil, return_prob=True)
                if prob > confidence_threshold and text.strip():
                    texts.append(text.strip())
            except Exception:
                continue

    elif engine_name == 'tesseract':
        for crop in crops:
            try:
                pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text = model.image_to_string(pil, lang='vie').strip()
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


# ── main processing ───────────────────────────────────────────────────────────

def run(images_folder: str, labels_folder: str, output_folder: str,
        engines: List[str], device: str):

    os.makedirs(output_folder, exist_ok=True)
    models  = load_models(engines, device)
    paddle  = models['paddle']
    images  = collect_images(images_folder)
    print(f"\nFound {len(images)} images. Engines: {engines}\n")

    rows = []
    for img_path, year in tqdm(images, desc="Processing"):
        image_name = Path(img_path).stem
        label_path = os.path.join(labels_folder, year,
                                  f"{image_name}{LABEL_SUFFIX}.docx")

        gt_text = None
        if os.path.exists(label_path):
            gt_text = read_docx_text(label_path)
        else:
            print(f"  WARNING: label not found: {label_path}")

        t0    = time.time()
        crops = detect(paddle, img_path)
        detect_time = time.time() - t0

        row = {'file_name': f"{image_name}.png", 'year': year}
        for eng in engines:
            t1   = time.time()
            pred = recognize(eng, models[eng], crops)
            row[f'time_{eng}'] = round(time.time() - t1 + detect_time / len(engines), 3)
            row[f'cer_{eng}']  = calc_cer(gt_text, pred) if gt_text else -1

        rows.append(row)

    # ── save CSV ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path  = os.path.join(output_folder, f'default_cer_{timestamp}.csv')
    fieldnames = ['file_name', 'year'] + \
                 [f for eng in engines for f in (f'cer_{eng}', f'time_{eng}')]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── summary ──
    summary_path = os.path.join(output_folder, f'summary_{timestamp}.txt')
    lines = ["=" * 60, "DEFAULT (NO PREPROCESSING) OCR — SUMMARY",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"Images: {len(rows)}", "=" * 60, ""]
    for eng in engines:
        valid = [r[f'cer_{eng}'] for r in rows if r[f'cer_{eng}'] >= 0]
        avg   = sum(valid) / len(valid) if valid else float('nan')
        lines.append(f"  {eng:<12}: {avg*100:.2f}%  (n={len(valid)})")
    lines += ["", f"CSV: {csv_path}", "=" * 60]

    summary_text = "\n".join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text)
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run default (no-preprocessing) OCR on 300 images for multiple engines')
    parser.add_argument('--engines', nargs='+',
                        default=['vietocr', 'tesseract', 'easyocr'],
                        choices=['vietocr', 'tesseract', 'easyocr'],
                        help='Which engines to run')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--images-folder',  default=DEFAULT_IMAGES)
    parser.add_argument('--labels-folder',  default=DEFAULT_LABELS)
    parser.add_argument('--output-folder',  default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run(args.images_folder, args.labels_folder, args.output_folder,
        args.engines, args.device)


if __name__ == '__main__':
    main()
