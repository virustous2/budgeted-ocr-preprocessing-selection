#!/usr/bin/env python3
"""
Boundary check: test in-range vs out-of-range preprocessing values on ~20 images.

For each parameter, vary it (one-at-a-time) while holding others at no-op,
then compare CER inside-range vs outside-range to justify the chosen bounds.

Uses all 3 engines: VietOCR, Tesseract, EasyOCR.

Output: analysis/outputs/boundary_check/
    - boundary_check_TIMESTAMP.csv   (one row per image × engine × param × value)
    - summary_TIMESTAMP.txt          (mean CER per config)

Usage:
    python analysis/check_range_boundaries.py
    python analysis/check_range_boundaries.py --n-images 20 --device cuda
    python analysis/check_range_boundaries.py --engines vietocr tesseract
"""

import os
import sys
import csv
import time
import argparse
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from docx import Document

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGES  = str(PROJECT_ROOT / 'datasets/300_images/300_images')
DEFAULT_LABELS  = str(PROJECT_ROOT / 'datasets/300_images_labels/300_images_labels')
DEFAULT_OUTPUT  = str(SCRIPT_DIR / 'outputs/boundary_check')
DEFAULT_NOPREC  = str(PROJECT_ROOT / 'outputs/default_output_multiengine_20260410_bbox/default_cer_20260410_053801.csv')
LABEL_SUFFIX    = '_label'

# ── parameter test configs (one-at-a-time, others at no-op) ──────────────────
# Format: (value, status)  status ∈ {noop, in-range, boundary, out-of-range}
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
        (2.0, 'out-of-range'),
        (2.5, 'out-of-range'),
        (3.0, 'out-of-range'),
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

def read_docx_text(path: str) -> str:
    doc = Document(path)
    return " ".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def calc_cer(gt: str, pred: str) -> float:
    from jiwer import cer as jiwer_cer
    gt_n   = " ".join(gt.split())
    pred_n = " ".join(pred.split())
    if len(gt_n) == 0:
        return 1.0 if len(pred_n) > 0 else 0.0
    return float(jiwer_cer(gt_n, pred_n))


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return np.array([])
    return image[y:y+h, x:x+w]


def detect(paddle_ocr, image_path: str) -> List[np.ndarray]:
    """Mirror the current default/collect detection path: read image, save temp PNG, detect."""
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


def collect_all_images(images_folder: str) -> List[Tuple[str, str]]:
    items = []
    for year in sorted(os.listdir(images_folder)):
        year_path = os.path.join(images_folder, year)
        if not os.path.isdir(year_path):
            continue
        for fname in sorted(os.listdir(year_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                items.append((os.path.join(year_path, fname), year))
    return items


def select_images(all_images: List[Tuple[str, str]],
                  noprec_csv: str, n: int, seed: int = 42) -> List[Tuple[str, str]]:
    """Select n images with medium VietOCR NoPre CER (0.02–0.20) for meaningful signal."""
    import pandas as pd
    try:
        df = pd.read_csv(noprec_csv)
        df = df[df['cer_vietocr'].between(0.02, 0.20)]
        target_names = set(df['file_name'].tolist())
        candidates = [(p, y) for p, y in all_images
                      if Path(p).name in target_names]
        rng = random.Random(seed)
        if len(candidates) >= n:
            return rng.sample(candidates, n)
        return rng.sample(all_images, min(n, len(all_images)))
    except Exception:
        rng = random.Random(seed)
        return rng.sample(all_images, min(n, len(all_images)))


# ── model init ────────────────────────────────────────────────────────────────

def load_models(engines: List[str], device: str) -> Dict:
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


# ── recognition per engine ────────────────────────────────────────────────────

def recognize_crops(engine: str, model, crops: List[np.ndarray],
                    params: Dict, conf_thr: float = 0.8) -> str:
    texts = []

    if engine == 'vietocr':
        for crop in crops:
            proc = preprocess(crop, params)
            try:
                pil = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                text, prob = model.predict(pil, return_prob=True)
                if prob > conf_thr and text.strip():
                    texts.append(text.strip())
            except Exception:
                continue

    elif engine == 'tesseract':
        for crop in crops:
            proc = preprocess(crop, params)
            try:
                pil  = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                text = model.image_to_string(pil, lang='vie').strip()
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
        noprec_csv: str, engines: List[str], n_images: int,
        device: str, seed: int, parameters: Optional[List[str]] = None):

    os.makedirs(output_folder, exist_ok=True)
    if parameters is None:
        parameters = list(PARAM_TESTS.keys())
    unknown_params = sorted(set(parameters) - set(PARAM_TESTS.keys()))
    if unknown_params:
        raise ValueError(f"Unknown parameters requested: {unknown_params}")

    all_images = collect_all_images(images_folder)
    images = select_images(all_images, noprec_csv, n_images, seed)
    print(f"\nSelected {len(images)} images. Engines: {engines}")
    print(f"Parameters: {parameters}\n")

    models = load_models(engines, device)
    paddle = models['paddle']

    rows = []
    for img_path, year in tqdm(images, desc="Images"):
        image_name = Path(img_path).stem
        label_path = os.path.join(labels_folder, year,
                                  f"{image_name}{LABEL_SUFFIX}.docx")
        if not os.path.exists(label_path):
            print(f"  SKIP (no label): {image_name}")
            continue

        gt_text = read_docx_text(label_path)
        if not gt_text.strip():
            continue

        # Detect once on the raw image — shared across all parameter configs and engines.
        crops_raw = detect(paddle, img_path)
        if not crops_raw:
            print(f"  SKIP (no detections): {image_name}")
            continue

        # Test each parameter one-at-a-time, for each engine
        for param_name in parameters:
            test_values = PARAM_TESTS[param_name]
            for value, status in test_values:
                params = {**NOOP_PARAMS, param_name: value}
                lo, hi = CHOSEN_BOUNDS[param_name]
                in_bound = (lo <= value <= hi)

                for eng in engines:
                    pred    = recognize_crops(eng, models[eng], crops_raw, params)
                    cer_val = calc_cer(gt_text, pred)
                    rows.append({
                        'image':     image_name,
                        'year':      year,
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

    # ── save CSV ──
    csv_path = os.path.join(output_folder, f'boundary_check_{timestamp}.csv')
    fieldnames = ['image', 'year', 'engine', 'parameter', 'value', 'status', 'in_bound', 'cer']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── summary ──
    import pandas as pd
    df = pd.DataFrame(rows)

    summary_lines = [
        "=" * 76,
        "BOUNDARY CHECK — Mean CER by engine, parameter, and value status",
        f"Images: {len(images)}  Engines: {engines}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Parameters tested: {parameters}",
        f"Chosen bounds: {CHOSEN_BOUNDS}",
        "=" * 76, "",
    ]

    for eng in engines:
        summary_lines.append(f"{'='*30}  ENGINE: {eng.upper()}  {'='*30}")
        df_eng = df[df['engine'] == eng]
        for param in parameters:
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

    # ── aggregate across all engines: in-range vs out-of-range ──
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
        description='Boundary check: verify parameter ranges across all 3 OCR engines')
    parser.add_argument('--engines', nargs='+',
                        default=['vietocr', 'tesseract', 'easyocr'],
                        choices=['vietocr', 'tesseract', 'easyocr'])
    parser.add_argument('--images-folder',  default=DEFAULT_IMAGES)
    parser.add_argument('--labels-folder',  default=DEFAULT_LABELS)
    parser.add_argument('--output-folder',  default=DEFAULT_OUTPUT)
    parser.add_argument('--noprec-csv',     default=DEFAULT_NOPREC,
                        help='CSV from run_default_multiengine.py (used to select images)')
    parser.add_argument('--n-images',       type=int, default=20,
                        help='Number of images to test (default: 20)')
    parser.add_argument('--device',         default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--parameters', nargs='+', default=None,
                        choices=list(PARAM_TESTS.keys()),
                        help='Subset of parameters to test, e.g. --parameters gamma')
    args = parser.parse_args()

    run(args.images_folder, args.labels_folder, args.output_folder,
        args.noprec_csv, args.engines, args.n_images, args.device, args.seed,
        args.parameters)


if __name__ == '__main__':
    main()
