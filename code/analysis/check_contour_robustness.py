#!/usr/bin/env python3
"""
Robustness check: does CER change significantly when contour removal thresholds vary?

Method: one-at-a-time sensitivity.
  - 4 threshold parameters × 5 values each = 20 variants
  - 30 images (3 per year × 10 years), VietOCR only
  - No crop preprocessing (just contour removal → detect → recognize)

Purpose: show thresholds are robust (CER stable ±0.2%) → not "magic numbers".
NOT for paper results — for paper methodology justification only.

Usage:
    python analysis/check_contour_robustness.py --device cuda
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from itertools import product

os.environ['FLAGS_pir_apply_inplace_pass'] = '1'
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from docx import Document
from jiwer import cer as jiwer_cer

IMAGES_DIR = PROJECT_ROOT / 'datasets/300_images/300_images'
LABELS_DIR = PROJECT_ROOT / 'datasets/300_images_labels/300_images_labels'
OUTPUT_DIR = PROJECT_ROOT / 'analysis/outputs/contour_robustness'

# Default thresholds (updated: area=500 based on visualization)
DEFAULT = {
    'area':       500,
    'aspect_max': 5.0,
    'min_w':      80,
    'min_h':      30,
}

# One-at-a-time variants (default marked with *)
VARIANTS = {
    'area':       [500, 1000, 1500, 2000, 3000],
    'aspect_max': [3.0, 4.0,  5.0,  6.0,  8.0],
    'min_w':      [50,  60,   80,   100,  120],
    'min_h':      [15,  20,   30,   40,   50],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_cer(pred: str, gt: str) -> float:
    pred = " ".join(pred.split())
    gt   = " ".join(gt.split())
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return float(jiwer_cer(gt, pred))


def read_docx_text(path: str) -> str:
    try:
        doc = Document(path)
        return "\t".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    x, y = max(0, x), max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return np.array([])
    return image[y:y+h, x:x+w]


# ── contour removal with configurable thresholds ──────────────────────────────

def contour_removal(image: np.ndarray,
                    area: int, aspect_max: float,
                    min_w: int, min_h: int) -> np.ndarray:
    out  = image.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = cv2.contourArea(cnt)
        asp      = w / h if h != 0 else 0
        if cnt_area > area and 1.0 < asp < aspect_max and w > min_w and h > min_h:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return out


# ── select 30 images (3 per year) ────────────────────────────────────────────

def select_images(n_per_year: int = 3) -> list:
    items = []
    for year_dir in sorted(IMAGES_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        for img_path in sorted(year_dir.glob('*.png'))[:n_per_year]:
            label = LABELS_DIR / year / f"{img_path.stem}_label.docx"
            if label.exists():
                items.append((str(img_path), str(label), year))
    return items


# ── VietOCR recognize ─────────────────────────────────────────────────────────

def recognize(vietocr, crops: list) -> str:
    texts = []
    for crop in crops:
        if crop.size == 0:
            continue
        try:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            text, prob = vietocr.predict(pil, return_prob=True)
            if prob > 0.8 and text.strip():
                texts.append(text.strip())
        except Exception:
            continue
    return "\t".join(texts)


# ── run one threshold config on all images ────────────────────────────────────

def run_config(paddle, vietocr, images: list,
               area: int, aspect_max: float,
               min_w: int, min_h: int) -> float:
    cers = []
    for img_path, label_path, _ in images:
        gt = read_docx_text(label_path)
        if not gt:
            continue

        raw  = cv2.imread(img_path)
        page = contour_removal(raw, area, aspect_max, min_w, min_h)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, page)
        try:
            det = paddle.predict(input=tmp_path)
        finally:
            os.unlink(tmp_path)

        polygons   = det[0].get('dt_polys', []) if det and det[0] else []
        crops      = [crop_polygon(page, p) for p in polygons]
        crops      = [c for c in crops if c.size > 0]
        pred       = recognize(vietocr, crops)
        cers.append(compute_cer(pred, gt))

    return float(np.mean(cers)) if cers else 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main(device: str, n_per_year: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PaddleOCR...")
    from paddleocr import PaddleOCR
    paddle = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='vi',
        device='gpu' if 'cuda' in device else 'cpu',
    )

    print("Loading VietOCR...")
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    cfg = Cfg.load_config_from_name('vgg_transformer')
    cfg['cnn']['pretrained'] = False
    cfg['device'] = device
    vietocr = Predictor(cfg)

    images = select_images(n_per_year)
    print(f"\nImages: {len(images)} | Variants: {sum(len(v) for v in VARIANTS.values())}\n")

    rows = []

    # Baseline: default thresholds
    print("Running default thresholds...")
    cer_default = run_config(paddle, vietocr, images, **DEFAULT)
    rows.append({
        'param': 'default', 'value': '-',
        'area': DEFAULT['area'], 'aspect_max': DEFAULT['aspect_max'],
        'min_w': DEFAULT['min_w'], 'min_h': DEFAULT['min_h'],
        'mean_cer': cer_default, 'is_default': True,
    })
    print(f"  Default CER: {cer_default*100:.2f}%\n")

    # One-at-a-time variants
    for param, values in VARIANTS.items():
        print(f"Parameter: {param}")
        for val in tqdm(values, desc=f"  {param}"):
            if val == DEFAULT[param]:
                continue  # already ran as default
            config = dict(DEFAULT)
            config[param] = val
            mean_cer = run_config(paddle, vietocr, images, **config)
            rows.append({
                'param': param, 'value': val,
                **config,
                'mean_cer': mean_cer, 'is_default': False,
            })
        print()

    # Results
    df = pd.DataFrame(rows)
    df['delta_vs_default'] = df['mean_cer'] - cer_default

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = OUTPUT_DIR / f'robustness_{ts}.csv'
    df.to_csv(csv_path, index=False)

    # Summary per parameter
    lines = [
        "=" * 65,
        "CONTOUR REMOVAL THRESHOLD ROBUSTNESS CHECK",
        f"Images: {len(images)} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Default CER: {cer_default*100:.2f}%",
        "=" * 65,
    ]
    for param in VARIANTS:
        lines.append(f"\n  Parameter: {param}")
        sub = df[df['param'].isin([param, 'default'])]
        # build full table for this param
        all_vals = []
        for val in VARIANTS[param]:
            if val == DEFAULT[param]:
                row = {'value': val, 'mean_cer': cer_default, 'delta': 0.0}
            else:
                r = df[(df['param'] == param) & (df['value'] == val)]
                if len(r):
                    row = {'value': val,
                           'mean_cer': r.iloc[0]['mean_cer'],
                           'delta':    r.iloc[0]['delta_vs_default']}
                else:
                    continue
            all_vals.append(row)

        for r in all_vals:
            marker = ' ← default' if r['value'] == DEFAULT[param] else ''
            lines.append(f"    {param}={r['value']:>6}  CER={r['mean_cer']*100:.2f}%  "
                         f"delta={r['delta']*100:+.2f}%{marker}")

        deltas = [abs(r['delta']) for r in all_vals]
        lines.append(f"  Max |delta|: {max(deltas)*100:.2f}%  "
                     f"→ {'ROBUST ✓' if max(deltas) < 0.002 else 'SENSITIVE ⚠'}")

    lines += ["", "=" * 65, f"CSV: {csv_path}"]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(OUTPUT_DIR / f'summary_{ts}.txt', 'w') as f:
        f.write(summary_text + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',   default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--n-images', type=int, default=3,
                        help='Images per year (default 3 → 30 total)')
    args = parser.parse_args()
    main(args.device, args.n_images)
