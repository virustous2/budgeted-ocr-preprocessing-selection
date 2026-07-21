#!/usr/bin/env python3
"""
Phase 3: Surrogate-Assisted Preprocessing Inference Pipeline

Given new image(s), uses the trained surrogate model to find the optimal
preprocessing parameters and run OCR with minimal OCR calls.

Pipeline:
  1. Extract image features
  2. Surrogate predicts CER for N candidate preprocessing combos (zero-cost)
  3. Rank → take top-K by predicted CER
  4. Apply K preprocessings → run OCR K times → pick best result
  5. Output: optimized OCR text + preprocessing params used

Usage:
    # Single image
    python inference_surrogate.py --image path/to/image.png --model-dir path/to/surrogate_models

    # Directory of images
    python inference_surrogate.py --image-dir path/to/images --model-dir path/to/surrogate_models

    # With ground truth (for validation)
    python inference_surrogate.py --image-dir path/to/images \
        --labels-dir path/to/references \
        --model-dir path/to/surrogate_models --top-k 10

Author: Research pipeline for QA-OCR
"""

import os
import sys
import pickle
import argparse
import time
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Suppress warnings
os.environ['FLAGS_pir_apply_inplace_pass'] = '1'
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent

import cv2
import numpy as np
import pandas as pd
from PIL import Image
try:
    from preprocessing_sampling import (
        LHSSampler,
        PARAM_NAMES,
        validate_executed_params,
    )
except ImportError:
    from main.preprocessing_sampling import (
        LHSSampler,
        PARAM_NAMES,
        validate_executed_params,
    )

FEATURE_COLS_BASE = [
    'orig_blur', 'orig_contrast', 'orig_brightness', 'orig_noise',
    'orig_edge_density', 'orig_entropy', 'orig_dynamic_range',
]

FEATURE_COLS_NEW = [
    'orig_text_density', 'orig_skew_angle', 'orig_connected_components',
    'orig_aspect_ratio', 'orig_line_count',
]

FEATURE_COLS = FEATURE_COLS_BASE + FEATURE_COLS_NEW

INTERACTIONS = [
    ('orig_blur', 'gaussian_ksize'),
    ('orig_contrast', 'clahe_clip'),
    ('orig_brightness', 'gamma'),
    ('orig_noise', 'bilateral_d'),
    ('orig_entropy', 'sharpen'),
]

INPUT_COLS = FEATURE_COLS + PARAM_NAMES + [
    f'inter_{f.replace("orig_", "")}_{p}' for f, p in INTERACTIONS
]


# ============================================================================
# Feature Extraction (same as Phase 1 - collect_preprocessing_data2.py)
# ============================================================================

class FeatureExtractor:
    @staticmethod
    def extract_features(image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h_img, w_img = gray.shape[:2]
        h = np.histogram(gray, bins=256, range=(0, 256), density=True)[0]

        features = {
            'orig_blur': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'orig_contrast': gray.std(),
            'orig_brightness': gray.mean(),
            'orig_noise': np.std(cv2.Laplacian(gray, cv2.CV_64F)),
            'orig_edge_density': float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
            'orig_entropy': float(-np.sum(h * np.log2(h + 1e-10))),
            'orig_dynamic_range': float(np.percentile(gray, 95) - np.percentile(gray, 5)),
        }

        # New features (matching collect_multiengine_data.py)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        features['orig_text_density'] = float(np.mean(binary > 0))

        # Skew angle via Hough lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=w_img // 4, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                if abs(dx) > 0:
                    angle = np.degrees(np.arctan2(y2 - y1, dx))
                    if abs(angle) < 30:
                        angles.append(angle)
            features['orig_skew_angle'] = abs(float(np.median(angles))) if angles else 0.0
        else:
            features['orig_skew_angle'] = 0.0

        # Connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        features['orig_connected_components'] = n_labels - 1

        # Aspect ratio
        features['orig_aspect_ratio'] = w_img / h_img

        # Line count via horizontal projection
        h_proj = np.sum(binary > 0, axis=1)
        threshold = w_img * 0.02
        in_line = False
        line_count = 0
        for val in h_proj:
            if val > threshold and not in_line:
                in_line = True
                line_count += 1
            elif val <= threshold:
                in_line = False
        features['orig_line_count'] = line_count

        return features


# ============================================================================
# Image Preprocessing (same as Phase 1)
# ============================================================================

class ImagePreprocessor:
    def preprocess(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        processed = image.copy()
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        validate_executed_params(params)

        # 1. Gaussian blur
        ksize = int(params['gaussian_ksize'])
        if ksize > 0:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # 2. Bilateral filter
        bilateral_d = int(params['bilateral_d'])
        if bilateral_d > 0:
            gray = cv2.bilateralFilter(gray, bilateral_d, 75, 75)

        # 3. CLAHE
        clahe_clip = params['clahe_clip']
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 4. Gamma correction
        gamma = params['gamma']
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            gray = cv2.LUT(gray, table)

        # 5. Sharpening
        sharpen = params['sharpen']
        if sharpen > 0:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32) * sharpen
            kernel[1, 1] = 1 + 4 * sharpen
            gray = cv2.filter2D(gray, -1, kernel)
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ============================================================================
# OCR Runner
# ============================================================================

class OCRRunner:
    def __init__(self, device: str = 'cuda'):
        from paddleocr import PaddleOCR
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        self.device = device

        self.paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='vi',
            device='gpu' if 'cuda' in device else 'cpu',
        )

        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = False
        config['device'] = device
        self.viet_ocr = Predictor(config)

    def run_ocr(self, image: np.ndarray, confidence_threshold: float = 0.8) -> str:
        temp_path = f"/tmp/temp_ocr_{os.getpid()}_{time.time()}.png"
        temp_folder = f"/tmp/ocr_crops_{os.getpid()}_{time.time()}"

        try:
            cv2.imwrite(temp_path, image)
            results = self.paddle_ocr.predict(input=temp_path)

            if results is None or len(results) == 0 or results[0] is None:
                return ""

            polygons = results[0].get('dt_polys', [])
            if len(polygons) == 0:
                return ""

            os.makedirs(temp_folder, exist_ok=True)
            texts = []

            for idx, polygon in enumerate(polygons):
                pts = np.array(polygon, dtype=np.float32)
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                x, y = max(0, x), max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                if w <= 0 or h <= 0:
                    continue
                cropped = image[y:y+h, x:x+w]

                crop_path = os.path.join(temp_folder, f"crop_{idx}.png")
                cv2.imwrite(crop_path, cropped)

                try:
                    pil_img = Image.open(crop_path)
                    text, prob = self.viet_ocr.predict(pil_img, return_prob=True)
                    if prob > confidence_threshold and text.strip():
                        texts.append(text.strip())
                except:
                    continue

            return "\n".join(texts)

        except Exception:
            return ""

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)


# ============================================================================
# Surrogate Inference Engine
# ============================================================================

class SurrogateInference:
    """Main inference engine: surrogate-assisted preprocessing optimization."""

    def __init__(self, model_dir: str, device: str = 'cuda',
                 n_candidates: int = 50, top_k: int = 10, engine: str = 'vietocr'):
        self.model_dir = model_dir
        self.top_k = top_k
        self.n_candidates = n_candidates

        # Load surrogate model — support both naming conventions:
        # old: best_model_*.pkl  (single-engine pipeline)
        # new: model_{engine}_*.pkl  (multiengine pipeline)
        model_files = (list(Path(model_dir).glob('best_model_*.pkl')) or
                       list(Path(model_dir).glob(f'model_{engine}_*.pkl')))
        if not model_files:
            raise FileNotFoundError(
                f"No model found in {model_dir} for engine '{engine}'. "
                f"Expected 'best_model_*.pkl' or 'model_{engine}_*.pkl'.")
        model_path = model_files[0]
        print(f"  Loading surrogate model: {model_path.name}")
        with open(model_path, 'rb') as f:
            self.surrogate = pickle.load(f)

        # Components
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = ImagePreprocessor()
        self.ocr_runner = OCRRunner(device)

        # Generate candidate combos (same LHS seed as training for consistency)
        self.lhs_sampler = LHSSampler(seed=42)
        self.candidate_combos = self.lhs_sampler.sample(n_candidates)
        print(f"  Generated {n_candidates} candidate preprocessing combos")
        print(f"  Top-K filter: K={top_k} ({top_k} OCR calls per image)")

    def predict_best_combos(self, image: np.ndarray) -> List[Tuple[Dict, float]]:
        """
        Use surrogate to rank preprocessing combos for an image.
        Returns: list of (params, predicted_cer) sorted by predicted CER.
        """
        features = self.feature_extractor.extract_features(image)

        # Build input matrix: [n_candidates × n_features]
        # Order: 12 image features + 5 params + 5 interaction terms = 22
        rows = []
        for params in self.candidate_combos:
            row = ([features[f] for f in FEATURE_COLS] +
                   [params[p] for p in PARAM_NAMES] +
                   [features[f] * params[p] for f, p in INTERACTIONS])
            rows.append(row)
        X = np.array(rows)

        # Predict CER for all combos (instant, no OCR)
        pred_cers = self.surrogate.predict(X)

        # Sort by predicted CER
        ranked = sorted(zip(self.candidate_combos, pred_cers), key=lambda x: x[1])
        return ranked

    def process_image(self, image_path: str, ground_truth: Optional[str] = None
                      ) -> Dict:
        """
        Full inference pipeline for a single image.

        Returns dict with:
          - best_text: OCR text from best preprocessing
          - best_params: preprocessing params used
          - all_results: list of {params, pred_cer, ocr_text} for top-K
          - timings: feature extraction, surrogate, OCR times
          - cer (if ground_truth provided)
        """
        from jiwer import cer as compute_cer_jiwer

        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Cannot load image: {image_path}'}

        img_name = Path(image_path).stem
        timings = {}

        # Step 1: Extract features + surrogate ranking (instant)
        t0 = time.time()
        ranked = self.predict_best_combos(image)
        timings['surrogate_rank_ms'] = (time.time() - t0) * 1000

        # Step 2: Take top-K
        top_k_combos = ranked[:self.top_k]

        # Step 3: Apply preprocessing + run OCR for each (K OCR calls)
        t0 = time.time()
        results = []
        for params, pred_cer in top_k_combos:
            processed = self.preprocessor.preprocess(image, params)
            ocr_text = self.ocr_runner.run_ocr(processed)
            results.append({
                'params': params,
                'pred_cer': float(pred_cer),
                'ocr_text': ocr_text,
            })
        timings['ocr_total_ms'] = (time.time() - t0) * 1000
        timings['ocr_per_call_ms'] = timings['ocr_total_ms'] / self.top_k

        # Step 4: Pick best (by surrogate ranking — top-1)
        best = results[0]  # Already sorted by pred_cer

        output = {
            'image_name': img_name,
            'image_path': image_path,
            'best_text': best['ocr_text'],
            'best_params': best['params'],
            'best_pred_cer': best['pred_cer'],
            'top_k_results': results,
            'timings': timings,
            'n_candidates': self.n_candidates,
            'top_k': self.top_k,
        }

        # Optional: compute CER if ground truth available
        if ground_truth:
            def _cer(pred, gt):
                pred = " ".join(pred.split())
                gt = " ".join(gt.split())
                if len(gt) == 0:
                    return 1.0 if len(pred) > 0 else 0.0
                return compute_cer_jiwer(gt, pred)

            # CER of surrogate's top-1
            output['cer_surrogate_top1'] = _cer(best['ocr_text'], ground_truth)

            # CER of best among K (oracle within K)
            cer_all = [_cer(r['ocr_text'], ground_truth) for r in results]
            best_k_idx = int(np.argmin(cer_all))
            output['cer_best_of_k'] = cer_all[best_k_idx]
            output['best_of_k_text'] = results[best_k_idx]['ocr_text']
            output['best_of_k_params'] = results[best_k_idx]['params']
            output['all_cers'] = cer_all

            # CER without preprocessing (run OCR on original)
            original_text = self.ocr_runner.run_ocr(image)
            output['cer_no_preprocessing'] = _cer(original_text, ground_truth)
            output['text_no_preprocessing'] = original_text

        return output


# ============================================================================
# Data Loading (for validation mode)
# ============================================================================

def collect_images(path: str) -> List[str]:
    """Collect image paths from file or directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]

    images = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        images.extend(p.rglob(ext))
    return sorted(str(img) for img in images)


def load_ground_truth(labels_dir: str, label_suffix: str = '_label') -> Dict[str, str]:
    """Load ground truth from label directory."""
    from docx import Document

    gt_dict = {}
    for year_folder in os.listdir(labels_dir):
        year_path = os.path.join(labels_dir, year_folder)
        if os.path.isdir(year_path):
            for label_file in os.listdir(year_path):
                if label_file.endswith('.docx'):
                    img_name = label_file.replace(f'{label_suffix}.docx', '')
                    label_path = os.path.join(year_path, label_file)
                    try:
                        doc = Document(label_path)
                        text = "\n".join(para.text.strip()
                                         for para in doc.paragraphs if para.text.strip())
                        gt_dict[img_name] = text
                    except:
                        pass
    return gt_dict


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Surrogate-Assisted Preprocessing Inference")

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image path')
    input_group.add_argument('--image-dir', type=str, help='Directory of images')

    # Model
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to trained surrogate model directory')
    parser.add_argument('--engine', type=str, default='vietocr',
                        choices=['vietocr', 'tesseract', 'easyocr'],
                        help='OCR engine to use (default: vietocr)')

    # Options
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of OCR calls per image (default: 10)')
    parser.add_argument('--n-candidates', type=int, default=50,
                        help='Number of candidate combos to rank (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')

    # Validation (optional)
    parser.add_argument('--labels-dir', type=str, default=None,
                        help='Ground truth labels dir (for CER validation)')
    parser.add_argument('--label-suffix', type=str, default='_label',
                        help='Label file suffix')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/recomputed/inference_TIMESTAMP)')

    args = parser.parse_args()

    # Output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = str(RELEASE_ROOT / f'results/recomputed/inference_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SURROGATE-ASSISTED PREPROCESSING INFERENCE")
    print("=" * 70)

    # Initialize engine
    print("\nInitializing...")
    engine = SurrogateInference(
        model_dir=args.model_dir,
        device=args.device,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        engine=args.engine,
    )

    # Collect images
    image_path = args.image or args.image_dir
    image_paths = collect_images(image_path)
    print(f"  Found {len(image_paths)} images")

    # Load ground truth if provided
    gt_dict = {}
    if args.labels_dir:
        gt_dict = load_ground_truth(args.labels_dir, args.label_suffix)
        print(f"  Loaded {len(gt_dict)} ground truth entries")

    # Process images
    print("\n" + "=" * 70)
    print(f"PROCESSING ({args.top_k} OCR calls per image, "
          f"{args.n_candidates} candidates)")
    print("=" * 70)

    all_results = []
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem
        gt = gt_dict.get(img_name)

        result = engine.process_image(img_path, ground_truth=gt)

        if 'error' in result:
            print(f"  [{i+1}/{len(image_paths)}] {img_name}: {result['error']}")
            continue

        # Print progress
        status = f"  [{i+1}/{len(image_paths)}] {img_name}"
        status += f"  surrogate={result['timings']['surrogate_rank_ms']:.0f}ms"
        status += f"  ocr={result['timings']['ocr_total_ms']:.0f}ms"
        if gt:
            status += (f"  CER: no_prep={result['cer_no_preprocessing']:.4f}"
                       f" → top1={result['cer_surrogate_top1']:.4f}"
                       f" → bestK={result['cer_best_of_k']:.4f}")
        print(status)

        all_results.append(result)

    elapsed = time.time() - start_time

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save OCR texts
    texts_dir = os.path.join(args.output_dir, 'texts')
    os.makedirs(texts_dir, exist_ok=True)
    for result in all_results:
        text_path = os.path.join(texts_dir, f"{result['image_name']}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(result['best_text'])
    print(f"  OCR texts: {texts_dir}/")

    # Save detailed results as CSV
    rows = []
    for r in all_results:
        row = {
            'image_name': r['image_name'],
            'best_pred_cer': r['best_pred_cer'],
            'surrogate_ms': r['timings']['surrogate_rank_ms'],
            'ocr_ms': r['timings']['ocr_total_ms'],
        }
        row.update(r['best_params'])
        if 'cer_surrogate_top1' in r:
            row['cer_no_preprocessing'] = r['cer_no_preprocessing']
            row['cer_surrogate_top1'] = r['cer_surrogate_top1']
            row['cer_best_of_k'] = r['cer_best_of_k']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results CSV: {csv_path}")

    # Summary report
    report_path = os.path.join(args.output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SURROGATE-ASSISTED PREPROCESSING INFERENCE REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write(f"  Model: {args.model_dir}\n")
        f.write(f"  Candidates: {args.n_candidates}\n")
        f.write(f"  Top-K: {args.top_k}\n")
        f.write(f"  Device: {args.device}\n")
        f.write(f"  Images processed: {len(all_results)}\n")
        per_image = f"{elapsed/len(all_results):.1f}s/image" if all_results else "N/A"
        f.write(f"  Total time: {elapsed:.1f}s ({per_image})\n\n")

        # Timing stats
        surr_times = [r['timings']['surrogate_rank_ms'] for r in all_results]
        ocr_times = [r['timings']['ocr_total_ms'] for r in all_results]
        f.write("TIMING\n")
        f.write(f"  Surrogate ranking: {np.mean(surr_times):.1f}ms avg\n")
        f.write(f"  OCR ({args.top_k} calls): {np.mean(ocr_times):.0f}ms avg\n")
        f.write(f"  Total per image: {np.mean(surr_times) + np.mean(ocr_times):.0f}ms avg\n\n")

        # CER stats (if ground truth available)
        if any('cer_surrogate_top1' in r for r in all_results):
            validated = [r for r in all_results if 'cer_surrogate_top1' in r]
            cer_noprep = [r['cer_no_preprocessing'] for r in validated]
            cer_top1 = [r['cer_surrogate_top1'] for r in validated]
            cer_bestk = [r['cer_best_of_k'] for r in validated]

            f.write("CER RESULTS (with ground truth)\n")
            f.write(f"  Images validated: {len(validated)}\n")
            f.write(f"  No preprocessing:     {np.mean(cer_noprep):.4f} "
                    f"(median {np.median(cer_noprep):.4f})\n")
            f.write(f"  Surrogate top-1:      {np.mean(cer_top1):.4f} "
                    f"(median {np.median(cer_top1):.4f})\n")
            f.write(f"  Best of K={args.top_k}:        {np.mean(cer_bestk):.4f} "
                    f"(median {np.median(cer_bestk):.4f})\n")
            f.write(f"\n  Improvement (no_prep → best_of_K): "
                    f"{np.mean(cer_noprep) - np.mean(cer_bestk):.4f} "
                    f"({(np.mean(cer_noprep) - np.mean(cer_bestk))/np.mean(cer_noprep)*100:.1f}%)\n")

            # Print to console too
            print(f"\n  CER Summary ({len(validated)} images with ground truth):")
            print(f"    No preprocessing: {np.mean(cer_noprep):.4f}")
            print(f"    Surrogate top-1:  {np.mean(cer_top1):.4f}")
            print(f"    Best of K={args.top_k}:    {np.mean(cer_bestk):.4f}")
            print(f"    Improvement:      {np.mean(cer_noprep) - np.mean(cer_bestk):.4f}")

    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Processed {len(all_results)} images in {elapsed:.1f}s")
    print(f"  OCR calls: {len(all_results) * args.top_k} total "
          f"(saved {(1 - args.top_k/args.n_candidates)*100:.0f}% vs exhaustive)")


if __name__ == "__main__":
    main()
