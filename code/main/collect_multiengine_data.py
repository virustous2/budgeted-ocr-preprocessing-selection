#!/usr/bin/env python3
"""
Multi-Engine Data Collection for Surrogate Model Training.

Pipeline: Image → Preprocess → PaddleOCR detect → crop lines → recognize with 3 engines:
  1. VietOCR (transformer-based, Vietnamese-specific)
  2. Tesseract (traditional, with Vietnamese lang pack)
  3. EasyOCR (CNN+LSTM, multilingual)

Detection is shared (PaddleOCR) — only recognition differs.

Output CSV columns:
  - image features (orig_blur, orig_contrast, ..., orig_line_count)
  - preprocessing params (gaussian_ksize, bilateral_d, ...)
  - cer_vietocr, cer_tesseract, cer_easyocr

Usage:
    python collect_multiengine_data.py --device cuda
    python collect_multiengine_data.py --device cuda --resume
    python collect_multiengine_data.py --device cuda --engines vietocr tesseract
"""

import os
import sys
import pickle
import argparse
import time
import json
import signal
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

os.environ['FLAGS_pir_apply_inplace_pass'] = '1'
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from docx import Document

# OCR Libraries — imported conditionally based on --engines
from paddleocr import PaddleOCR
try:
    from preprocessing_sampling import (
        LHSSampler,
        sampler_validation_summary,
        validate_executed_params,
    )
except ImportError:
    from main.preprocessing_sampling import (
        LHSSampler,
        sampler_validation_summary,
        validate_executed_params,
    )

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n Shutdown requested. Saving checkpoint...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_cer(prediction: str, ground_truth: str) -> float:
    from jiwer import cer as compute_cer_jiwer
    pred = " ".join(prediction.split())
    gt = " ".join(ground_truth.split())
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return compute_cer_jiwer(gt, pred)


def read_docx_text(docx_path: str) -> str:
    try:
        doc = Document(docx_path)
        text = "\t".join(para.text.strip()
                         for para in doc.paragraphs if para.text.strip())
        return text
    except:
        return ""


def crop_polygon(image: np.ndarray, polygon: list) -> np.ndarray:
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


# ============================================================================
# Preprocessing
# ============================================================================

class ImagePreprocessor:
    def preprocess(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        processed = image.copy()
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        validate_executed_params(params)

        ksize = int(params['gaussian_ksize'])
        if ksize > 0:
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
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            gray = cv2.LUT(gray, table)

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
# Feature Extraction (includes new features)
# ============================================================================

class FeatureExtractor:
    @staticmethod
    def extract_features(image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        # Base features
        features = {
            'blur': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'contrast': gray.std(),
            'brightness': gray.mean(),
            'noise': np.std(cv2.Laplacian(gray, cv2.CV_64F)),
            'edge_density': float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
            'entropy': float(-np.sum(h_val * np.log2(h_val + 1e-10)))
                       if (h_val := np.histogram(gray, bins=256, range=(0, 256),
                                                  density=True)[0]) is not None else 0.0,
            'dynamic_range': float(np.percentile(gray, 95) - np.percentile(gray, 5)),
        }

        # New features
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        features['text_density'] = float(np.mean(binary > 0))

        # Skew angle
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=w // 4, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                if abs(dx) > 0:
                    angle = np.degrees(np.arctan2(y2 - y1, dx))
                    if abs(angle) < 30:
                        angles.append(angle)
            features['skew_angle'] = abs(float(np.median(angles))) if angles else 0.0
        else:
            features['skew_angle'] = 0.0

        # Connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        features['connected_components'] = n_labels - 1

        # Aspect ratio
        features['aspect_ratio'] = w / h

        # Line count
        h_proj = np.sum(binary > 0, axis=1)
        threshold = w * 0.02
        in_line = False
        line_count = 0
        for val in h_proj:
            if val > threshold and not in_line:
                in_line = True
                line_count += 1
            elif val <= threshold:
                in_line = False
        features['line_count'] = line_count

        return features


# ============================================================================
# Data Loading
# ============================================================================

def collect_all_images(images_folder: str) -> List[Tuple[str, str]]:
    all_images = []
    for year_folder in sorted(os.listdir(images_folder)):
        year_path = os.path.join(images_folder, year_folder)
        if os.path.isdir(year_path):
            for img_file in sorted(os.listdir(year_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(year_path, img_file)
                    all_images.append((img_path, year_folder))
    return all_images


def load_ground_truth(labels_folder: str, label_suffix: str = '_label') -> Dict[str, str]:
    gt_dict = {}
    for year_folder in os.listdir(labels_folder):
        year_path = os.path.join(labels_folder, year_folder)
        if os.path.isdir(year_path):
            for label_file in os.listdir(year_path):
                if label_file.endswith('.docx'):
                    img_name = label_file.replace(f'{label_suffix}.docx', '')
                    label_path = os.path.join(year_path, label_file)
                    try:
                        gt_text = read_docx_text(label_path)
                        gt_dict[img_name] = gt_text
                    except:
                        pass
    return gt_dict


# ============================================================================
# Multi-Engine OCR Runner
# ============================================================================

class MultiEngineOCRRunner:
    """
    Shared PaddleOCR detection + multiple recognition engines.
    Detection runs once per image, recognition runs per engine.
    """

    def __init__(self, device: str = 'cuda', engines: List[str] = None):
        self.device = device
        self.engines = engines or ['vietocr', 'tesseract', 'easyocr']
        self.recognizers = {}

        # Shared detector: PaddleOCR
        print(f"  Loading PaddleOCR detector...")
        self.paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='vi',
            device='gpu' if 'cuda' in device else 'cpu',
        )

        # Recognition engines
        if 'vietocr' in self.engines:
            print(f"  Loading VietOCR...")
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = False
            config['device'] = device
            self.recognizers['vietocr'] = Predictor(config)

        if 'tesseract' in self.engines:
            print(f"  Loading Tesseract...")
            import pytesseract
            # Verify Vietnamese language pack
            langs = pytesseract.get_languages()
            if 'vie' not in langs:
                print(f"    WARNING: 'vie' not in Tesseract languages: {langs}")
                print(f"    Install: sudo apt install tesseract-ocr-vie")
            self.recognizers['tesseract'] = pytesseract

        if 'easyocr' in self.engines:
            print(f"  Loading EasyOCR...")
            import easyocr
            gpu = 'cuda' in device
            self.recognizers['easyocr'] = easyocr.Reader(['vi'], gpu=gpu, verbose=False)

        print(f"  Active engines: {list(self.recognizers.keys())}")

    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Run PaddleOCR detection, return list of cropped line images."""
        temp_path = f"/tmp/temp_detect_{os.getpid()}_{time.time()}.png"
        try:
            cv2.imwrite(temp_path, image)
            results = self.paddle_ocr.predict(input=temp_path)

            if results is None or len(results) == 0 or results[0] is None:
                return []

            polygons = results[0].get('dt_polys', [])
            if len(polygons) == 0:
                return []

            crops = []
            for polygon in polygons:
                cropped = crop_polygon(image, polygon)
                if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    crops.append(cropped)
            return crops
        except:
            return []
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _recognize_vietocr(self, crops: List[np.ndarray],
                           confidence_threshold: float = 0.8) -> str:
        predictor = self.recognizers['vietocr']
        texts = []
        for crop in crops:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text, prob = predictor.predict(pil_img, return_prob=True)
                if prob > confidence_threshold and text.strip():
                    texts.append(text.strip())
            except:
                continue
        return "\t".join(texts)

    def _recognize_tesseract(self, crops: List[np.ndarray]) -> str:
        pytesseract = self.recognizers['tesseract']
        texts = []
        for crop in crops:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(pil_img, lang='vie').strip()
                if text:
                    # Clean tesseract output: remove excessive whitespace
                    text = " ".join(text.split())
                    texts.append(text)
            except:
                continue
        return "\t".join(texts)

    def _recognize_easyocr(self, crops: List[np.ndarray]) -> str:
        reader = self.recognizers['easyocr']
        texts = []
        for crop in crops:
            try:
                results = reader.readtext(crop, detail=0)
                for text in results:
                    text = text.strip()
                    if text:
                        texts.append(text)
            except:
                continue
        return "\t".join(texts)

    def run_ocr(self, image: np.ndarray) -> Dict[str, str]:
        """
        Run detection once, then each recognition engine.
        Returns dict: {engine_name: recognized_text}
        """
        # Shared detection
        crops = self._detect(image)

        if not crops:
            return {engine: "" for engine in self.recognizers}

        # Recognition per engine
        results = {}
        for engine_name in self.recognizers:
            if engine_name == 'vietocr':
                results[engine_name] = self._recognize_vietocr(crops)
            elif engine_name == 'tesseract':
                results[engine_name] = self._recognize_tesseract(crops)
            elif engine_name == 'easyocr':
                results[engine_name] = self._recognize_easyocr(crops)

        return results


# ============================================================================
# Main Data Collection
# ============================================================================

def collect_data(
    images_folder: str,
    labels_folder: str,
    output_folder: str,
    samples_per_image: int = 50,
    device: str = 'cuda',
    engines: List[str] = None,
    label_suffix: str = '_label',
    checkpoint_interval: int = 10,
    resume: bool = False,
):
    global shutdown_requested

    engines = engines or ['vietocr', 'tesseract', 'easyocr']
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 70)
    print("MULTI-ENGINE DATA COLLECTION FOR SURROGATE MODEL")
    print("=" * 70)
    print(f"Engines: {engines}")
    print(f"Samples per image: {samples_per_image}")
    print(f"Device: {device}")
    print(f"Output: {output_folder}")

    # Load data
    print("\n Loading data...")
    all_images = collect_all_images(images_folder)
    gt_dict = load_ground_truth(labels_folder, label_suffix)
    print(f"   Found {len(all_images)} images")
    print(f"   Found {len(gt_dict)} ground truth entries")

    valid_images = [(path, year) for path, year in all_images
                    if Path(path).stem in gt_dict]
    print(f"   Valid images with GT: {len(valid_images)}")

    # Initialize components
    print("\n Initializing components...")
    ocr_runner = MultiEngineOCRRunner(device, engines)
    preprocessor = ImagePreprocessor()
    feature_extractor = FeatureExtractor()

    # LHS Sampler
    print("\n Generating LHS preprocessing samples...")
    lhs_sampler = LHSSampler(seed=42)
    global_param_samples = lhs_sampler.sample(samples_per_image)
    print(f"   Generated {len(global_param_samples)} combinations")

    validation_summary = sampler_validation_summary(global_param_samples, seed=42)
    validation_json = os.path.join(output_folder, 'sampling_validation.json')
    with open(validation_json, 'w') as f:
        json.dump(validation_summary, f, indent=2)
    pd.DataFrame(global_param_samples).to_csv(
        os.path.join(output_folder, 'sampled_candidate_pool.csv'),
        index=False,
    )
    print(f"   Sampling validation: {validation_json}")

    # Checkpoint
    all_data = []
    start_idx = 0
    checkpoint_path = os.path.join(output_folder, 'checkpoint.pkl')

    if resume and os.path.exists(checkpoint_path):
        print(f"\n Loading checkpoint...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        all_data = checkpoint['data']
        start_idx = checkpoint['processed_images']
        print(f"   Resumed from image {start_idx}/{len(valid_images)}")

    # Collect
    print("\n" + "=" * 70)
    print("COLLECTING DATA")
    print("=" * 70)

    total_expected = len(valid_images) * samples_per_image
    print(f"Expected total samples: {total_expected:,}")
    print(f"Expected OCR calls: detect={total_expected:,}, "
          f"recognize={total_expected:,} × {len(engines)} engines")

    start_time = time.time()
    active_engines = list(ocr_runner.recognizers.keys())

    for img_idx, (img_path, year) in enumerate(tqdm(
            valid_images[start_idx:],
            initial=start_idx,
            total=len(valid_images),
            desc="Processing images")):

        if shutdown_requested:
            break

        actual_idx = start_idx + img_idx
        img_name = Path(img_path).stem
        gt_text = gt_dict[img_name]

        image = cv2.imread(img_path)
        if image is None:
            print(f"\n Cannot load: {img_path}")
            continue

        # Extract features (once per image)
        orig_features = feature_extractor.extract_features(image)

        # Run each preprocessing combo
        for params in global_param_samples:
            processed = preprocessor.preprocess(image, params)

            # OCR: detect once + recognize with each engine
            ocr_results = ocr_runner.run_ocr(processed)

            # Compute CER per engine
            cer_values = {}
            for engine_name, prediction in ocr_results.items():
                cer_values[f'cer_{engine_name}'] = compute_cer(prediction, gt_text)

            data_point = {
                'image_id': actual_idx,
                'image_name': img_name,
                'year': year,
                # Original features
                'orig_blur': orig_features['blur'],
                'orig_contrast': orig_features['contrast'],
                'orig_brightness': orig_features['brightness'],
                'orig_noise': orig_features['noise'],
                'orig_edge_density': orig_features['edge_density'],
                'orig_entropy': orig_features['entropy'],
                'orig_dynamic_range': orig_features['dynamic_range'],
                'orig_text_density': orig_features['text_density'],
                'orig_skew_angle': orig_features['skew_angle'],
                'orig_connected_components': orig_features['connected_components'],
                'orig_aspect_ratio': orig_features['aspect_ratio'],
                'orig_line_count': orig_features['line_count'],
                # Preprocessing params
                'gaussian_ksize': params['gaussian_ksize'],
                'bilateral_d': params['bilateral_d'],
                'clahe_clip': params['clahe_clip'],
                'gamma': params['gamma'],
                'sharpen': params['sharpen'],
                # CER per engine
                **cer_values,
            }
            all_data.append(data_point)

        # Checkpoint
        if (actual_idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'data': all_data,
                    'processed_images': actual_idx + 1,
                }, f)

            elapsed = time.time() - start_time
            rate = (img_idx + 1) / elapsed * 60
            remaining = (len(valid_images) - actual_idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"    Checkpoint: {actual_idx + 1} images, "
                       f"{len(all_data):,} samples, "
                       f"ETA: {remaining:.1f} min")

    # Save
    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)

    # Pickle (full data)
    output_pkl = os.path.join(output_folder, 'preprocessing_data.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"  Pickle: {output_pkl}")

    # CSV (main output)
    df = pd.DataFrame(all_data)
    output_csv = os.path.join(output_folder, 'preprocessing_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"  CSV: {output_csv}")

    # Also save per-engine CSVs (for direct use with train_surrogate.py)
    cer_cols = [c for c in df.columns if c.startswith('cer_')]
    non_cer_cols = [c for c in df.columns if not c.startswith('cer_')]

    for cer_col in cer_cols:
        engine_name = cer_col.replace('cer_', '')
        engine_df = df[non_cer_cols + [cer_col]].copy()
        engine_df = engine_df.rename(columns={cer_col: 'cer'})
        engine_csv = os.path.join(output_folder, f'preprocessing_data_{engine_name}.csv')
        engine_df.to_csv(engine_csv, index=False)
        print(f"  {engine_name} CSV: {engine_csv}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_images = len(set(d['image_name'] for d in all_data)) if all_data else 0
    print(f"Total images: {n_images}")
    print(f"Total data points: {len(all_data):,}")
    print(f"Samples per image: {samples_per_image}")

    if all_data:
        print(f"\nCER Statistics per engine:")
        for cer_col in cer_cols:
            values = [d[cer_col] for d in all_data if cer_col in d]
            if values:
                engine = cer_col.replace('cer_', '')
                print(f"  {engine:12s}: mean={np.mean(values):.4f}, "
                      f"std={np.std(values):.4f}, "
                      f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total / 60:.1f} min ({elapsed_total / 3600:.2f} hours)")

    return df


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Engine Data Collection for Surrogate Model")

    parser.add_argument('--images-folder', type=str,
                        default=str(PROJECT_ROOT / 'datasets/300_images/300_images'),
                        help='Path to images folder')
    parser.add_argument('--labels-folder', type=str,
                        default=str(PROJECT_ROOT / 'datasets/300_images_labels/300_images_labels'),
                        help='Path to labels folder')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Output folder')
    parser.add_argument('--samples-per-image', type=int, default=50,
                        help='Number of preprocessing combos per image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--engines', nargs='+',
                        default=['vietocr', 'tesseract', 'easyocr'],
                        choices=['vietocr', 'tesseract', 'easyocr'],
                        help='Recognition engines to use')
    parser.add_argument('--label-suffix', type=str, default='_label')
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    if args.output_folder is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_folder = str(
            PROJECT_ROOT / f'outputs/multiengine_data_{timestamp}')

    collect_data(
        images_folder=args.images_folder,
        labels_folder=args.labels_folder,
        output_folder=args.output_folder,
        samples_per_image=args.samples_per_image,
        device=args.device,
        engines=args.engines,
        label_suffix=args.label_suffix,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
