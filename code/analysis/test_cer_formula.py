#!/usr/bin/env python3
"""Exploratory comparison of whitespace-only and NFC-normalized CER.

This script is not part of the reported evaluation. The manuscript's reported
CER protocol uses whitespace normalization only; the NFC branch below is kept
solely as a synthetic diagnostic of a possible alternative protocol.
"""
import argparse
import sys
import unicodedata
from pathlib import Path
from docx import Document

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

# ── two formulas ──────────────────────────────────────────────────────────────

def cer_old(prediction: str, ground_truth: str) -> float:
    from jiwer import cer as jiwer_cer
    pred = " ".join(prediction.split())
    gt   = " ".join(ground_truth.split())
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return float(jiwer_cer(gt, pred))


def cer_new(prediction: str, ground_truth: str) -> float:
    from jiwer import cer as jiwer_cer

    def normalize(text):
        text = unicodedata.normalize("NFC", text)
        return " ".join(text.split())

    pred = normalize(prediction)
    gt   = normalize(ground_truth)
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return float(jiwer_cer(gt, pred))


# ── helpers ───────────────────────────────────────────────────────────────────

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\t".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def run_vietocr(img_path: str) -> str:
    import cv2
    from PIL import Image
    from paddleocr import PaddleOCR
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    paddle = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='vi', device='cpu',
    )
    cfg = Cfg.load_config_from_name('vgg_transformer')
    cfg['cnn']['pretrained'] = False
    cfg['device'] = 'cpu'
    model = Predictor(cfg)

    results = paddle.predict(input=img_path)
    polygons = results[0].get('dt_polys', []) if results and results[0] else []

    image = cv2.imread(img_path)
    texts = []
    for poly in polygons:
        poly = __import__('numpy').array(poly, dtype=__import__('numpy').int32)
        x, y, w, h = cv2.boundingRect(poly)
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            text, prob = model.predict(pil, return_prob=True)
            if prob > 0.8 and text.strip():
                texts.append(text.strip())
        except Exception:
            continue
    return "\t".join(texts)


# ── main ──────────────────────────────────────────────────────────────────────

def main(labels_root: Path):
    label_files = sorted(labels_root.rglob('*.docx'))[:2]

    print("=" * 60)
    print("EXPLORATORY CER FORMULA COMPARISON (not the reported protocol)")
    print("=" * 60)

    for label_path in label_files:
        name = label_path.stem
        gt = read_docx(str(label_path))

        # Simulate a prediction with ~5% errors: drop every 20th char
        pred_chars = list(gt)
        pred_chars = [c for i, c in enumerate(pred_chars) if i % 20 != 0]
        pred = "".join(pred_chars)

        old = cer_old(pred, gt)
        new = cer_new(pred, gt)
        diff = new - old

        gt_nfc = unicodedata.normalize("NFC", gt)
        nfc_changed_gt   = gt != gt_nfc
        nfc_changed_pred = pred != unicodedata.normalize("NFC", pred)

        print(f"\nFile: {name}")
        print(f"  GT length: {len(gt)} chars")
        print(f"  GT (first 100): {gt[:100]!r}")
        print(f"  NFC changed GT?   {nfc_changed_gt}")
        print(f"  NFC changed pred? {nfc_changed_pred}")
        print(f"  CER old : {old*100:.4f}%")
        print(f"  CER new : {new*100:.4f}%")
        print(f"  Diff    : {diff*100:+.6f}%")

    # Synthetic test: NFD vs NFC — explicit case where NFC matters
    print("\n" + "=" * 60)
    print("SYNTHETIC TEST: NFD (decomposed) vs NFC (precomposed)")
    nfc_text = "Hà Nội"
    nfd_text = unicodedata.normalize("NFD", nfc_text)
    print(f"  NFC: {nfc_text!r}  len={len(nfc_text)}")
    print(f"  NFD: {nfd_text!r}  len={len(nfd_text)}")
    print(f"  cer_old(NFD, NFC) = {cer_old(nfd_text, nfc_text)*100:.4f}%")
    print(f"  cer_new(NFD, NFC) = {cer_new(nfd_text, nfc_text)*100:.4f}%")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore whitespace-only and NFC-normalized CER.')
    parser.add_argument('--labels-dir', default=str(RELEASE_ROOT / 'data/source_documents_not_included/vietnamese/references'))
    args = parser.parse_args()
    main(Path(args.labels_dir))
