"""Load the reported VietOCR model without contacting vocr.vn."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


EXPECTED_WEIGHTS_SHA256 = (
    "380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59"
)
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = CODE_ROOT / "config/vietocr_vgg_transformer_offline.yml"
DEFAULT_WEIGHTS_PATH = Path("/tmp/vgg_transformer.pth")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vietocr_config(device: str):
    from vietocr.tool.config import Cfg

    config_path = Path(
        os.environ.get("VIETOCR_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    ).resolve()
    weights_path = Path(
        os.environ.get("VIETOCR_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))
    ).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Offline VietOCR config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Cached VietOCR weights not found: {weights_path}")

    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            "VietOCR weights hash mismatch: "
            f"expected {EXPECTED_WEIGHTS_SHA256}, got {weights_sha256} "
            f"for {weights_path}"
        )

    config = Cfg.load_config_from_file(str(config_path))
    config["weights"] = str(weights_path)
    config["pretrain"] = str(weights_path)
    config["cnn"]["pretrained"] = False
    config["device"] = device
    print(f"    VietOCR config: {config_path}")
    print(f"    VietOCR weights: {weights_path}")
    print(f"    VietOCR weights SHA-256: {weights_sha256}")
    return config
