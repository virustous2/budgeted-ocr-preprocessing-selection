"""Report imported cv2 separately from wheel metadata; never install packages.

Default mode reports problems without failing; --require-single-distribution
enforces a single matching OpenCV wheel for a prospective clean environment.
This is a current-runtime diagnostic, not a reconstruction of historical runs.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

PACKAGES = {"opencv-python", "opencv-python-headless",
            "opencv-contrib-python", "opencv-contrib-python-headless"}


def diagnose(runtime_version, package_versions):
    installed = {name: value for name, value in sorted(package_versions.items()) if name in PACKAGES}
    matching = {name: bool(runtime_version) and
                value.split(".")[:3] == runtime_version.split(".")[:3]
                for name, value in installed.items()}
    single = len(installed) == 1
    clean = bool(runtime_version) and single and all(matching.values())
    return {"runtime_version": runtime_version, "distribution_versions": installed,
            "single_cv2_distribution": single, "metadata_runtime_agreement": matching,
            "clean_single_distribution": clean,
            "status": "single_matching_distribution" if clean else "requires_review",
            "scope": "current imported runtime only; historical OCR runtime not inferred"}


def collect():
    import cv2
    versions = {d.metadata.get("Name", "").lower().replace("_", "-"): d.version
                for d in distributions()}
    report = diagnose(cv2.__version__, versions)
    module = Path(cv2.__file__)
    report.update(python=platform.python_version(), observed_at_utc=datetime.now(timezone.utc).isoformat(),
                  module_filename=module.name,
                  module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return report


def save_new_report(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "opencv_environment.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Fresh directory; existing paths are refused")
    parser.add_argument("--require-single-distribution", action="store_true")
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("Refusing existing output directory")
    report = collect()
    if args.output_dir is not None:
        save_new_report(args.output_dir, report)
    print(json.dumps(report, indent=2))
    if args.require_single_distribution and not report["clean_single_distribution"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
