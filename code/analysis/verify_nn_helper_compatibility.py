"""Replay saved NN helper tests without OCR, NN retrieval, fitting or overwrite."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluate_nn_shortlist_baseline import safe_wilcoxon

PAIRS = {"nn_vs_fixed_topk": ("nn", "fixed_topk"),
         "fixed_topk_vs_nn": ("fixed_topk", "nn"),
         "nn_vs_random": ("nn", "random"),
         "nn_vs_surrogate": ("nn", "surr"),
         "surrogate_vs_nn": ("surr", "nn")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(config, root):
    checks, sources = [], {}
    for block, paths in config.items():
        folder = root / paths["nn"]
        per_path, metrics_path = folder / "nn_per_image.csv", folder / "nn_metrics.json"
        frame = pd.read_csv(per_path)
        metrics = json.loads(metrics_path.read_text())
        sources[str(per_path)] = sha(per_path)
        sources[str(metrics_path)] = sha(metrics_path)
        for engine, group in frame.groupby("engine"):
            for k in (3, 5, 10, 15):
                for label, (a, b) in PAIRS.items():
                    x, y = group[f"{a}_k{k}_cer"].to_numpy(), group[f"{b}_k{k}_cer"].to_numpy()
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise AssertionError("Nonfinite current comparison")
                    got = safe_wilcoxon(x, y)
                    expected = 1.0 if np.all(x == y) else float(wilcoxon(
                        x, y, alternative="two-sided", zero_method="wilcox").pvalue)
                    saved = metrics[engine]["wilcoxon"][f"{label}_k{k}"]["p"]
                    if got["p"] != expected or not math.isfinite(saved) or got["p"] != saved:
                        raise AssertionError(f"Helper/saved/direct mismatch: {block}/{engine}/{label}/{k}")
                    checks.append({"block": block, "engine": engine, "k": k, "contrast": label,
                                   "p": got["p"], "delta_saved_p": got["p"] - saved,
                                   "false_allclose": bool(np.allclose(x, y) and not np.all(x == y))})
    return {"scope": "saved helper p-values only, no fit or OCR", "comparisons": len(checks),
            "max_absolute_saved_p_change": max(abs(r["delta_saved_p"]) for r in checks),
            "false_allclose_comparisons": sum(r["false_allclose"] for r in checks),
            "source_sha256": sources, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing existing output directory")
    result = verify(json.loads(args.config.read_text()), args.root)
    result.update(config_sha256=sha(args.config), verifier_sha256=sha(__file__),
                  helper_sha256=sha(Path(__file__).with_name("evaluate_nn_shortlist_baseline.py")))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "nn_helper_compatibility.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["comparisons", "max_absolute_saved_p_change",
                                            "false_allclose_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
