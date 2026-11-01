"""Versioned best-of-K statistics from completed per-page outputs only.

No OCR or fits. Five contrasts per block, 12 tests per VN contrast and eight
per SROIE contrast. Reuse established percentile/bootstrap seed conventions.
"""
import argparse
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from export_sroie_casefold_sensitivity import (
    KS, CONTRASTS, sha, seed_for, paired_ci, pvalue, holm,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {"raw": ("vietocr", "tesseract", "easyocr"),
           "contour": ("vietocr", "tesseract", "easyocr"),
           "sroie": ("tesseract", "easyocr")}


def aligned(frame, ids, count):
    if len(frame) != count or frame.image_name.nunique() != count:
        raise ValueError("Wrong/duplicate page coverage")
    result = frame.set_index("image_name").sort_index()
    if ids is not None and not result.index.equals(ids):
        raise ValueError("Page cohort mismatch")
    return result


def calculate(config):
    if set(config) != set(ENGINES):
        raise ValueError("Require exactly raw, contour and sroie blocks")
    paired, baselines, sources = [], [], {}
    for block, engines in ENGINES.items():
        paths = config[block]
        if set(paths) != {"surrogate", "nn", "folds"}:
            raise ValueError("Only surrogate/nn/folds paths are configurable")
        paths = {k: ROOT / v for k, v in paths.items()}
        count = 616 if block == "sroie" else 300
        folds = aligned(pd.read_csv(paths["folds"]), None, count)
        if set(folds.fold) != set(range(1, 6)):
            raise ValueError("Require five frozen folds")
        nn_path = paths["nn"] / "nn_per_image.csv"
        nn = pd.read_csv(nn_path)
        if set(nn.engine) != set(engines) or len(nn) != count * len(engines):
            raise ValueError("Wrong NN engine/page coverage")
        sources[block] = {str(nn_path): sha(nn_path), str(paths["folds"]): sha(paths["folds"])}
        for engine in engines:
            path = paths["surrogate"] / f"per_page_results_{engine}.csv"
            sources[block][str(path)] = sha(path)
            s = aligned(pd.read_csv(path), folds.index, count)
            n = aligned(nn[nn.engine == engine], folds.index, count)
            np.testing.assert_array_equal(n.fold, folds.fold)
            for name in ["no_preprocessing", "oracle", "fixed_best"]:
                values = s[name].to_numpy()
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid baseline CER")
                baselines.append(dict(block=block, engine=engine, policy=name,
                                      mean_cer_pct=100 * values.mean()))
            for k in KS:
                for nc, sc in [(f"surr_k{k}_cer", f"surr_k{k}"),
                               (f"random_k{k}_cer", f"rand_k{k}"),
                               (f"fixed_topk_k{k}_cer", f"fixed_topk_k{k}")]:
                    np.testing.assert_allclose(n[nc], s[sc], rtol=0, atol=1e-12)
                vals = {"Surrogate": s[f"surr_k{k}"], "Random": s[f"rand_k{k}"],
                        "FixedTopK": s[f"fixed_topk_k{k}"], "NN": n[f"nn_k{k}_cer"]}
                for v in vals.values():
                    if not np.isfinite(v).all() or (v < 0).any():
                        raise ValueError("Invalid policy CER")
                for page in folds.index:
                    paired.append(dict(block=block, engine=engine, k=k, image_name=page,
                        fold=int(folds.loc[page, "fold"]), **{p: float(v.loc[page]) for p, v in vals.items()}))
    paired = pd.DataFrame(paired)
    rows = []
    for (block, engine, k), group in paired.groupby(["block", "engine", "k"], sort=True):
        for a, b in CONTRASTS:
            x, y = group[a].to_numpy(), group[b].to_numpy()
            diff = x - y
            seed = seed_for(engine, k, a, b)
            lo, hi = paired_ci(diff, seed)
            rows.append(dict(block=block, engine=engine, k=k, method_a=a, method_b=b,
                n=len(diff), mean_a_pct=100*x.mean(), mean_b_pct=100*y.mean(),
                difference_pp=100*diff.mean(), ci_low_pp=100*lo, ci_high_pp=100*hi,
                p_two_sided=pvalue(x,y), n_zero=int((diff == 0).sum()),
                bootstrap_seed=seed, bootstrap_replicates=10000,
                alternative="two-sided", zero_method="wilcox",
                ci_method="paired_page_percentile", ci_multiplicity_adjusted=False))
    stats = pd.DataFrame(rows)
    for (block, _, _), idx in stats.groupby(["block", "method_a", "method_b"]).groups.items():
        expected = 8 if block == "sroie" else 12
        if len(idx) != expected:
            raise ValueError("Incomplete prespecified Holm family")
        stats.loc[idx, "p_holm"] = holm(stats.loc[idx, "p_two_sided"])
        stats.loc[idx, "holm_family_size"] = expected
    if len(stats) != 160:
        raise ValueError("Expected 160 comparisons")
    return paired, stats, pd.DataFrame(baselines), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing existing output; choose a new versioned directory")
    paired, stats, baselines, sources = calculate(json.loads(args.config.read_text()))
    args.output_dir.mkdir(parents=True)
    outputs = {"paired_values.csv": paired, "statistics.csv": stats, "baseline_means.csv": baselines}
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    metadata = dict(protocol="bestk_paired_statistics_v1", sources=sources,
        config=dict(path=str(args.config.resolve()), sha256=sha(args.config)),
        scripts={str(Path(__file__).resolve()): sha(__file__),
                 "export_sroie_casefold_sensitivity.py": sha(ROOT / "analysis/export_sroie_casefold_sensitivity.py")},
        outputs={name: sha(args.output_dir / name) for name in outputs},
        versions={p: version(p) for p in ["numpy", "pandas", "scipy"]},
        metric=dict(raw="VN v3 NFC/whitespace case-sensitive", contour="VN v3 NFC/whitespace case-sensitive",
                    sroie="NFC/whitespace casefold; primary transition pending publication sync"),
        bootstrap="10,000 paired percentile resamples; per-row stored seed; pointwise 95%, not adjusted",
        holm="Separate policy-contrast/evaluation-block families; VN12, SROIE8; no reverse duplicates")
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(paired)} paired rows and {len(stats)} comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
