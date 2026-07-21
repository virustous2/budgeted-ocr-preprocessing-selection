#!/usr/bin/env python3
"""
Bootstrap confidence intervals for shortlist paired differences.

This script does not run OCR. It reconstructs per-image Surrogate K and
Random K / FixedTopK best-of-K CER values from an existing preprocessing_data.csv
using the same nested GroupKFold logic as main/train_surrogate_multiengine_kfold.py,
then bootstraps the paired page-level differences.

Outputs:
  - bootstrap_per_image.csv: one row per engine/image/K with paired CER values
  - bootstrap_summary.csv: mean differences and percentile CIs
  - bootstrap_summary.json: same summary in JSON form

If bootstrap_per_image.csv already exists, pass it via --per-image-input to
skip model fitting and recompute CIs only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GROUP_COL = "image_name"

FEATURE_COLS_BASE = [
    "orig_blur",
    "orig_contrast",
    "orig_brightness",
    "orig_noise",
    "orig_edge_density",
    "orig_entropy",
    "orig_dynamic_range",
]

FEATURE_COLS_NEW = [
    "orig_text_density",
    "orig_skew_angle",
    "orig_connected_components",
    "orig_aspect_ratio",
    "orig_line_count",
]

PARAM_COLS = [
    "gaussian_ksize",
    "bilateral_d",
    "clahe_clip",
    "gamma",
    "sharpen",
]


def combo_mask(frame: pd.DataFrame, combo_vals) -> np.ndarray:
    """Return rows matching one preprocessing combo, tolerant to float roundoff."""
    mask = np.ones(len(frame), dtype=bool)
    for col, val in zip(PARAM_COLS, combo_vals):
        mask &= np.isclose(frame[col].values, val)
    return mask


def prepare_data(df: pd.DataFrame) -> list[str]:
    feature_cols = FEATURE_COLS_BASE.copy()
    feature_cols.extend([c for c in FEATURE_COLS_NEW if c in df.columns])
    input_cols = feature_cols + PARAM_COLS

    interactions = [
        ("orig_blur", "gaussian_ksize"),
        ("orig_contrast", "clahe_clip"),
        ("orig_brightness", "gamma"),
        ("orig_noise", "bilateral_d"),
        ("orig_entropy", "sharpen"),
    ]
    for feat, param in interactions:
        if feat in df.columns and param in df.columns:
            col_name = f"inter_{feat.replace('orig_', '')}_{param}"
            df[col_name] = df[feat] * df[param]
            input_cols.append(col_name)

    return input_cols


def load_training_helpers():
    try:
        from main.train_surrogate_multiengine_kfold import get_models, select_best_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing model-training dependency while importing "
            "main/train_surrogate_multiengine_kfold.py. Run this script in the same "
            "environment used for surrogate training, with xgboost and lightgbm installed."
        ) from exc
    return get_models, select_best_model


def normalize_engine_name(name: str) -> str:
    return name[4:] if name.startswith("cer_") else name


def infer_engines(df: pd.DataFrame, requested: Iterable[str] | None) -> list[str]:
    available = sorted(c[4:] for c in df.columns if c.startswith("cer_"))
    if requested:
        requested = [normalize_engine_name(e) for e in requested]
        missing = [e for e in requested if f"cer_{e}" not in df.columns]
        if missing:
            raise ValueError(f"Requested engines not found in data: {missing}")
        return requested
    return available


def reconstruct_per_image_results(
    data_path: Path,
    engines: list[str] | None,
    k_values: list[int],
    n_folds: int,
    random_trials: int,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    input_cols = prepare_data(df)
    engines = infer_engines(df, engines)
    get_models, select_best_model = load_training_helpers()
    models = get_models()

    rows = []
    groups = df[GROUP_COL].values
    outer_gkf = GroupKFold(n_splits=n_folds)

    for engine in engines:
        target_col = f"cer_{engine}"
        print(f"[{engine}] reconstructing nested out-of-fold shortlists")

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_gkf.split(df[input_cols].values, df[target_col].values, groups), start=1):
            train_df = df.iloc[tr_idx].copy()
            test_df = df.iloc[te_idx].copy()

            best_model_name = select_best_model(train_df, target_col, input_cols, verbose=False)
            model = clone(models[best_model_name])
            model.fit(train_df[input_cols].values, train_df[target_col].values)

            test_eval = test_df.copy()
            test_eval["pred_cer"] = model.predict(test_df[input_cols].values)
            print(f"  fold {fold_idx}/{n_folds}: {best_model_name}")

            combo_train_mean = train_df.groupby(PARAM_COLS)[target_col].mean()
            fixed_topk_combos = {
                k: combo_train_mean.nsmallest(k).index
                for k in k_values
            }

            for image_name, group_pred in test_eval.groupby(GROUP_COL):
                group_actual = test_df[test_df[GROUP_COL] == image_name]
                for k in k_values:
                    sample_size = min(k, len(group_actual))
                    top_k = group_pred.nsmallest(sample_size, "pred_cer")
                    surr_cer = float(top_k[target_col].min())
                    random_cers = [
                        float(group_actual.sample(sample_size, random_state=s)[target_col].min())
                        for s in range(random_trials)
                    ]
                    rand_cer = float(np.mean(random_cers))

                    fixed_topk_mask = np.zeros(len(group_actual), dtype=bool)
                    for combo_vals in fixed_topk_combos[k]:
                        fixed_topk_mask |= combo_mask(group_actual, combo_vals)
                    if not np.any(fixed_topk_mask):
                        raise ValueError(
                            f"FixedTopK K={k} did not match image {image_name} "
                            f"in fold {fold_idx} for engine {engine}"
                        )
                    fixed_topk_cer = float(group_actual.loc[fixed_topk_mask, target_col].min())

                    rows.append(
                        {
                            "engine": engine,
                            "fold": fold_idx,
                            "image_name": image_name,
                            "k": k,
                            "surr_cer": surr_cer,
                            "rand_cer": rand_cer,
                            "fixed_topk_cer": fixed_topk_cer,
                            "diff": surr_cer - rand_cer,
                            "diff_vs_random": surr_cer - rand_cer,
                            "diff_vs_fixed_topk": surr_cer - fixed_topk_cer,
                        }
                    )

    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[indices].mean(axis=1)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bca_ci(values: np.ndarray, n_bootstrap: int, seed: int, alpha: float) -> tuple[float, float]:
    result = bootstrap(
        (values,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def stable_group_seed(base_seed: int, engine: str, k: int, comparator: str) -> int:
    # Avoid Python's salted hash so results are stable across processes.
    token = f"{engine}:{k}:{comparator}".encode("utf-8")
    checksum = sum((idx + 1) * byte for idx, byte in enumerate(token))
    return int((base_seed + checksum) % (2**32 - 1))


def summarize(per_image: pd.DataFrame, n_bootstrap: int, seed: int, alpha: float, ci_method: str) -> pd.DataFrame:
    rows = []
    for (engine, k), group in per_image.groupby(["engine", "k"], sort=True):
        surr = group["surr_cer"].to_numpy(dtype=float)
        comparator_specs = [
            ("random", "rand_cer", "diff_vs_random" if "diff_vs_random" in group.columns else "diff"),
        ]
        if "fixed_topk_cer" in group.columns:
            comparator_specs.append(("fixed_topk", "fixed_topk_cer", "diff_vs_fixed_topk"))

        for comparator, baseline_col, diff_col in comparator_specs:
            baseline = group[baseline_col].to_numpy(dtype=float)
            diff = group[diff_col].to_numpy(dtype=float)
            group_seed = stable_group_seed(seed, str(engine), int(k), comparator)
            if ci_method == "bca":
                ci_low, ci_high = bca_ci(diff, n_bootstrap=n_bootstrap, seed=group_seed, alpha=alpha)
            else:
                ci_low, ci_high = bootstrap_ci(diff, n_bootstrap=n_bootstrap, seed=group_seed, alpha=alpha)
            p_value = 1.0
            if np.any(diff != 0):
                p_value = float(wilcoxon(surr, baseline, alternative="two-sided").pvalue)
            rows.append(
                {
                    "engine": engine,
                    "k": int(k),
                    "comparator": comparator,
                    "n_images": int(len(group)),
                    "surr_mean": float(surr.mean()),
                    "baseline_mean": float(baseline.mean()),
                    "rand_mean": float(group["rand_cer"].mean()) if "rand_cer" in group.columns else float("nan"),
                    "fixed_topk_mean": float(group["fixed_topk_cer"].mean()) if "fixed_topk_cer" in group.columns else float("nan"),
                    "diff_mean": float(diff.mean()),
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                    "surr_mean_pct": float(100 * surr.mean()),
                    "baseline_mean_pct": float(100 * baseline.mean()),
                    "rand_mean_pct": float(100 * group["rand_cer"].mean()) if "rand_cer" in group.columns else float("nan"),
                    "fixed_topk_mean_pct": float(100 * group["fixed_topk_cer"].mean()) if "fixed_topk_cer" in group.columns else float("nan"),
                    "diff_mean_pp": float(100 * diff.mean()),
                    "diff_ci_low_pp": float(100 * ci_low),
                    "diff_ci_high_pp": float(100 * ci_high),
                    "wilcoxon_p": p_value,
                    "wilcoxon_alternative": "two-sided",
                    "ci_method": ci_method,
                    "bootstrap_seed": group_seed,
                }
            )
    return pd.DataFrame(rows).sort_values(["engine", "k", "comparator"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, help="Path to preprocessing_data.csv")
    parser.add_argument("--per-image-input", type=Path, help="Existing bootstrap_per_image.csv to reuse")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for bootstrap outputs")
    parser.add_argument("--engines", nargs="*", help="Engines to evaluate, e.g. vietocr tesseract easyocr")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10, 15])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--ci-method",
        choices=["percentile", "bca"],
        default="percentile",
        help="Bootstrap CI method for paired mean differences",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_image_input:
        per_image = pd.read_csv(args.per_image_input)
    else:
        if not args.data_path:
            raise SystemExit("Provide --data-path, or use --per-image-input to reuse existing per-image results.")
        per_image = reconstruct_per_image_results(
            data_path=args.data_path,
            engines=args.engines,
            k_values=args.k_values,
            n_folds=args.n_folds,
            random_trials=args.random_trials,
        )

    per_image_path = args.output_dir / "bootstrap_per_image.csv"
    summary_path = args.output_dir / "bootstrap_summary.csv"
    json_path = args.output_dir / "bootstrap_summary.json"

    per_image.to_csv(per_image_path, index=False)
    summary = summarize(
        per_image,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        alpha=args.alpha,
        ci_method=args.ci_method,
    )
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    metadata = {
        "data_path": str(args.data_path) if args.data_path else None,
        "per_image_input": str(args.per_image_input) if args.per_image_input else None,
        "engines": args.engines,
        "k_values": args.k_values,
        "n_folds": args.n_folds,
        "random_trials": args.random_trials,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "ci_method": args.ci_method,
        "estimand": (
            "Bootstrap CIs are computed over paired page-level differences "
            "between Surrogate K and each comparator's best-of-K CER. Random K is "
            "estimated as the mean over random_trials reproducible random draws "
            "from each page's already evaluated candidate pool; FixedTopK is the "
            "fold-specific static shortlist of K best-average training candidates. "
            "The CI reflects uncertainty across pages rather than random-draw "
            "Monte Carlo variance."
        ),
    }
    metadata_path = args.output_dir / "bootstrap_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved per-image results: {per_image_path}")
    print(f"Saved bootstrap summary: {summary_path}")
    print(f"Saved bootstrap JSON: {json_path}")
    print(f"Saved bootstrap metadata: {metadata_path}")
    print()
    print(summary[["engine", "k", "comparator", "n_images", "diff_mean_pp", "diff_ci_low_pp", "diff_ci_high_pp", "wilcoxon_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
