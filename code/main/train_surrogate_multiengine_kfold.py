#!/usr/bin/env python3
"""
K-fold cross-validation evaluation of surrogate models (5-fold GroupKFold).

Unlike train_surrogate_multiengine.py (single 80/20 split), this script:
  - Splits all images into 5 non-overlapping outer folds
  - Every image appears exactly once in an outer test fold
  - Wilcoxon runs on pooled per-image results across all outer folds
  - Reports mean ± std metrics across folds

Model selection is nested: each outer fold selects the model family using
inner GroupKFold on that fold's training images only. A final deployable model
is then selected on all available images and saved for inference.

Note: the best_model field in saved results is the final refit model family for
inference artifacts. Reported CV metrics and Wilcoxon tests come only from
outer folds with nested model selection; they do not use this final refit model.

Usage:
    python main/train_surrogate_multiengine_kfold.py \
        --data-path data/preprocessing_response_tables/raw_preprocessing_data.csv \
        --nopre-path /path/to/exact_nopre_per_image.csv
"""

import os
import json
import pickle
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

try:
    from random_baseline import expected_minimum_without_replacement
except ImportError:
    from main.random_baseline import expected_minimum_without_replacement

SCRIPT_DIR = Path(__file__).parent.resolve()
CODE_ROOT = SCRIPT_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent

GROUP_COL = 'image_name'

FEATURE_COLS_BASE = [
    'orig_blur', 'orig_contrast', 'orig_brightness', 'orig_noise',
    'orig_edge_density', 'orig_entropy', 'orig_dynamic_range',
]

FEATURE_COLS_NEW = [
    'orig_text_density', 'orig_skew_angle', 'orig_connected_components',
    'orig_aspect_ratio', 'orig_line_count',
]

PARAM_COLS = [
    'gaussian_ksize', 'bilateral_d', 'clahe_clip', 'gamma', 'sharpen',
]


def combo_mask(frame: pd.DataFrame, combo_vals) -> np.ndarray:
    """Return rows matching one preprocessing combo, tolerant to float roundoff."""
    mask = np.ones(len(frame), dtype=bool)
    for col, val in zip(PARAM_COLS, combo_vals):
        mask &= np.isclose(frame[col].values, val)
    return mask


# ============================================================================
# Models
# ============================================================================

def get_models() -> Dict:
    return {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
        ]),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            objective='reg:pseudohubererror', subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            objective='huber', subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


# ============================================================================
# Prepare Data
# ============================================================================

def prepare_data(df: pd.DataFrame) -> List[str]:
    """Add interaction terms, return input_cols list."""
    feature_cols = FEATURE_COLS_BASE.copy()
    feature_cols.extend([c for c in FEATURE_COLS_NEW if c in df.columns])
    input_cols = feature_cols + PARAM_COLS

    interactions = [
        ('orig_blur', 'gaussian_ksize'),
        ('orig_contrast', 'clahe_clip'),
        ('orig_brightness', 'gamma'),
        ('orig_noise', 'bilateral_d'),
        ('orig_entropy', 'sharpen'),
    ]
    for feat, param in interactions:
        if feat in df.columns and param in df.columns:
            col_name = f'inter_{feat.replace("orig_", "")}_{param}'
            df[col_name] = df[feat] * df[param]
            input_cols.append(col_name)

    return input_cols


def resolve_exact_nopre_csv(data_path: str, explicit_path: Optional[str] = None) -> Path:
    if not explicit_path:
        raise ValueError(
            "The public release requires --nopre-path with the exact NoPre "
            "per-image CSV; source OCR outputs are not bundled."
        )

    csv_path = Path(explicit_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Exact NoPre CSV not found: {csv_path}")
    return csv_path


def load_exact_nopre_maps(csv_path: Path, engine_list: List[str]) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    if 'file_name' not in df.columns:
        raise ValueError(f"Expected 'file_name' column in exact NoPre CSV: {csv_path}")

    df = df.copy()
    df['image_name'] = df['file_name'].map(lambda x: Path(str(x)).stem)
    if df['image_name'].duplicated().any():
        dupes = df.loc[df['image_name'].duplicated(), 'image_name'].tolist()[:5]
        raise ValueError(f"Duplicate image_name entries in exact NoPre CSV: {dupes}")

    exact_maps = {}
    for engine in engine_list:
        cer_col = f'cer_{engine}'
        if cer_col not in df.columns:
            raise ValueError(f"Missing column '{cer_col}' in exact NoPre CSV: {csv_path}")
        exact_maps[engine] = df.set_index('image_name')[cer_col].astype(float).to_dict()

    return exact_maps


# ============================================================================
# Model selection (inner CV on full data)
# ============================================================================

def select_best_model(
    df: pd.DataFrame,
    target_col: str,
    input_cols: List[str],
    verbose: bool = True,
) -> str:
    """Run inner GroupKFold(5) on the provided split to pick best model type."""
    models = get_models()
    X = df[input_cols].values
    y = df[target_col].values
    groups = df[GROUP_COL].values
    gkf = GroupKFold(n_splits=5)

    mae_scores = {}
    for name, model in models.items():
        fold_maes = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            m = clone(model)
            m.fit(X[tr_idx], y[tr_idx])
            fold_maes.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))
        mae_scores[name] = np.mean(fold_maes)
        if verbose:
            print(f"    {name:15s}: MAE={mae_scores[name]:.4f}")

    best_name = min(mae_scores, key=lambda m: mae_scores[m])
    if verbose:
        print(f"  → Best model: {best_name}")
    return best_name


def summarize_model_selection(selected_models_per_fold: List[str]) -> str:
    counts = Counter(selected_models_per_fold)
    return ", ".join(
        f"{name}×{count}" for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    )


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 42) -> List[float]:
    """Percentile bootstrap CI for a paired mean difference."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    if len(diff) == 0:
        return [float('nan'), float('nan')]
    sample_idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[sample_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return [float(lo), float(hi)]


# ============================================================================
# Feature importance
# ============================================================================

def extract_feature_importance(model, input_cols: List[str]) -> Dict[str, float]:
    m = model
    if hasattr(m, 'feature_importances_'):
        fi = m.feature_importances_
    elif hasattr(m, 'named_steps'):  # Ridge Pipeline
        fi = np.abs(m.named_steps['ridge'].coef_)
    else:
        fi = np.zeros(len(input_cols))
    fi_norm = fi / fi.sum() if fi.sum() > 0 else fi
    return dict(zip(input_cols, fi_norm.tolist()))


# ============================================================================
# K-fold evaluation
# ============================================================================

def train_one_engine_kfold(
    df: pd.DataFrame,
    target_col: str,
    input_cols: List[str],
    exact_noprep_by_image: Dict[str, float],
    n_folds: int = 5,
) -> Dict:
    """
    Outer GroupKFold(n_folds) evaluation.
    Each image appears in exactly one test fold.
    Returns pooled per-image results + final model trained on all data.
    """
    models = get_models()

    X_all = df[input_cols].values
    y_all = df[target_col].values
    groups = df[GROUP_COL].values

    outer_gkf = GroupKFold(n_splits=n_folds)

    # Pooled per-image results across all folds
    pooled_oracle = {}
    pooled_noprep = {}
    pooled_fixed = {}
    pooled_fixed_topk = {k: {} for k in [3, 5, 10, 15]}
    pooled_surr = {k: {} for k in [3, 5, 10, 15]}
    pooled_rand = {k: {} for k in [3, 5, 10, 15]}
    pooled_surr_params = {k: {} for k in [3, 15]}  # selected params for text inference

    fold_metrics = []
    selected_models_per_fold = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_gkf.split(X_all, y_all, groups)):
        print(f"  Fold {fold_idx + 1}/{n_folds} ...", end=" ", flush=True)

        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()

        X_tr = train_df[input_cols].values
        y_tr = train_df[target_col].values
        X_te = test_df[input_cols].values
        y_te = test_df[target_col].values

        # Nested selection: choose model family using only this outer fold's training images.
        best_model_name_fold = select_best_model(train_df, target_col, input_cols, verbose=False)
        selected_models_per_fold.append(best_model_name_fold)

        # Train selected model on this fold's training split.
        model = clone(models[best_model_name_fold])
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        test_eval = test_df.copy()
        test_eval['pred_cer'] = y_pred

        per_image_spearman = []
        for _img_name, group in test_eval.groupby(GROUP_COL):
            if len(group) <= 1:
                continue
            corr = stats.spearmanr(group[target_col], group['pred_cer']).correlation
            if not np.isnan(corr):
                per_image_spearman.append(float(corr))

        spearman_global = stats.spearmanr(y_te, y_pred).correlation

        fold_metrics.append({
            'mae': mean_absolute_error(y_te, y_pred),
            'r2': r2_score(y_te, y_pred),
            'spearman_global': spearman_global,
            # Backward-compatible alias for older report consumers.
            'spearman_r': spearman_global,
            'spearman_per_image_mean': (
                float(np.mean(per_image_spearman)) if per_image_spearman else float('nan')
            ),
            'spearman_per_image_median': (
                float(np.median(per_image_spearman)) if per_image_spearman else float('nan')
            ),
            'spearman_per_image_n': len(per_image_spearman),
            'pearson_r': stats.pearsonr(y_te, y_pred)[0],
            'selected_model': best_model_name_fold,
        })
        print(f"model={best_model_name_fold}  MAE={fold_metrics[-1]['mae']:.4f}  "
              f"GlobalSpearman={fold_metrics[-1]['spearman_global']:.4f}  "
              f"PerImageSpearman={fold_metrics[-1]['spearman_per_image_mean']:.4f}")

        # Sampled-pool oracle + exact no-preprocessing for each test image
        for img_name, group in test_eval.groupby(GROUP_COL):
            pooled_oracle[img_name] = group[target_col].min()
            if img_name not in exact_noprep_by_image:
                raise KeyError(f"Missing exact NoPre CER for image '{img_name}'")
            pooled_noprep[img_name] = float(exact_noprep_by_image[img_name])

        # Fixed best: best combo from THIS fold's training split
        combo_train_mean = train_df.groupby(PARAM_COLS)[target_col].mean()
        best_combo_vals = combo_train_mean.idxmin()
        best_combo_mask = combo_mask(test_df, best_combo_vals)
        missing_fixed_imgs = []
        for img_name, group in test_df.groupby(GROUP_COL):
            img_mask = (test_df[GROUP_COL] == img_name).values
            combined = best_combo_mask & img_mask
            if combined.sum() > 0:
                pooled_fixed[img_name] = test_df[combined].iloc[0][target_col]
            else:
                missing_fixed_imgs.append(img_name)
        if missing_fixed_imgs:
            raise ValueError(
                f"FixedBest combo {dict(zip(PARAM_COLS, best_combo_vals))} "
                f"not matched for {len(missing_fixed_imgs)} test images in "
                f"fold {fold_idx + 1}; first few: {missing_fixed_imgs[:3]}"
            )

        # FixedTopK: K globally best-average combos from THIS fold's training split.
        # This is a stronger static-shortlist baseline than Random K.
        fixed_topk_combos = {
            k: combo_train_mean.nsmallest(k).index
            for k in [3, 5, 10, 15]
        }
        for k, combos in fixed_topk_combos.items():
            topk_mask = np.zeros(len(test_df), dtype=bool)
            for combo_vals in combos:
                topk_mask |= combo_mask(test_df, combo_vals)
            matched = test_df[topk_mask]
            for img_name, group in matched.groupby(GROUP_COL):
                pooled_fixed_topk[k][img_name] = group[target_col].min()

            missing_fixed_topk_imgs = (
                set(test_df[GROUP_COL].unique()) - set(pooled_fixed_topk[k].keys())
            )
            if missing_fixed_topk_imgs:
                missing_list = sorted(missing_fixed_topk_imgs)[:3]
                raise ValueError(
                    f"FixedTopK K={k} did not match {len(missing_fixed_topk_imgs)} "
                    f"test images in fold {fold_idx + 1}; first few: {missing_list}"
                )

        # Surrogate filter + random for each K
        for k in [3, 5, 10, 15]:
            for img_name, group in test_eval.groupby(GROUP_COL):
                top_k = group.nsmallest(k, 'pred_cer')
                pooled_surr[k][img_name] = top_k[target_col].min()
                if k in pooled_surr_params:
                    best_row = top_k.loc[top_k[target_col].idxmin()]
                    pooled_surr_params[k][img_name] = best_row[PARAM_COLS].to_dict()

            for img_name, group in test_df.groupby(GROUP_COL):
                pooled_rand[k][img_name] = expected_minimum_without_replacement(
                    group[target_col].to_numpy(dtype=float), min(k, len(group))
                )

    # Aggregate fold metrics
    cv_metrics = {
        'mae_mean': np.mean([f['mae'] for f in fold_metrics]),
        'mae_std': np.std([f['mae'] for f in fold_metrics]),
        'r2_mean': np.mean([f['r2'] for f in fold_metrics]),
        'r2_std': np.std([f['r2'] for f in fold_metrics]),
        'spearman_global_mean': np.mean([f['spearman_global'] for f in fold_metrics]),
        'spearman_global_std': np.std([f['spearman_global'] for f in fold_metrics]),
        'spearman_per_image_mean': np.nanmean(
            [f['spearman_per_image_mean'] for f in fold_metrics]
        ),
        'spearman_per_image_std': np.nanstd(
            [f['spearman_per_image_mean'] for f in fold_metrics]
        ),
        'spearman_per_image_median_mean': np.nanmean(
            [f['spearman_per_image_median'] for f in fold_metrics]
        ),
        'spearman_per_image_n': int(np.sum(
            [f['spearman_per_image_n'] for f in fold_metrics]
        )),
        # Backward-compatible aliases: these are global row-level Spearman values.
        'spearman_mean': np.mean([f['spearman_global'] for f in fold_metrics]),
        'spearman_std': np.std([f['spearman_global'] for f in fold_metrics]),
        'pearson_mean': np.mean([f['pearson_r'] for f in fold_metrics]),
        'pearson_std': np.std([f['pearson_r'] for f in fold_metrics]),
    }

    baselines = {
        'oracle': np.mean(list(pooled_oracle.values())),
        'no_preprocessing': np.mean(list(pooled_noprep.values())),
        'fixed_best': np.mean(list(pooled_fixed.values())) if pooled_fixed else float('nan'),
    }

    # Two-sided Wilcoxon tests on pooled images across all outer folds.
    # Direction is reported separately through the paired mean differences.
    wilcoxon_results = {}
    imgs_sorted = sorted(pooled_surr[3].keys())
    per_page_rows = []
    for img_name in imgs_sorted:
        row = {
            'image_name': img_name,
            'oracle': float(pooled_oracle[img_name]),
            'no_preprocessing': float(pooled_noprep[img_name]),
            'fixed_best': float(pooled_fixed[img_name]),
        }
        for k in [3, 5, 10, 15]:
            row[f'surr_k{k}'] = float(pooled_surr[k][img_name])
            row[f'rand_k{k}'] = float(pooled_rand[k][img_name])
            row[f'fixed_topk_k{k}'] = float(pooled_fixed_topk[k][img_name])
            row[f'surr_minus_rand_k{k}'] = row[f'surr_k{k}'] - row[f'rand_k{k}']
            row[f'surr_minus_fixed_topk_k{k}'] = row[f'surr_k{k}'] - row[f'fixed_topk_k{k}']
        per_page_rows.append(row)

    for k in [3, 5, 10, 15]:
        surr_arr = np.array([pooled_surr[k][i] for i in imgs_sorted])
        rand_arr = np.array([pooled_rand[k][i] for i in imgs_sorted])
        fixed_topk_arr = np.array([pooled_fixed_topk[k][i] for i in imgs_sorted])
        diff = surr_arr - rand_arr
        if np.any(diff != 0):
            stat, p_val = wilcoxon(surr_arr, rand_arr, alternative='two-sided')
        else:
            stat = np.nan
            p_val = 1.0
        diff_fixed_topk = surr_arr - fixed_topk_arr
        if np.any(diff_fixed_topk != 0):
            stat_ftk, p_ftk = wilcoxon(surr_arr, fixed_topk_arr, alternative='two-sided')
        else:
            stat_ftk = np.nan
            p_ftk = 1.0
        wilcoxon_results[k] = {
            'alternative': 'two-sided',
            'statistic_vs_random': float(stat) if np.isfinite(stat) else float('nan'),
            'p_value': float(p_val),
            'surr_mean': float(np.mean(surr_arr)),
            'rand_mean': float(np.mean(rand_arr)),
            'diff': float(np.mean(diff)),
            'diff_ci95': bootstrap_mean_ci(diff, seed=4200 + k),
            'fixed_topk_mean': float(np.mean(fixed_topk_arr)),
            'surr_minus_fixed_topk': float(np.mean(diff_fixed_topk)),
            'surr_minus_fixed_topk_ci95': bootstrap_mean_ci(diff_fixed_topk, seed=5200 + k),
            'statistic_vs_fixed_topk': float(stat_ftk) if np.isfinite(stat_ftk) else float('nan'),
            'p_vs_fixed_topk': float(p_ftk),
            'n_images': len(imgs_sorted),
        }

    # Final deployable model: select family on all available images, then refit on all data.
    final_model_name = select_best_model(df, target_col, input_cols, verbose=False)
    print(f"  Training final model on all data ({final_model_name}) ...", end=" ", flush=True)
    final_model = clone(models[final_model_name])
    final_model.fit(X_all, y_all)
    print("done")

    feature_importance = extract_feature_importance(final_model, input_cols)

    return {
        'best_model': final_model_name,
        'selected_models_per_fold': selected_models_per_fold,
        'model_selection_summary': summarize_model_selection(selected_models_per_fold),
        'fold_metrics': fold_metrics,
        'cv_metrics': cv_metrics,
        'baselines': baselines,
        'fixed_topk_results': {
            k: np.mean(list(pooled_fixed_topk[k].values())) for k in [3, 5, 10, 15]
        },
        'filter_results': {k: np.mean(list(pooled_surr[k].values())) for k in [3, 5, 10, 15]},
        'random_results': {k: np.mean(list(pooled_rand[k].values())) for k in [3, 5, 10, 15]},
        'wilcoxon': wilcoxon_results,
        'per_page_results': pd.DataFrame(per_page_rows),
        'feature_importance': feature_importance,
        'model': final_model,
        'surr_params': pooled_surr_params,  # {3: {img_name: {param: val}}, 15: {...}}
    }


# ============================================================================
# Report
# ============================================================================

def write_report(all_results: Dict, output_dir: str, input_cols: List[str], n_folds: int):
    report_path = os.path.join(output_dir, 'report.txt')
    engines = list(all_results.keys())
    pooled_n_images = None
    if engines:
        sample_engine = all_results[engines[0]]
        if sample_engine.get('wilcoxon'):
            first_k = sorted(sample_engine['wilcoxon'].keys(), key=int)[0]
            pooled_n_images = sample_engine['wilcoxon'][first_k].get('n_images')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-ENGINE SURROGATE — K-FOLD CV REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation: {n_folds}-fold GroupKFold (no image leakage)\n")
        if pooled_n_images is not None:
            f.write(f"Wilcoxon: pooled across all folds (all {pooled_n_images} images)\n")
        else:
            f.write("Wilcoxon: pooled across all folds\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Engines: {engines}\n")
        f.write(f"Input features: {len(input_cols)}\n\n")

        # 1. Model selection
        f.write("1. FINAL MODEL FAMILY / OUTER-FOLD SELECTION\n")
        f.write("-" * 50 + "\n")
        for engine in engines:
            final_model = all_results[engine]['best_model']
            summary = all_results[engine].get('model_selection_summary', final_model)
            f.write(f"  {engine:12s}: final={final_model} | folds={summary}\n")
        f.write("\n")

        # 2. CV metrics (mean ± std across folds)
        f.write("2. CROSS-VALIDATION METRICS (mean ± std across folds)\n")
        f.write("   Spearman-global is computed over all held-out rows in a fold.\n")
        f.write("   Spearman-per-image is the mean within-page candidate-rank correlation.\n")
        f.write(f"  {'Engine':<12s} {'MAE':>14s} {'R²':>14s} "
                f"{'SpearGlob':>14s} {'SpearImg':>14s} {'Pearson':>14s}\n")
        f.write("  " + "-" * 76 + "\n")
        for engine in engines:
            m = all_results[engine]['cv_metrics']
            f.write(f"  {engine:<12s} "
                    f"{m['mae_mean']:>6.4f}±{m['mae_std']:.4f} "
                    f"{m['r2_mean']:>6.4f}±{m['r2_std']:.4f} "
                    f"{m['spearman_global_mean']:>6.4f}±{m['spearman_global_std']:.4f} "
                    f"{m['spearman_per_image_mean']:>6.4f}±{m['spearman_per_image_std']:.4f} "
                    f"{m['pearson_mean']:>6.4f}±{m['pearson_std']:.4f}\n")
        f.write("\n")

        # 2b. Per-fold detail
        f.write("2b. PER-FOLD METRICS\n")
        for engine in engines:
            f.write(f"  {engine}:\n")
            f.write(f"    {'Fold':<6s} {'MAE':>8s} {'R²':>8s} "
                    f"{'SpearGlob':>10s} {'SpearImg':>10s}  {'Model':<12s}\n")
            f.write("    " + "-" * 65 + "\n")
            for i, fm in enumerate(all_results[engine]['fold_metrics']):
                f.write(f"    {i+1:<6d} {fm['mae']:>8.4f} {fm['r2']:>8.4f} "
                        f"{fm['spearman_global']:>10.4f} "
                        f"{fm['spearman_per_image_mean']:>10.4f}  "
                        f"{fm['selected_model']}\n")
        f.write("\n")

        # 3. Baseline CERs (pooled across all outer-fold test images)
        baseline_title = "3. BASELINE CERs"
        if pooled_n_images is not None:
            baseline_title += f" (pooled, all {pooled_n_images} images)"
        f.write(baseline_title + "\n")
        f.write(f"  {'Engine':<12s} {'SampOracle':>10s} {'FixedBest':>10s} {'NoPre':>8s}\n")
        f.write("  " + "-" * 42 + "\n")
        for engine in engines:
            b = all_results[engine]['baselines']
            f.write(f"  {engine:<12s} {b['oracle']:>10.4f} "
                    f"{b['fixed_best']:>10.4f} {b['no_preprocessing']:>8.4f}\n")
        f.write("\n")

        # 3b. Surrogate filter vs Fixed Best
        f.write("3b. SURROGATE FILTER vs FIXED BEST-AVERAGE\n")
        f.write("  (Fixed best: best combo from each fold's training set)\n\n")
        for k in [3, 5, 10, 15]:
            f.write(f"  K={k}:\n")
            f.write(f"    {'Engine':<12s} {'FixedBest':>10s} {'Surr':>8s} {'Diff':>8s}\n")
            f.write("    " + "-" * 42 + "\n")
            for engine in engines:
                r = all_results[engine]
                surr = r['filter_results'][k]
                fixed = r['baselines']['fixed_best']
                f.write(f"    {engine:<12s} {fixed:>10.4f} {surr:>8.4f} {surr - fixed:>+8.4f}\n")
            f.write("\n")

        # 3c. Surrogate filter vs FixedTopK
        f.write("3c. SURROGATE FILTER vs FIXEDTOPK STATIC SHORTLIST\n")
        f.write("  (FixedTopK: K best-average combos from each fold's training set)\n\n")
        for k in [3, 5, 10, 15]:
            f.write(f"  K={k}:\n")
            f.write(f"    {'Engine':<12s} {'FixedTopK':>10s} {'Surr':>8s} "
                    f"{'Diff':>8s} {'p-value':>10s} {'Sig':>5s}\n")
            f.write("    " + "-" * 58 + "\n")
            for engine in engines:
                r = all_results[engine]
                surr = r['filter_results'][k]
                fixed_topk = r['fixed_topk_results'][k]
                w = r['wilcoxon'][k]
                p = w.get('p_vs_fixed_topk', 1.0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                f.write(f"    {engine:<12s} {fixed_topk:>10.4f} {surr:>8.4f} "
                        f"{surr - fixed_topk:>+8.4f} {p:>10.6f} {sig:>5s}\n")
            f.write("\n")

        # 4. Surrogate filter vs Random — Wilcoxon on pooled images
        f.write("4. SURROGATE FILTER vs RANDOM SHORTLISTS\n")
        n_test = all_results[engines[0]]['wilcoxon'][3]['n_images']
        f.write(f"  (Wilcoxon on {n_test} pooled images, lower CER is better)\n\n")
        for k in [3, 5, 10, 15]:
            f.write(f"  K={k}:\n")
            f.write(f"    {'Engine':<12s} {'Surr':>8s} {'Rand':>8s} {'Diff':>8s} "
                    f"{'p-value':>10s} {'Sig':>5s}\n")
            f.write("    " + "-" * 53 + "\n")
            for engine in engines:
                r = all_results[engine]
                surr = r['filter_results'][k]
                rand = r['random_results'][k]
                w = r['wilcoxon'][k]
                p = w['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                f.write(f"    {engine:<12s} {surr:>8.4f} {rand:>8.4f} "
                        f"{surr - rand:>+8.4f} {p:>10.6f} {sig:>5s}\n")
            f.write("\n")

        # 5. Feature importance (from final model trained on all available images)
        primary_engine = engines[0]
        fi = all_results[primary_engine].get('feature_importance', {})
        if fi:
            f.write("5. FEATURE IMPORTANCE (from final model on all available images)\n")
            f.write(f"   Primary engine: {primary_engine}\n")
            f.write("-" * 50 + "\n")
            for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(imp * 50)
                f.write(f"  {feat:<35s} {imp:>6.1%}  {bar}\n")
            f.write("\n")
            f.write("  All engines (top 5 each):\n")
            for engine in engines:
                fi_e = all_results[engine].get('feature_importance', {})
                top5 = sorted(fi_e.items(), key=lambda x: x[1], reverse=True)[:5]
                f.write(f"  {engine}: " + ", ".join(f"{k}={v:.1%}" for k, v in top5) + "\n")
            f.write("\n")

        # 6. Summary
        f.write("6. CROSS-ENGINE SUMMARY\n")
        f.write("-" * 50 + "\n")
        for engine in engines:
            r = all_results[engine]
            n_sig = sum(1 for k in [3, 5, 10, 15]
                        if r['wilcoxon'][k]['p_value'] < 0.05)
            m = r['cv_metrics']
            f.write(f"  {engine:12s}: R²={m['r2_mean']:.3f}±{m['r2_std']:.3f}, "
                    f"GlobalSpearman={m['spearman_global_mean']:.3f}±{m['spearman_global_std']:.3f}, "
                    f"PerImageSpearman={m['spearman_per_image_mean']:.3f}±{m['spearman_per_image_std']:.3f}, "
                    f"Significant K: {n_sig}/4\n")
        f.write("\n")

        # Conclusion
        all_sig = all(
            all(r['wilcoxon'][k]['p_value'] < 0.05 for k in [3, 5, 10, 15])
            for r in all_results.values()
        )
        any_sig = any(
            any(r['wilcoxon'][k]['p_value'] < 0.05 for k in [3, 5, 10, 15])
            for r in all_results.values()
        )
        f.write("CONCLUSION\n")
        f.write("-" * 50 + "\n")
        if all_sig:
            f.write(f"  Surrogate filter significantly beats random shortlists\n")
            f.write(f"  at ALL K values for ALL engines ({n_test} pooled images, {n_folds}-fold CV).\n")
            f.write(f"  → Strong evidence independent of train/test split.\n")
        elif any_sig:
            f.write("  Surrogate filter shows significance for SOME engines/K values.\n")
            f.write("  → Partial evidence for generalization.\n")
        else:
            f.write("  No significant improvement found.\n")

    print(f"  Report: {report_path}")
    return report_path


# ============================================================================
# Plots
# ============================================================================

def plot_comparison(all_results: Dict, output_dir: str):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    engines = list(all_results.keys())
    colors = {'vietocr': '#2196F3', 'tesseract': '#FF9800', 'easyocr': '#4CAF50'}
    markers = {'vietocr': 'o', 'tesseract': 's', 'easyocr': '^'}

    # Figure 1: Surrogate vs Random across K values
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in engines:
        r = all_results[engine]
        ks = [3, 5, 10, 15]
        surr = [r['filter_results'][k] * 100 for k in ks]
        rand = [r['random_results'][k] * 100 for k in ks]
        fixed_topk = [r['fixed_topk_results'][k] * 100 for k in ks]
        c = colors.get(engine, '#999')
        m = markers.get(engine, 'o')
        ax.plot(ks, surr, f'{m}-', color=c, lw=2, ms=8, label=f'{engine} (surrogate)', zorder=3)
        ax.plot(ks, rand, f'{m}--', color=c, lw=1.5, ms=6, alpha=0.5,
                label=f'{engine} (random)')
        ax.plot(ks, fixed_topk, f'{m}:', color=c, lw=1.8, ms=6, alpha=0.8,
                label=f'{engine} (FixedTopK)')
    ax.set_xlabel('Number of OCR Calls (K)', fontsize=12)
    ax.set_ylabel('Mean CER (%)', fontsize=12)
    ax.set_title(f'Surrogate Filter vs Random — {len(engines)}-Engine K-Fold CV', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([3, 5, 10, 15])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pareto_multiengine.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: CV metrics bar chart (mean ± std)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_keys = [
        ('r2_mean', 'r2_std'),
        ('mae_mean', 'mae_std'),
        ('spearman_global_mean', 'spearman_global_std'),
        ('spearman_per_image_mean', 'spearman_per_image_std'),
    ]
    titles = [
        'R² (higher=better)',
        'MAE (lower=better)',
        'Global Spearman (higher=better)',
        'Per-Image Spearman (higher=better)',
    ]
    for ax, (mu_k, std_k), title in zip(axes, metric_keys, titles):
        vals = [all_results[e]['cv_metrics'][mu_k] for e in engines]
        errs = [all_results[e]['cv_metrics'][std_k] for e in engines]
        bars = ax.bar(engines, vals, yerr=errs, capsize=5,
                      color=[colors.get(e, '#999') for e in engines], alpha=0.8)
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=9)
    plt.suptitle('K-Fold CV Metrics (mean ± std)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Significance heatmap
    ks = [3, 5, 10, 15]
    pvals = np.array([
        [all_results[e]['wilcoxon'][k]['p_value'] for k in ks]
        for e in engines
    ])
    log_p = np.clip(-np.log10(pvals + 1e-10), 0, 4)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(log_p, cmap='YlOrRd', aspect='auto', vmin=0, vmax=4)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'K={k}' for k in ks])
    ax.set_yticks(range(len(engines)))
    ax.set_yticklabels(engines)
    for i in range(len(engines)):
        for j in range(len(ks)):
            p = pvals[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(j, i, f'{sig}\n{p:.4f}', ha='center', va='center', fontsize=9)
    pooled_n_images = None
    if engines and all_results[engines[0]].get('wilcoxon'):
        pooled_n_images = all_results[engines[0]]['wilcoxon'][ks[0]].get('n_images')
    if pooled_n_images is None:
        ax.set_title('Statistical Significance (Wilcoxon, pooled images)', fontsize=12)
    else:
        ax.set_title(
            f'Statistical Significance (Wilcoxon, {pooled_n_images} pooled images)',
            fontsize=12,
        )
    plt.colorbar(im, ax=ax, label='-log10(p-value)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'significance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures: {fig_dir}/")


# ============================================================================
# Main
# ============================================================================

def main(data_path: str,
         output_dir: str,
         engines: List[str] = None,
         n_folds: int = 5,
         nopre_path: Optional[str] = None):
    print("=" * 70)
    print(f"MULTI-ENGINE SURROGATE — {n_folds}-FOLD GROUP CV")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}, Images: {df[GROUP_COL].nunique()}")

    available_cer_cols = [c for c in df.columns if c.startswith('cer_')]
    if available_cer_cols:
        available_engines = [c.replace('cer_', '') for c in available_cer_cols]
    elif 'cer' in df.columns:
        available_engines = ['vietocr']
        df['cer_vietocr'] = df['cer']
    else:
        print("ERROR: No CER columns found.")
        return

    engine_list = [e for e in (engines or available_engines) if e in available_engines]
    print(f"  Engines: {engine_list}")

    nopre_csv = resolve_exact_nopre_csv(data_path, nopre_path)
    exact_nopre_maps = load_exact_nopre_maps(nopre_csv, engine_list)
    missing_images = sorted(set(df[GROUP_COL].unique()) - set(next(iter(exact_nopre_maps.values())).keys()))
    if missing_images:
        raise KeyError(f"Exact NoPre CSV is missing {len(missing_images)} images; first few: {missing_images[:5]}")
    print(f"  Exact NoPre: {nopre_csv}")

    input_cols = prepare_data(df)
    print(f"  Input features: {len(input_cols)}")

    all_results = {}
    for engine in engine_list:
        target_col = f'cer_{engine}'
        print(f"\n{'=' * 70}")
        print(f"ENGINE: {engine}")
        print(f"{'=' * 70}")

        if target_col not in df.columns:
            print(f"  SKIP: {target_col} not in data")
            continue

        print(f"  CER mean={df[target_col].mean():.4f}  "
              f"range=[{df[target_col].min():.4f}, {df[target_col].max():.4f}]")

        print(f"  Running {n_folds}-fold nested GroupKFold CV:")
        results = train_one_engine_kfold(
            df,
            target_col,
            input_cols,
            exact_nopre_maps[engine],
            n_folds,
        )

        m = results['cv_metrics']
        n_eval_images = results['wilcoxon'][3]['n_images']
        print(f"\n  CV summary: MAE={m['mae_mean']:.4f}±{m['mae_std']:.4f}  "
              f"GlobalSpearman={m['spearman_global_mean']:.4f}±{m['spearman_global_std']:.4f}  "
              f"PerImageSpearman={m['spearman_per_image_mean']:.4f}±{m['spearman_per_image_std']:.4f}")
        print(f"  Final model family: {results['best_model']}")
        print(f"  Outer-fold selections: {results['model_selection_summary']}")
        print(f"  Surrogate filter (pooled {n_eval_images} images):")
        for k in [3, 5, 10, 15]:
            w = results['wilcoxon'][k]
            sig = "***" if w['p_value'] < 0.001 else "**" if w['p_value'] < 0.01 \
                else "*" if w['p_value'] < 0.05 else "ns"
            sig_ftk = "***" if w['p_vs_fixed_topk'] < 0.001 else "**" if w['p_vs_fixed_topk'] < 0.01 \
                else "*" if w['p_vs_fixed_topk'] < 0.05 else "ns"
            print(f"    K={k:2d}: surr={results['filter_results'][k]:.4f} "
                  f"vs rand={results['random_results'][k]:.4f} "
                  f"p={w['p_value']:.6f} ({sig}); "
                  f"vs FixedTopK={results['fixed_topk_results'][k]:.4f} "
                  f"p={w['p_vs_fixed_topk']:.6f} ({sig_ftk})")

        # Save final model
        final_model_name = results['best_model']
        model_path = os.path.join(output_dir, f'model_{engine}_{final_model_name}.pkl')
        with open(model_path, 'wb') as fp:
            pickle.dump(results['model'], fp)

        per_page_csv = os.path.join(output_dir, f'per_page_results_{engine}.csv')
        results['per_page_results'].to_csv(per_page_csv, index=False)
        print(f"  Saved per-page results: {per_page_csv}")

        all_results[engine] = results

    # Save selected params CSV for text inference (VietOCR only)
    if 'vietocr' in all_results and all_results['vietocr']['surr_params'][3]:
        surr_params = all_results['vietocr']['surr_params']
        rows = []
        for img_name in sorted(surr_params[3].keys()):
            if img_name not in surr_params[15]:
                continue
            row = {'image_name': img_name}
            for k in [3, 15]:
                for param, val in surr_params[k][img_name].items():
                    row[f'k{k}_{param}'] = val
            rows.append(row)
        params_df = pd.DataFrame(rows)
        params_csv = os.path.join(output_dir, 'surr_selected_params.csv')
        params_df.to_csv(params_csv, index=False)
        print(f"  Saved selected params: {params_csv} ({len(rows)} images)")

    print(f"\n{'=' * 70}")
    print("GENERATING REPORT & PLOTS")
    write_report(all_results, output_dir, input_cols, n_folds)
    plot_comparison(all_results, output_dir)

    # Save metrics JSON
    metrics_json = {}
    for engine, r in all_results.items():
        metrics_json[engine] = {
            'best_model': r['best_model'],
            'selected_models_per_fold': r.get('selected_models_per_fold', []),
            'model_selection_summary': r.get('model_selection_summary', ''),
            'cv_metrics': r['cv_metrics'],
            'baselines': r['baselines'],
            'fixed_topk_results': {str(k): v for k, v in r['fixed_topk_results'].items()},
            'filter_results': {str(k): v for k, v in r['filter_results'].items()},
            'random_results': {str(k): v for k, v in r['random_results'].items()},
            'wilcoxon': {str(k): v for k, v in r['wilcoxon'].items()},
        }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        json.dump(metrics_json, fp, indent=2, default=str)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-fold CV surrogate evaluation (GroupKFold, no image leakage)")
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--engines', nargs='+', default=None)
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of outer CV folds (default: 5)')
    parser.add_argument('--nopre-path', type=str, required=True,
                        help='Exact NoPre per-image CSV generated for this run')
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = str(
            RELEASE_ROOT / f'results/recomputed/surrogate_multiengine_kfold_{timestamp}')

    main(args.data_path, args.output_dir, args.engines, args.n_folds, args.nopre_path)
