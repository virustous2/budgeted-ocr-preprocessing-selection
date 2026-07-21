#!/usr/bin/env python3
"""
Train surrogate models for multiple OCR engines and compare.

Reads multi-engine CSV (from collect_multiengine_data.py) with columns:
  cer_vietocr, cer_tesseract, cer_easyocr

Trains a separate surrogate for each engine, then compares:
  - R², MAE, Spearman per engine
  - Surrogate filter vs Random at each K per engine
  - Statistical significance per engine

This default-parameter script is intended for sensitivity checks. The main
paper metrics should come from train_surrogate_multiengine_kfold.py.

Usage with a locally collected candidate-response table and exact NoPre CSV:
    python code/main/train_surrogate_multiengine_default.py \
        --data-path /path/to/preprocessing_data.csv \
        --nopre-path /path/to/default_cer_multiengine.csv

    # Optionally restrict the rerun to selected recognizers:
    python code/main/train_surrogate_multiengine_default.py \
        --data-path /path/to/preprocessing_data.csv \
        --nopre-path /path/to/default_cer_multiengine.csv \
        --engines vietocr tesseract
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]

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


# ============================================================================
# Models
# ============================================================================

def get_models() -> Dict:
    """All models use library defaults. Only random_state and n_jobs are set
    for reproducibility and parallelism — these do not affect model capacity.

    XGBoost defaults: n_estimators=100, max_depth=6, learning_rate=0.3,
                      subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
                      reg_alpha=0, reg_lambda=1, objective=reg:squarederror
    LightGBM defaults: n_estimators=100, max_depth=-1, learning_rate=0.1,
                       subsample=1.0, colsample_bytree=1.0, objective=regression
    RandomForest defaults: n_estimators=100, max_depth=None, min_samples_leaf=1
    """
    return {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge()),                          # alpha=1.0
        ]),
        'RandomForest': RandomForestRegressor(
            n_jobs=-1, random_state=42,                  # n_estimators=100, max_depth=None
        ),
        'XGBoost': xgb.XGBRegressor(
            n_jobs=-1, random_state=42, verbosity=0,     # n_estimators=100, lr=0.3
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_jobs=-1, random_state=42, verbose=-1,      # n_estimators=100, lr=0.1
        ),
        'Stacking': StackingRegressor(
            estimators=[
                ('rf',   RandomForestRegressor(n_jobs=-1, random_state=42)),
                ('xgb',  xgb.XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)),
                ('lgbm', lgb.LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)),
            ],
            final_estimator=Ridge(),
            cv=5, n_jobs=-1,
        ),
    }


# ============================================================================
# Prepare Data
# ============================================================================

def prepare_data(df: pd.DataFrame, target_col: str, use_interactions: bool = True):
    """Detect features, add interactions, return INPUT_COLS list."""
    feature_cols = FEATURE_COLS_BASE.copy()
    available_new = [c for c in FEATURE_COLS_NEW if c in df.columns]
    feature_cols.extend(available_new)

    input_cols = feature_cols + PARAM_COLS

    if use_interactions:
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

    return input_cols, feature_cols


def resolve_exact_nopre_csv(data_path: str, explicit_path: Optional[str] = None) -> Path:
    if not explicit_path:
        raise ValueError(
            "This public-release script requires --nopre-path because the original "
            "NoPre OCR outputs and source documents are not distributed."
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


def select_best_model(
    df: pd.DataFrame,
    target_col: str,
    input_cols: List[str],
    verbose: bool = True,
) -> str:
    """Run grouped inner CV on the provided training split to pick model family."""
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


def per_image_spearman_summary(
    eval_df: pd.DataFrame,
    target_col: str,
    pred_col: str = 'pred_cer',
) -> Dict[str, float]:
    """Summarize within-page candidate-rank correlation."""
    correlations = []
    for _img_name, group in eval_df.groupby(GROUP_COL):
        if len(group) <= 1:
            continue
        corr = stats.spearmanr(group[target_col], group[pred_col]).correlation
        if not np.isnan(corr):
            correlations.append(float(corr))

    return {
        'spearman_per_image_mean': (
            float(np.mean(correlations)) if correlations else float('nan')
        ),
        'spearman_per_image_median': (
            float(np.median(correlations)) if correlations else float('nan')
        ),
        'spearman_per_image_n': len(correlations),
    }


# ============================================================================
# Train & Evaluate one engine
# ============================================================================

def train_one_engine(
    df: pd.DataFrame,
    target_col: str,
    input_cols: List[str],
    feature_cols: List[str],
    exact_noprep_by_image: Dict[str, float],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Train surrogate for one engine, return full results dict."""

    # Split by image
    images = df[GROUP_COL].unique()
    train_imgs, test_imgs = train_test_split(images, test_size=test_ratio, random_state=seed)
    train_df = df[df[GROUP_COL].isin(train_imgs)].copy()
    test_df = df[df[GROUP_COL].isin(test_imgs)].copy()

    # Model selection on train images only.
    models = get_models()
    best_name = select_best_model(train_df, target_col, input_cols, verbose=False)

    # CV summary for the selected family on the training split.
    X = train_df[input_cols].values
    y = train_df[target_col].values
    groups = train_df[GROUP_COL].values
    gkf = GroupKFold(n_splits=5)
    selected_fold_metrics = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        m = clone(models[best_name])
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])
        val_eval = train_df.iloc[val_idx].copy()
        val_eval['pred_cer'] = y_pred
        spearman_global = stats.spearmanr(y[val_idx], y_pred).correlation
        per_image = per_image_spearman_summary(val_eval, target_col)
        selected_fold_metrics.append({
            'mae': mean_absolute_error(y[val_idx], y_pred),
            'r2': r2_score(y[val_idx], y_pred),
            'spearman_global': spearman_global,
            # Backward-compatible alias for older report consumers.
            'spearman_r': spearman_global,
            **per_image,
        })

    cv_results = {
        f'{k}_mean': np.mean([f[k] for f in selected_fold_metrics])
        for k in selected_fold_metrics[0]
    }
    cv_results.update({
        f'{k}_std': np.std([f[k] for f in selected_fold_metrics])
        for k in selected_fold_metrics[0]
    })

    # Train on full train set
    best_model = clone(models[best_name])
    X_train = train_df[input_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[input_cols].values
    y_test = test_df[target_col].values

    best_model.fit(X_train, y_train)
    y_pred_train = best_model.predict(X_train)
    y_pred = best_model.predict(X_test)
    test_eval = test_df.copy()
    test_eval['pred_cer'] = y_pred
    test_spearman_global = stats.spearmanr(y_test, y_pred).correlation
    test_per_image = per_image_spearman_summary(test_eval, target_col)

    train_metrics = {
        'mae': mean_absolute_error(y_train, y_pred_train),
        'r2': r2_score(y_train, y_pred_train),
    }
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'spearman_global': test_spearman_global,
        # Backward-compatible alias for older report consumers.
        'spearman_r': test_spearman_global,
        **test_per_image,
        'pearson_r': stats.pearsonr(y_test, y_pred)[0],
    }

    # Surrogate filter & random baselines
    baselines = {}

    # Oracle
    oracle_cers = []
    for _, group in test_eval.groupby(GROUP_COL):
        oracle_cers.append(group[target_col].min())
    baselines['oracle'] = np.mean(oracle_cers)

    noprep_cers = []
    for _, group in test_df.groupby(GROUP_COL):
        img_name = group[GROUP_COL].iloc[0]
        if img_name not in exact_noprep_by_image:
            raise KeyError(f"Missing exact NoPre CER for image '{img_name}'")
        noprep_cers.append(float(exact_noprep_by_image[img_name]))
    baselines['no_preprocessing'] = np.mean(noprep_cers)

    # Fixed best-average: best single combo from training, applied uniformly to all test images
    combo_train_mean = train_df.groupby(PARAM_COLS)[target_col].mean()
    best_combo_vals = combo_train_mean.idxmin()  # tuple of param values
    best_combo_mask = np.ones(len(test_df), dtype=bool)
    for col, val in zip(PARAM_COLS, best_combo_vals):
        best_combo_mask &= np.isclose(test_df[col].values, val)
    fixed_cers = []
    missing_fixed_imgs = []
    for img_name, group in test_df.groupby(GROUP_COL):
        img_mask = (test_df[GROUP_COL] == img_name).values
        combined = best_combo_mask & img_mask
        if combined.sum() > 0:
            fixed_cers.append(test_df[combined].iloc[0][target_col])
        else:
            missing_fixed_imgs.append(img_name)
    if missing_fixed_imgs:
        raise ValueError(
            f"FixedBest combo {dict(zip(PARAM_COLS, best_combo_vals))} "
            f"not matched for {len(missing_fixed_imgs)} test images; "
            f"first few: {missing_fixed_imgs[:3]}"
        )
    baselines['fixed_best'] = float(np.mean(fixed_cers))

    # Surrogate filter + Random at each K
    filter_results = {}
    random_results = {}
    wilcoxon_results = {}

    for k in [3, 5, 10, 15]:
        # Surrogate filter
        surr_per_image = {}
        for img_name, group in test_eval.groupby(GROUP_COL):
            top_k = group.nsmallest(k, 'pred_cer')
            surr_per_image[img_name] = top_k[target_col].min()
        filter_results[k] = surr_per_image

        # Random search (avg 20 trials)
        rand_per_image = {}
        for img_name, group in test_df.groupby(GROUP_COL):
            trial_cers = []
            for s in range(20):
                sampled = group.sample(min(k, len(group)), random_state=s)
                trial_cers.append(sampled[target_col].min())
            rand_per_image[img_name] = np.mean(trial_cers)
        random_results[k] = rand_per_image

        # Wilcoxon
        imgs = sorted(surr_per_image.keys())
        surr_arr = np.array([surr_per_image[i] for i in imgs])
        rand_arr = np.array([rand_per_image[i] for i in imgs])
        diff = surr_arr - rand_arr
        if np.any(diff != 0):
            stat, p_val = wilcoxon(surr_arr, rand_arr, alternative='less')
            wilcoxon_results[k] = {
                'p_value': float(p_val),
                'surr_mean': float(np.mean(surr_arr)),
                'rand_mean': float(np.mean(rand_arr)),
                'diff': float(np.mean(diff)),
                'n_images': len(imgs),
            }
        else:
            wilcoxon_results[k] = {'p_value': 1.0, 'surr_mean': float(np.mean(surr_arr)),
                                   'rand_mean': float(np.mean(rand_arr)), 'diff': 0.0,
                                   'n_images': len(imgs)}

    return {
        'best_model': best_name,
        'cv_results': cv_results,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'baselines': baselines,
        'filter_results': {k: np.mean(list(v.values())) for k, v in filter_results.items()},
        'random_results': {k: np.mean(list(v.values())) for k, v in random_results.items()},
        'wilcoxon': wilcoxon_results,
        'model': best_model,
        'train_imgs': list(train_imgs),
        'test_imgs': list(test_imgs),
    }


# ============================================================================
# Comparison Report
# ============================================================================

def write_report(all_results: Dict, output_dir: str, input_cols: List[str]):
    """Write comparison report across engines."""
    report_path = os.path.join(output_dir, 'report.txt')
    engines = list(all_results.keys())

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-ENGINE SURROGATE COMPARISON REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Engines: {engines}\n")
        f.write(f"Input features: {len(input_cols)}\n\n")

        # 1. Model selection per engine
        f.write("1. BEST MODEL PER ENGINE\n")
        f.write("-" * 50 + "\n")
        for engine in engines:
            r = all_results[engine]
            f.write(f"  {engine:12s}: {r['best_model']}\n")
        f.write("\n")

        # 2. Test metrics comparison
        f.write("2. TEST SET METRICS\n")
        f.write("   Spearman-global is computed over all held-out rows.\n")
        f.write("   Spearman-per-image is the mean within-page candidate-rank correlation.\n")
        f.write(f"  {'Engine':<12s} {'MAE':>8s} {'R²':>8s} "
                f"{'SpearGlob':>10s} {'SpearImg':>10s} {'Pearson':>10s}\n")
        f.write("  " + "-" * 62 + "\n")
        for engine in engines:
            m = all_results[engine]['test_metrics']
            f.write(f"  {engine:<12s} {m['mae']:>8.4f} {m['r2']:>8.4f} "
                    f"{m['spearman_global']:>10.4f} "
                    f"{m['spearman_per_image_mean']:>10.4f} "
                    f"{m['pearson_r']:>10.4f}\n")
        f.write("\n")

        # 2b. Train vs Test — overfitting check
        f.write("2b. TRAIN vs TEST MAE (Overfitting Check)\n")
        f.write(f"  {'Engine':<12s} {'Train MAE':>10s} {'Test MAE':>10s} {'Ratio':>8s} {'Status':>10s}\n")
        f.write("  " + "-" * 52 + "\n")
        for engine in engines:
            r = all_results[engine]
            tr_mae = r['train_metrics']['mae']
            te_mae = r['test_metrics']['mae']
            ratio = te_mae / tr_mae if tr_mae > 0 else 0
            status = "OK" if ratio < 1.5 else "OVERFIT?"
            f.write(f"  {engine:<12s} {tr_mae:>10.4f} {te_mae:>10.4f} {ratio:>8.2f}x {status:>10s}\n")
        f.write("\n")

        # 3. Baseline CERs
        f.write("3. BASELINE CERs\n")
        f.write(f"  {'Engine':<12s} {'Oracle':>8s} {'FixedBest':>10s} {'NoPre':>8s}\n")
        f.write("  " + "-" * 42 + "\n")
        for engine in engines:
            b = all_results[engine]['baselines']
            fixed = b.get('fixed_best', float('nan'))
            f.write(f"  {engine:<12s} {b['oracle']:>8.4f} {fixed:>10.4f} {b['no_preprocessing']:>8.4f}\n")
        f.write("\n")

        # 3b. Surrogate filter vs Fixed Best-Average
        f.write("3b. SURROGATE FILTER vs FIXED BEST-AVERAGE\n")
        f.write("  (Fixed best-average = best single combo from training, applied uniformly)\n\n")
        for k in [3, 5, 10, 15]:
            f.write(f"  K={k}:\n")
            f.write(f"    {'Engine':<12s} {'FixedBest':>10s} {'Surr':>8s} {'Diff':>8s}\n")
            f.write("    " + "-" * 42 + "\n")
            for engine in engines:
                r = all_results[engine]
                surr = r['filter_results'][k]
                fixed = r['baselines'].get('fixed_best', float('nan'))
                diff = surr - fixed
                f.write(f"    {engine:<12s} {fixed:>10.4f} {surr:>8.4f} {diff:>+8.4f}\n")
            f.write("\n")

        # 4. Surrogate filter vs Random per K per engine
        f.write("4. SURROGATE FILTER vs RANDOM SEARCH\n")
        f.write("  (Mean CER, lower is better)\n\n")
        for k in [3, 5, 10, 15]:
            f.write(f"  K={k}:\n")
            f.write(f"    {'Engine':<12s} {'Surr':>8s} {'Rand':>8s} {'Diff':>8s} {'p-value':>10s} {'Sig':>5s}\n")
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

        # 5. Cross-engine summary
        f.write("5. CROSS-ENGINE SUMMARY\n")
        f.write("-" * 50 + "\n")
        for engine in engines:
            r = all_results[engine]
            n_sig = sum(1 for k in [3, 5, 10, 15]
                        if r['wilcoxon'][k]['p_value'] < 0.05)
            f.write(f"  {engine:12s}: R²={r['test_metrics']['r2']:.3f}, "
                    f"GlobalSpearman={r['test_metrics']['spearman_global']:.3f}, "
                    f"PerImageSpearman={r['test_metrics']['spearman_per_image_mean']:.3f}, "
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
            f.write("  Surrogate filter significantly beats random search\n")
            f.write("  at ALL K values for ALL engines.\n")
            f.write("  → Strong evidence for cross-engine generalization.\n")
        elif any_sig:
            f.write("  Surrogate filter shows significance for SOME engines/K values.\n")
            f.write("  → Partial cross-engine generalization.\n")
        else:
            f.write("  No significant improvement found.\n")

    print(f"  Report: {report_path}")
    return report_path


def plot_comparison(all_results: Dict, output_dir: str):
    """Plot cross-engine comparison figures."""
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    engines = list(all_results.keys())

    # Figure 1: Pareto front per engine
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {'vietocr': '#2196F3', 'tesseract': '#FF9800', 'easyocr': '#4CAF50'}
    markers_surr = {'vietocr': 'o', 'tesseract': 's', 'easyocr': '^'}

    for engine in engines:
        r = all_results[engine]
        ks = [3, 5, 10, 15]
        surr_cers = [r['filter_results'][k] * 100 for k in ks]
        rand_cers = [r['random_results'][k] * 100 for k in ks]
        color = colors.get(engine, '#999999')
        marker = markers_surr.get(engine, 'o')

        ax.plot(ks, surr_cers, f'{marker}-', color=color, linewidth=2,
                markersize=8, label=f'{engine} (surrogate)', zorder=3)
        ax.plot(ks, rand_cers, f'{marker}--', color=color, linewidth=1.5,
                markersize=6, alpha=0.5, label=f'{engine} (random)')

    ax.set_xlabel('Number of OCR Calls (K)', fontsize=12)
    ax.set_ylabel('Mean CER (%)', fontsize=12)
    ax.set_title('Surrogate Filter vs Random Search — Multi-Engine', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([3, 5, 10, 15])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pareto_multiengine.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: test-set metric comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = ['r2', 'mae', 'spearman_global', 'spearman_per_image_mean']
    titles = [
        'R² (higher=better)',
        'MAE (lower=better)',
        'Global Spearman (higher=better)',
        'Per-Image Spearman (higher=better)',
    ]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [all_results[e]['test_metrics'][metric] for e in engines]
        bars = ax.bar(engines, vals, color=[colors.get(e, '#999') for e in engines], alpha=0.8)
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Significance heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ks = [3, 5, 10, 15]
    pvals = np.array([
        [all_results[e]['wilcoxon'][k]['p_value'] for k in ks]
        for e in engines
    ])
    # Convert to -log10(p) for visualization, cap at 4
    log_p = np.clip(-np.log10(pvals + 1e-10), 0, 4)
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

    ax.set_title('Statistical Significance: Surrogate Filter vs Random', fontsize=12)
    plt.colorbar(im, ax=ax, label='-log10(p-value)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'significance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures: {fig_dir}/")


# ============================================================================
# Main
# ============================================================================

def main(data_path: str, output_dir: str, engines: List[str] = None,
         test_ratio: float = 0.2,
         nopre_path: Optional[str] = None):
    print("=" * 70)
    print("MULTI-ENGINE SURROGATE TRAINING")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}, Images: {df[GROUP_COL].nunique()}")

    # Detect available engines from CSV columns
    available_cer_cols = [c for c in df.columns if c.startswith('cer_')]

    if available_cer_cols:
        # Multi-engine CSV (cer_vietocr, cer_tesseract, cer_easyocr)
        available_engines = [c.replace('cer_', '') for c in available_cer_cols]
    elif 'cer' in df.columns:
        # Single-engine CSV (backwards compatible)
        available_engines = ['vietocr']
        df['cer_vietocr'] = df['cer']
    else:
        print("ERROR: No CER columns found in CSV.")
        return

    # Filter to requested engines
    if engines:
        engine_list = [e for e in engines if e in available_engines]
    else:
        engine_list = available_engines

    print(f"  Available engines: {available_engines}")
    print(f"  Training for: {engine_list}")

    nopre_csv = resolve_exact_nopre_csv(data_path, nopre_path)
    exact_nopre_maps = load_exact_nopre_maps(nopre_csv, engine_list)
    missing_images = sorted(set(df[GROUP_COL].unique()) - set(next(iter(exact_nopre_maps.values())).keys()))
    if missing_images:
        raise KeyError(f"Exact NoPre CSV is missing {len(missing_images)} images; first few: {missing_images[:5]}")
    print(f"  Exact NoPre: {nopre_csv}")

    # Prepare features
    input_cols, feature_cols = prepare_data(df, 'cer_vietocr')
    print(f"  Input features: {len(input_cols)}")

    # Train for each engine
    all_results = {}
    for engine in engine_list:
        target_col = f'cer_{engine}'
        print(f"\n{'=' * 70}")
        print(f"ENGINE: {engine} (target={target_col})")
        print(f"{'=' * 70}")

        if target_col not in df.columns:
            print(f"  SKIP: column {target_col} not in data")
            continue

        print(f"  CER range: [{df[target_col].min():.4f}, {df[target_col].max():.4f}]")
        print(f"  CER mean: {df[target_col].mean():.4f}")

        results = train_one_engine(
            df,
            target_col,
            input_cols,
            feature_cols,
            exact_nopre_maps[engine],
            test_ratio,
        )

        print(f"\n  Best model: {results['best_model']}")
        print(f"  Train MAE: {results['train_metrics']['mae']:.4f}  "
              f"Train R²: {results['train_metrics']['r2']:.4f}")
        print(f"  Test  MAE: {results['test_metrics']['mae']:.4f}  "
              f"Test  R²: {results['test_metrics']['r2']:.4f}")
        ratio = results['test_metrics']['mae'] / results['train_metrics']['mae']
        print(f"  MAE ratio (test/train): {ratio:.2f}x "
              f"({'OK' if ratio < 1.5 else 'OVERFIT?'})")
        print(f"  Test GlobalSpearman: {results['test_metrics']['spearman_global']:.4f}")
        print(f"  Test PerImageSpearman: "
              f"{results['test_metrics']['spearman_per_image_mean']:.4f}")
        b = results['baselines']
        print(f"  Baselines: oracle={b['oracle']:.4f}  "
              f"fixed_best={b['fixed_best']:.4f}  "
              f"no_prep={b['no_preprocessing']:.4f}")
        print(f"  Surrogate filter results:")
        for k in [3, 5, 10, 15]:
            w = results['wilcoxon'][k]
            sig = "***" if w['p_value'] < 0.001 else "**" if w['p_value'] < 0.01 \
                else "*" if w['p_value'] < 0.05 else "ns"
            print(f"    K={k:2d}: surr={results['filter_results'][k]:.4f} "
                  f"vs rand={results['random_results'][k]:.4f} "
                  f"p={w['p_value']:.6f} ({sig})")

        # Save model
        model_path = os.path.join(output_dir, f'model_{engine}_{results["best_model"]}.pkl')
        with open(model_path, 'wb') as fp:
            pickle.dump(results['model'], fp)

        all_results[engine] = results

    # Report & plots
    print(f"\n{'=' * 70}")
    print("GENERATING REPORT & PLOTS")
    print(f"{'=' * 70}")

    write_report(all_results, output_dir, input_cols)
    plot_comparison(all_results, output_dir)

    # Save all metrics as JSON
    metrics_for_json = {}
    for engine, r in all_results.items():
        metrics_for_json[engine] = {
            'best_model': r['best_model'],
            'cv_results': r['cv_results'],
            'test_metrics': r['test_metrics'],
            'baselines': r['baselines'],
            'filter_results': {str(k): v for k, v in r['filter_results'].items()},
            'random_results': {str(k): v for k, v in r['random_results'].items()},
            'wilcoxon': {str(k): v for k, v in r['wilcoxon'].items()},
        }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        json.dump(metrics_for_json, fp, indent=2, default=str)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train surrogate models for multiple OCR engines")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to multi-engine CSV')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--engines', nargs='+', default=None,
                        help='Engines to train (default: all available)')
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--nopre-path', type=str, default=None,
                        help='Path to exact NoPre CSV; required because original OCR outputs are not distributed', required=True)

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = str(
            PROJECT_ROOT / f'results/recomputed/surrogate_multiengine_default_{timestamp}')

    main(args.data_path, args.output_dir, args.engines, args.test_ratio, args.nopre_path)
