#!/usr/bin/env python3
"""
Run feature ablations on the current nested GroupKFold surrogate pipeline.

This script reuses the corrected multi-engine k-fold logic:
  - executed-space sampled pools
  - exact NoPre baselines
  - nested model-family selection inside each outer fold

By default it targets the latest corrected raw outputs. Use --setting contour
or explicit --data-path/--nopre-path overrides to evaluate other datasets.
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from main.train_surrogate_multiengine_kfold import (  # noqa: E402
    FEATURE_COLS_BASE,
    FEATURE_COLS_NEW,
    PARAM_COLS,
    GROUP_COL,
    load_exact_nopre_maps,
    plot_comparison,
    resolve_exact_nopre_csv,
    train_one_engine_kfold,
    write_report,
)

DEFAULTS = {
    'raw': {
        'data_path': RELEASE_ROOT / 'data/preprocessing_response_tables/raw_preprocessing_data.csv',
        'output_dir': RELEASE_ROOT / 'results/recomputed/ablation_nested_raw',
    },
    'contour': {
        'data_path': RELEASE_ROOT / 'data/preprocessing_response_tables/contour_preprocessing_data.csv',
        'output_dir': RELEASE_ROOT / 'results/recomputed/ablation_nested_contour',
    },
}

INTERACTIONS: List[Tuple[str, str]] = [
    ('orig_blur', 'gaussian_ksize'),
    ('orig_contrast', 'clahe_clip'),
    ('orig_brightness', 'gamma'),
    ('orig_noise', 'bilateral_d'),
    ('orig_entropy', 'sharpen'),
]

CONDITIONS = OrderedDict({
    'full': 'Full (22): base + new + params + interactions',
    'no_new': 'No new 5 features (17): base + params + interactions',
    'no_inter': 'No interactions (17): base + new + params',
    'base_only': 'Base only (12): base + params',
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run nested feature ablations on the corrected surrogate pipeline.'
    )
    parser.add_argument('--setting', choices=sorted(DEFAULTS.keys()), default='raw')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--nopre-path', type=str, required=True,
                        help='Exact NoPre per-image CSV; not included in the public release')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--engines', nargs='+', default=None)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument(
        '--condition',
        choices=list(CONDITIONS.keys()) + ['all'],
        default='all',
        help='Ablation condition to run (default: all)',
    )
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> Dict[str, Path]:
    defaults = DEFAULTS[args.setting]
    data_path = Path(args.data_path) if args.data_path else defaults['data_path']
    nopre_path = Path(args.nopre_path)
    output_dir = Path(args.output_dir) if args.output_dir else defaults['output_dir']

    for path in [data_path, nopre_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    return {
        'data_path': data_path,
        'nopre_path': nopre_path,
        'output_dir': output_dir,
    }


def add_interaction_columns(df: pd.DataFrame) -> List[str]:
    interaction_cols = []
    for feat, param in INTERACTIONS:
        if feat in df.columns and param in df.columns:
            col_name = f'inter_{feat.replace("orig_", "")}_{param}'
            if col_name not in df.columns:
                df[col_name] = df[feat] * df[param]
            interaction_cols.append(col_name)
    return interaction_cols


def build_input_cols(df: pd.DataFrame, condition: str) -> List[str]:
    available_base = [c for c in FEATURE_COLS_BASE if c in df.columns]
    available_new = [c for c in FEATURE_COLS_NEW if c in df.columns]
    param_cols = [c for c in PARAM_COLS if c in df.columns]
    interaction_cols = add_interaction_columns(df)

    if condition == 'full':
        return available_base + available_new + param_cols + interaction_cols
    if condition == 'no_new':
        return available_base + param_cols + interaction_cols
    if condition == 'no_inter':
        return available_base + available_new + param_cols
    if condition == 'base_only':
        return available_base + param_cols
    raise ValueError(f'Unsupported condition: {condition}')


def get_condition_output_dir(root_output_dir: Path, condition: str, run_all: bool) -> Path:
    return root_output_dir / condition if run_all else root_output_dir


def run_condition(
    df: pd.DataFrame,
    condition: str,
    output_dir: Path,
    engine_list: List[str],
    exact_nopre_maps: Dict[str, Dict[str, float]],
    n_folds: int,
) -> Dict:
    print("\n" + "=" * 70)
    print(f"ABLATION CONDITION: {condition} — {CONDITIONS[condition]}")
    print("=" * 70)

    run_df = df.copy()
    input_cols = build_input_cols(run_df, condition)
    print(f"  Input features: {len(input_cols)}")

    all_results = {}
    for engine in engine_list:
        target_col = f'cer_{engine}'
        if target_col not in run_df.columns:
            print(f"  SKIP {engine}: missing {target_col}")
            continue

        print(f"\nENGINE: {engine}")
        print(f"  CER mean={run_df[target_col].mean():.4f}  "
              f"range=[{run_df[target_col].min():.4f}, {run_df[target_col].max():.4f}]")
        results = train_one_engine_kfold(
            run_df,
            target_col,
            input_cols,
            exact_nopre_maps[engine],
            n_folds,
        )
        all_results[engine] = results

        m = results['cv_metrics']
        print(f"  CV summary: MAE={m['mae_mean']:.4f}±{m['mae_std']:.4f}  "
              f"Spearman={m['spearman_mean']:.4f}±{m['spearman_std']:.4f}")
        print(f"  Final model family: {results['best_model']}")
        print(f"  Outer-fold selections: {results['model_selection_summary']}")

    os.makedirs(output_dir, exist_ok=True)

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
        pd.DataFrame(rows).to_csv(output_dir / 'surr_selected_params.csv', index=False)

    write_report(all_results, str(output_dir), input_cols, n_folds)
    plot_comparison(all_results, str(output_dir))

    metrics_json = {}
    for engine, result in all_results.items():
        metrics_json[engine] = {
            'best_model': result['best_model'],
            'selected_models_per_fold': result.get('selected_models_per_fold', []),
            'cv_metrics': result['cv_metrics'],
            'baselines': result['baselines'],
            'filter_results': {str(k): v for k, v in result['filter_results'].items()},
            'random_results': {str(k): v for k, v in result['random_results'].items()},
            'fixed_topk_results': {
                str(k): v for k, v in result['fixed_topk_results'].items()
            },
            'wilcoxon': {str(k): v for k, v in result['wilcoxon'].items()},
        }
    with open(output_dir / 'metrics.json', 'w') as fp:
        json.dump(metrics_json, fp, indent=2, default=str)

    for engine, result in all_results.items():
        model_path = output_dir / f"model_{engine}_{result['best_model']}.pkl"
        with open(model_path, 'wb') as fp:
            pickle.dump(result['model'], fp)

    return {
        'condition': condition,
        'label': CONDITIONS[condition],
        'n_features': len(input_cols),
        'metrics': metrics_json,
        'output_dir': str(output_dir),
    }


def write_summary(summary: Dict[str, Dict], output_dir: Path):
    summary_json_path = output_dir / 'ablation_summary.json'
    summary_txt_path = output_dir / 'ablation_summary.txt'

    with open(summary_json_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)

    lines = [
        "=" * 80,
        "NESTED FEATURE ABLATION SUMMARY",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        f"{'Condition':<16} {'Feat':>4} {'Engine':<12} {'Spearman':>10} {'R²':>10} {'K=3':>10} {'K=10':>10}",
        "-" * 80,
    ]

    for condition in summary['condition_order']:
        cond = summary['conditions'][condition]
        first_engine = True
        for engine, metrics in cond['metrics'].items():
            cv = metrics['cv_metrics']
            filt = metrics['filter_results']
            label = condition if first_engine else ''
            feat = cond['n_features'] if first_engine else ''
            lines.append(
                f"{label:<16} {str(feat):>4} {engine:<12} "
                f"{cv['spearman_mean']:>10.4f} {cv['r2_mean']:>10.4f} "
                f"{filt['3']:>10.4f} {filt['10']:>10.4f}"
            )
            first_engine = False
        lines.append("")

    with open(summary_txt_path, 'w') as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\nSummary JSON: {summary_json_path}")
    print(f"Summary TXT:  {summary_txt_path}")


def main():
    args = parse_args()
    paths = resolve_run_paths(args)

    print("=" * 70)
    print("NESTED FEATURE ABLATION")
    print("=" * 70)
    print(f"Data:  {paths['data_path']}")
    print(f"NoPre: {paths['nopre_path']}")

    df = pd.read_csv(paths['data_path'])
    print(f"Rows: {len(df):,}, Images: {df[GROUP_COL].nunique()}")

    available_engines = [c.replace('cer_', '') for c in df.columns if c.startswith('cer_')]
    if not available_engines:
        raise ValueError("No CER columns found in data.")
    engine_list = [e for e in (args.engines or available_engines) if e in available_engines]
    print(f"Engines: {engine_list}")

    nopre_csv = resolve_exact_nopre_csv(str(paths['data_path']), str(paths['nopre_path']))
    exact_nopre_maps = load_exact_nopre_maps(nopre_csv, engine_list)
    missing_images = sorted(
        set(df[GROUP_COL].unique()) - set(next(iter(exact_nopre_maps.values())).keys())
    )
    if missing_images:
        raise KeyError(
            f"Exact NoPre CSV is missing {len(missing_images)} images; "
            f"first few: {missing_images[:5]}"
        )

    root_output_dir = paths['output_dir']
    root_output_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(CONDITIONS.keys()) if args.condition == 'all' else [args.condition]
    summary = {
        'condition_order': conditions,
        'conditions': {},
    }

    for condition in conditions:
        condition_output_dir = get_condition_output_dir(
            root_output_dir, condition, run_all=args.condition == 'all'
        )
        result = run_condition(
            df,
            condition,
            condition_output_dir,
            engine_list,
            exact_nopre_maps,
            args.n_folds,
        )
        summary['conditions'][condition] = result

    write_summary(summary, root_output_dir)
    print(f"\nDone. Output: {root_output_dir}")


if __name__ == '__main__':
    main()
