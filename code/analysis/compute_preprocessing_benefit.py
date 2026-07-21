#!/usr/bin/env python3
"""
Compute per-image preprocessing benefit statistics from the corrected pipeline.

By default this script targets the latest raw/contour nested outputs:
  - sampled-pool oracle from the latest collect CSV
  - exact NoPre from the rerun-confirmed default CSV
  - aggregate FixedBest / Surrogate K from the latest nested metrics.json

Usage:
    python analysis/compute_preprocessing_benefit.py --setting raw
    python analysis/compute_preprocessing_benefit.py --setting contour
    python analysis/compute_preprocessing_benefit.py --lhs-csv ... --nopre-csv ... --metrics-json ...
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
ENGINES = ['vietocr', 'tesseract', 'easyocr']

DEFAULTS = {
    'raw': {
        'lhs_csv': PROJECT / 'outputs/multiengine_data_level1_fix_raw_g03757_nested/preprocessing_data.csv',
        'nopre_csv': PROJECT / 'outputs/default_output_multiengine_20260410_bbox/default_cer_20260410_053801.csv',
        'metrics_json': PROJECT / 'outputs/surrogate_multiengine_kfold_level1_fix_raw_g03757_nested/metrics.json',
    },
    'contour': {
        'lhs_csv': PROJECT / 'outputs_contour/multiengine_data_level1_fix_contour_g13757_nested/preprocessing_data.csv',
        'nopre_csv': PROJECT / 'outputs_contour/default_output_contour_bbox_20260410/default_cer_20260410_053815.csv',
        'metrics_json': PROJECT / 'outputs_contour/surrogate_multiengine_kfold_level1_fix_contour_g13757_nested/metrics.json',
    },
}


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    defaults = DEFAULTS[args.setting]
    lhs_csv = Path(args.lhs_csv) if args.lhs_csv else defaults['lhs_csv']
    nopre_csv = Path(args.nopre_csv) if args.nopre_csv else defaults['nopre_csv']
    metrics_json = Path(args.metrics_json) if args.metrics_json else defaults['metrics_json']

    for path in [lhs_csv, nopre_csv, metrics_json]:
        if not path.exists():
            raise FileNotFoundError(path)

    return {
        'lhs_csv': lhs_csv,
        'nopre_csv': nopre_csv,
        'metrics_json': metrics_json,
    }


def load_oracle(lhs_csv: Path, engines: List[str]) -> pd.DataFrame:
    df = pd.read_csv(lhs_csv)
    grouped = df.groupby('image_name').agg(
        **{f'oracle_{e}': (f'cer_{e}', 'min') for e in engines}
    ).reset_index()
    return grouped


def load_nopre(nopre_csv: Path, engines: List[str]) -> pd.DataFrame:
    df = pd.read_csv(nopre_csv).copy()
    if 'file_name' not in df.columns:
        raise ValueError(f"Expected file_name in {nopre_csv}")
    df['image_name'] = df['file_name'].map(lambda x: Path(str(x)).stem)
    keep_cols = ['image_name'] + [f'cer_{e}' for e in engines]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {nopre_csv}: {missing}")
    df = df[keep_cols].rename(columns={f'cer_{e}': f'nopre_{e}' for e in engines})
    return df


def per_image_stats(merged: pd.DataFrame, engine: str) -> Dict[str, float]:
    oracle_col = f'oracle_{engine}'
    nopre_col = f'nopre_{engine}'
    valid = merged[(merged[oracle_col] >= 0) & (merged[nopre_col] >= 0)].copy()
    valid['rel_imp'] = (valid[nopre_col] - valid[oracle_col]) / valid[nopre_col]
    benefit = (valid[oracle_col] < valid[nopre_col]).sum()
    total = len(valid)
    return {
        'n_total': int(total),
        'n_benefit': int(benefit),
        'pct_benefit': float(benefit / total * 100 if total else 0.0),
        'mean_rel_imp': float(valid['rel_imp'].mean() * 100),
        'median_rel_imp': float(valid['rel_imp'].median() * 100),
    }


def format_sources(paths: Dict[str, Path]) -> List[str]:
    return [
        f"  Oracle CSV:  {paths['lhs_csv']}",
        f"  NoPre CSV:   {paths['nopre_csv']}",
        f"  Metrics JSON:{paths['metrics_json']}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Compute per-image preprocessing benefit stats.")
    parser.add_argument('--setting', choices=sorted(DEFAULTS.keys()), default='raw')
    parser.add_argument('--lhs-csv', type=str, default=None)
    parser.add_argument('--nopre-csv', type=str, default=None)
    parser.add_argument('--metrics-json', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    paths = resolve_paths(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(paths['metrics_json']) as f:
        metrics = json.load(f)
    engines = [e for e in ENGINES if e in metrics]

    oracle = load_oracle(paths['lhs_csv'], engines)
    nopre = load_nopre(paths['nopre_csv'], engines)
    merged = oracle.merge(nopre, on='image_name', how='inner')
    if merged.empty:
        raise ValueError("No matched images between oracle and NoPre inputs.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'preprocessing_benefit_{args.setting}_{timestamp}'
    summary_path = out_dir / f'{stem}.txt'
    per_image_path = out_dir / f'{stem}.csv'

    lines = [
        "=" * 72,
        f"PREPROCESSING BENEFIT ANALYSIS ({args.setting.upper()})",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Images matched: {len(merged)}",
        "",
        "Sources:",
        *format_sources(paths),
        "=" * 72,
        "",
        "1. PER-IMAGE: Sampled-Pool Oracle vs Exact NoPre",
        "   (oracle = best CER across the sampled 50-candidate pool; NoPre = exact default run)",
        "",
        f"{'Engine':<12} {'Benefit':<14} {'Mean rel.imp':<16} {'Median rel.imp'}",
        "-" * 60,
    ]

    for eng in engines:
        stats = per_image_stats(merged, eng)
        lines.append(
            f"  {eng:<10} {stats['n_benefit']}/{stats['n_total']} "
            f"({stats['pct_benefit']:.0f}%)     "
            f"{stats['mean_rel_imp']:+.1f}%           "
            f"{stats['median_rel_imp']:+.1f}%"
        )

    lines += [
        "",
        "  Note: 'rel. improvement' = (NoPre - Oracle) / NoPre per image, then averaged.",
        "",
        "2. AGGREGATE: Nested K-fold baselines vs Exact NoPre",
        "",
        f"{'Engine':<12} {'NoPre':>8} {'FixedBest':>10} {'FB vs NoPre':>12} "
        f"{'Surr K=3':>10} {'Surr vs NoPre':>14}",
        "-" * 72,
    ]

    for eng in engines:
        m = metrics[eng]
        nopre_agg = merged[f'nopre_{eng}'].mean()
        fixed_best = m['baselines']['fixed_best']
        surr_k3 = m['filter_results']['3']
        fb_diff = (fixed_best - nopre_agg) / nopre_agg * 100
        surr_diff = (surr_k3 - nopre_agg) / nopre_agg * 100
        lines.append(
            f"  {eng:<10} {nopre_agg*100:>7.2f}% {fixed_best*100:>9.2f}% "
            f"{fb_diff:>+11.1f}% {surr_k3*100:>9.2f}% {surr_diff:>+13.1f}%"
        )

    lines += ["", "=" * 72]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")

    rows = []
    for _, row in merged.iterrows():
        out = {'image_name': row['image_name']}
        for eng in engines:
            nopre_val = row[f'nopre_{eng}']
            oracle_val = row[f'oracle_{eng}']
            out[f'nopre_{eng}'] = round(nopre_val, 6)
            out[f'oracle_{eng}'] = round(oracle_val, 6)
            out[f'rel_imp_{eng}'] = round((nopre_val - oracle_val) / nopre_val, 6) if nopre_val > 0 else None
            out[f'benefit_{eng}'] = int(oracle_val < nopre_val)
        rows.append(out)

    fieldnames = ['image_name'] + [
        field
        for eng in engines
        for field in (f'nopre_{eng}', f'oracle_{eng}', f'rel_imp_{eng}', f'benefit_{eng}')
    ]
    with open(per_image_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary:  {summary_path}")
    print(f"Per-image:{per_image_path}")


if __name__ == '__main__':
    main()
