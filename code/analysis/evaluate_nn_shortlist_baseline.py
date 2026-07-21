#!/usr/bin/env python3
"""
Evaluate a nearest-neighbor preprocessing-shortlist baseline.

For each held-out page, this baseline finds visually similar training pages
using OCR-free page features, takes the oracle-best preprocessing combo from
those neighbors, and evaluates the resulting K-combo shortlist on the held-out
page. It uses the already collected candidate-response table, so no OCR rerun
is required.

This is intended as a stronger page-conditioned baseline than FixedTopK:

    test page -> nearest training pages in feature space
              -> each neighbor's best training combo
              -> top-K unique neighbor combos
              -> best observed CER among those K combos for evaluation

Usage:
    python analysis/evaluate_nn_shortlist_baseline.py \
        --data-path outputs/multiengine_data_level1_fix_raw_g03757_nested/preprocessing_data.csv \
        --output-dir outputs/nn_shortlist_raw_g03757 \
        --compare-metrics-json outputs/surrogate_multiengine_kfold_level1_fix_raw_g03757_nested_fixedtopk/metrics.json \
        --compare-per-page-path outputs/surrogate_multiengine_kfold_level1_fix_raw_g03757_nested_fixedtopk
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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

DEFAULT_K_VALUES = [3, 5, 10, 15]


Combo = Tuple[int, int, float, float, float]


def combo_from_row(row: pd.Series) -> Combo:
    return (
        int(row["gaussian_ksize"]),
        int(row["bilateral_d"]),
        round(float(row["clahe_clip"]), 10),
        round(float(row["gamma"]), 10),
        round(float(row["sharpen"]), 10),
    )


def combo_from_values(values: Sequence[object]) -> Combo:
    return (
        int(values[0]),
        int(values[1]),
        round(float(values[2]), 10),
        round(float(values[3]), 10),
        round(float(values[4]), 10),
    )


def combo_to_record(prefix: str, combo: Combo) -> Dict[str, object]:
    return {
        f"{prefix}_{col}": value
        for col, value in zip(PARAM_COLS, combo)
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_wilcoxon(a: Iterable[float], b: Iterable[float], alternative: str = "two-sided") -> Dict[str, object]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or np.allclose(x, y):
        return {"stat": math.nan, "p": math.nan, "n": int(len(x)), "alternative": alternative}
    stat, p_value = wilcoxon(x, y, alternative=alternative)
    return {"stat": float(stat), "p": float(p_value), "n": int(len(x)), "alternative": alternative}


def pairwise_stats(a: Iterable[float], b: Iterable[float]) -> Dict[str, object]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = x[mask] - y[mask]
    if len(diff) == 0:
        return {
            "n": 0,
            "mean_diff": math.nan,
            "median_diff": math.nan,
            "win_rate": math.nan,
            "tie_rate": math.nan,
            "loss_rate": math.nan,
        }
    return {
        "n": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "win_rate": float(np.mean(diff < 0)),
        "tie_rate": float(np.mean(np.isclose(diff, 0.0))),
        "loss_rate": float(np.mean(diff > 0)),
    }


def available_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in FEATURE_COLS_BASE + FEATURE_COLS_NEW if c in df.columns]
    missing = [c for c in FEATURE_COLS_BASE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base feature columns: {missing}")
    return cols


def infer_engines(df: pd.DataFrame, explicit_engines: Optional[List[str]]) -> List[str]:
    if explicit_engines:
        missing = [f"cer_{engine}" for engine in explicit_engines if f"cer_{engine}" not in df.columns]
        if missing:
            raise ValueError(f"Missing requested engine CER columns: {missing}")
        return explicit_engines
    engines = sorted(c.replace("cer_", "") for c in df.columns if c.startswith("cer_"))
    if not engines:
        raise ValueError("No cer_<engine> columns found")
    return engines


def build_page_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    page_df = df[[GROUP_COL] + feature_cols].drop_duplicates(subset=[GROUP_COL]).copy()
    if page_df[GROUP_COL].duplicated().any():
        dupes = page_df.loc[page_df[GROUP_COL].duplicated(), GROUP_COL].tolist()[:5]
        raise ValueError(f"Duplicate page feature rows: {dupes}")
    return page_df.sort_values(GROUP_COL).reset_index(drop=True)


def build_combo_cer_maps(df: pd.DataFrame, engines: List[str]) -> Dict[str, Dict[str, Dict[Combo, float]]]:
    maps: Dict[str, Dict[str, Dict[Combo, float]]] = {
        engine: defaultdict(dict) for engine in engines
    }
    for _, row in df.iterrows():
        image_name = str(row[GROUP_COL])
        combo = combo_from_row(row)
        for engine in engines:
            maps[engine][image_name][combo] = float(row[f"cer_{engine}"])
    return maps


def best_combo_by_page(train_df: pd.DataFrame, target_col: str) -> Dict[str, Combo]:
    idx = train_df.groupby(GROUP_COL)[target_col].idxmin()
    rows = train_df.loc[idx, [GROUP_COL] + PARAM_COLS]
    return {
        str(row[GROUP_COL]): combo_from_row(row)
        for _, row in rows.iterrows()
    }


def fixed_topk_combos(train_df: pd.DataFrame, target_col: str, max_k: int) -> List[Combo]:
    means = train_df.groupby(PARAM_COLS, dropna=False)[target_col].mean().sort_values()
    return [combo_from_values(idx) for idx in means.index[:max_k]]


def cer_for_shortlist(combo_cer: Dict[Combo, float], combos: Sequence[Combo]) -> float:
    values = [combo_cer[combo] for combo in combos if combo in combo_cer]
    return float(np.min(values)) if values else math.nan


def unique_preserve_order(combos: Iterable[Combo]) -> List[Combo]:
    seen = set()
    out = []
    for combo in combos:
        if combo in seen:
            continue
        seen.add(combo)
        out.append(combo)
    return out


def load_compare_metrics(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Compare metrics JSON not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _infer_engine_from_per_page_path(path: Path) -> Optional[str]:
    stem = path.stem
    prefix = "per_page_results_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None


def load_compare_per_page(
    path: Optional[str],
    engines: List[str],
    k_values: List[int],
) -> Dict[str, pd.DataFrame]:
    """Load surrogate per-page results for paired NN-vs-surrogate tests.

    Accepts either:
      - a directory containing per_page_results_<engine>.csv files; or
      - one CSV with an engine column; or
      - one per_page_results_<engine>.csv file.
    """
    if not path:
        return {}

    compare_path = Path(path)
    if not compare_path.exists():
        raise FileNotFoundError(f"Compare per-page path not found: {compare_path}")

    files: List[Path]
    if compare_path.is_dir():
        files = sorted(compare_path.glob("per_page_results_*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No per_page_results_<engine>.csv files found in {compare_path}. "
                "Rerun main/train_surrogate_multiengine_kfold.py with the current script."
            )
    else:
        files = [compare_path]

    out: Dict[str, pd.DataFrame] = {}
    required_cols = [GROUP_COL] + [f"surr_k{k}" for k in k_values]

    for file_path in files:
        frame = pd.read_csv(file_path)
        missing = [c for c in required_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{file_path} is missing required columns: {missing}")

        if "engine" in frame.columns:
            for engine, group in frame.groupby("engine"):
                engine = str(engine)
                if engine not in engines:
                    continue
                out[engine] = group[required_cols].copy()
            continue

        engine = _infer_engine_from_per_page_path(file_path)
        if engine is None:
            if len(engines) == 1:
                engine = engines[0]
            else:
                raise ValueError(
                    f"Cannot infer engine from {file_path}; expected "
                    "per_page_results_<engine>.csv or an 'engine' column."
                )
        if engine in engines:
            out[engine] = frame[required_cols].copy()

    return out


def evaluate_engine(
    df: pd.DataFrame,
    page_df: pd.DataFrame,
    combo_maps: Dict[str, Dict[str, Dict[Combo, float]]],
    engine: str,
    feature_cols: List[str],
    k_values: List[int],
    n_folds: int,
    random_trials: int,
    distance_metric: str,
    compare_per_page: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    target_col = f"cer_{engine}"
    max_k = max(k_values)
    groups = page_df[GROUP_COL].values
    gkf = GroupKFold(n_splits=n_folds)
    image_groups = {
        str(image_name): group.copy()
        for image_name, group in df.groupby(GROUP_COL)
    }
    per_image_rows = []

    for fold_idx, (train_page_idx, test_page_idx) in enumerate(gkf.split(page_df[feature_cols].values, groups=groups), start=1):
        train_pages = page_df.iloc[train_page_idx].copy()
        test_pages = page_df.iloc[test_page_idx].copy()
        train_names = train_pages[GROUP_COL].astype(str).tolist()
        test_names = test_pages[GROUP_COL].astype(str).tolist()

        train_df = df[df[GROUP_COL].isin(train_names)].copy()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_pages[feature_cols].values)
        x_test = scaler.transform(test_pages[feature_cols].values)
        distances = pairwise_distances(x_test, x_train, metric=distance_metric)

        neighbor_best = best_combo_by_page(train_df, target_col)
        fixed_combos = fixed_topk_combos(train_df, target_col, max_k)

        for test_pos, image_name in enumerate(test_names):
            neighbor_order = np.argsort(distances[test_pos])
            neighbor_names = [train_names[i] for i in neighbor_order]
            nn_combos = unique_preserve_order(neighbor_best[name] for name in neighbor_names)
            if len(nn_combos) < max_k:
                nn_combos = unique_preserve_order(nn_combos + fixed_combos)
            nn_combos = nn_combos[:max_k]

            image_combo_cer = combo_maps[engine][image_name]
            all_cer_values = list(image_combo_cer.values())
            row = {
                "engine": engine,
                "fold": fold_idx,
                GROUP_COL: image_name,
                "samporacle_cer": float(np.min(all_cer_values)),
                "n_unique_nn_combos": int(len(nn_combos)),
            }

            # Nearest-neighbor distance diagnostics.
            for rank, train_name in enumerate(neighbor_names[:max_k], start=1):
                row[f"nn{rank}_image_name"] = train_name
                row[f"nn{rank}_distance"] = float(distances[test_pos, neighbor_order[rank - 1]])
                row.update(combo_to_record(f"nn{rank}", neighbor_best[train_name]))

            for k in k_values:
                nn_shortlist = nn_combos[:k]
                fixed_shortlist = fixed_combos[:k]
                row[f"nn_k{k}_cer"] = cer_for_shortlist(image_combo_cer, nn_shortlist)
                row[f"fixed_topk_k{k}_cer"] = cer_for_shortlist(image_combo_cer, fixed_shortlist)
                random_scores = []
                image_group = image_groups[image_name]
                for seed in range(random_trials):
                    sampled = image_group.sample(min(k, len(image_group)), random_state=seed)
                    random_scores.append(float(sampled[target_col].min()))
                row[f"random_k{k}_cer"] = float(np.mean(random_scores))
                row[f"nn_minus_fixed_topk_k{k}"] = row[f"nn_k{k}_cer"] - row[f"fixed_topk_k{k}_cer"]
                row[f"nn_minus_random_k{k}"] = row[f"nn_k{k}_cer"] - row[f"random_k{k}_cer"]

            per_image_rows.append(row)

    per_image = pd.DataFrame(per_image_rows).sort_values(["engine", GROUP_COL]).reset_index(drop=True)
    if compare_per_page is not None:
        rename_map = {f"surr_k{k}": f"surr_k{k}_cer" for k in k_values}
        compare_cols = [GROUP_COL] + [f"surr_k{k}" for k in k_values]
        compare_frame = compare_per_page[compare_cols].rename(columns=rename_map).copy()
        compare_frame[GROUP_COL] = compare_frame[GROUP_COL].astype(str)
        per_image = per_image.merge(compare_frame, on=GROUP_COL, how="left", validate="one_to_one")
        for k in k_values:
            missing = int(per_image[f"surr_k{k}_cer"].isna().sum())
            if missing:
                raise ValueError(
                    f"Missing surrogate per-page values for {missing} {engine} images at K={k}"
                )
    summary = summarize_engine(per_image, engine, k_values)
    return per_image, summary


def summarize_engine(per_image: pd.DataFrame, engine: str, k_values: List[int]) -> Dict:
    means = {"samporacle_cer": float(per_image["samporacle_cer"].mean())}
    tests = {}
    pairwise = {}

    for k in k_values:
        nn_col = f"nn_k{k}_cer"
        ftk_col = f"fixed_topk_k{k}_cer"
        rand_col = f"random_k{k}_cer"
        means[nn_col] = float(per_image[nn_col].mean())
        means[ftk_col] = float(per_image[ftk_col].mean())
        means[rand_col] = float(per_image[rand_col].mean())
        tests[f"nn_vs_fixed_topk_k{k}"] = safe_wilcoxon(per_image[nn_col], per_image[ftk_col])
        tests[f"fixed_topk_vs_nn_k{k}"] = safe_wilcoxon(per_image[ftk_col], per_image[nn_col])
        tests[f"nn_vs_random_k{k}"] = safe_wilcoxon(per_image[nn_col], per_image[rand_col])
        pairwise[f"nn_minus_fixed_topk_k{k}"] = pairwise_stats(per_image[nn_col], per_image[ftk_col])
        pairwise[f"nn_minus_random_k{k}"] = pairwise_stats(per_image[nn_col], per_image[rand_col])

        surr_col = f"surr_k{k}_cer"
        if surr_col in per_image.columns:
            means[surr_col] = float(per_image[surr_col].mean())
            tests[f"nn_vs_surrogate_k{k}"] = safe_wilcoxon(per_image[nn_col], per_image[surr_col])
            tests[f"surrogate_vs_nn_k{k}"] = safe_wilcoxon(per_image[surr_col], per_image[nn_col])
            pairwise[f"nn_minus_surrogate_k{k}"] = pairwise_stats(per_image[nn_col], per_image[surr_col])

    return {
        "engine": engine,
        "n_images": int(per_image[GROUP_COL].nunique()),
        "means": means,
        "wilcoxon": tests,
        "pairwise": pairwise,
    }


def compare_surrogate_mean(compare_metrics: Optional[Dict], engine: str, k: int) -> Optional[float]:
    if not compare_metrics or engine not in compare_metrics:
        return None
    result = compare_metrics[engine]
    values = result.get("filter_results", {})
    value = values.get(str(k), values.get(k))
    return float(value) if value is not None else None


def write_report(
    summaries: Dict[str, Dict],
    output_dir: Path,
    k_values: List[int],
    compare_metrics: Optional[Dict],
    data_path: Path,
    feature_cols: List[str],
    distance_metric: str,
    has_compare_per_page: bool,
) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("NEAREST-NEIGHBOR SHORTLIST BASELINE")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Data: {data_path}")
    lines.append(f"Distance metric: {distance_metric}")
    lines.append(f"Page features ({len(feature_cols)}): {', '.join(feature_cols)}")
    lines.append("")
    lines.append("NN-K: choose each held-out page's K candidate combos from the")
    lines.append("oracle-best combos of its nearest training pages in OCR-free feature space.")
    lines.append("Reference text/CER are used only for evaluation and for neighbor best-combo")
    lines.append("selection inside the training fold.")
    lines.append("")

    for engine, summary in summaries.items():
        lines.append("-" * 80)
        lines.append(f"Engine: {engine}")
        lines.append(f"Images: {summary['n_images']}")
        lines.append(f"SampledOracle: {format_pct(summary['means']['samporacle_cer'])}")
        lines.append("")
        lines.append("Mean CER")
        header = "  K    NN-K     FixedTopK  RandomK   SurrK    NN-Surr"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for k in k_values:
            nn = summary["means"][f"nn_k{k}_cer"]
            ftk = summary["means"][f"fixed_topk_k{k}_cer"]
            rand = summary["means"][f"random_k{k}_cer"]
            surr = summary["means"].get(f"surr_k{k}_cer")
            if surr is None:
                surr = compare_surrogate_mean(compare_metrics, engine, k)
            if surr is None:
                surr_text = "   n/a "
                diff_text = "    n/a "
            else:
                surr_text = f"{surr * 100:7.2f}%"
                diff_text = f"{(nn - surr) * 100:+8.2f}pp"
            lines.append(
                f"  {k:<4d} {nn * 100:7.2f}%  {ftk * 100:7.2f}%  "
                f"{rand * 100:7.2f}%  {surr_text}  {diff_text}"
            )
        if has_compare_per_page:
            lines.append("  Surrogate values are per-page values loaded from surrogate per-page CSVs.")
            lines.append("  NN-vs-Surrogate Wilcoxon tests below are paired by image.")
        elif compare_metrics:
            lines.append("  Surrogate values are aggregate means loaded from compare metrics JSON.")
            lines.append("  Paired NN-vs-Surrogate tests require --compare-per-page-path.")
        lines.append("")
        lines.append("Wilcoxon two-sided paired tests")
        lines.append("  p-values compare paired page-level CER values")
        for k in k_values:
            nn_ftk = summary["wilcoxon"][f"nn_vs_fixed_topk_k{k}"]
            ftk_nn = summary["wilcoxon"][f"fixed_topk_vs_nn_k{k}"]
            nn_rand = summary["wilcoxon"][f"nn_vs_random_k{k}"]
            line = (
                f"  K={k:<2d} NN vs FixedTopK p={nn_ftk['p']:.6g}, "
                f"FixedTopK vs NN p={ftk_nn['p']:.6g}, "
                f"NN vs Random p={nn_rand['p']:.6g}"
            )
            nn_surr_key = f"nn_vs_surrogate_k{k}"
            if nn_surr_key in summary["wilcoxon"]:
                nn_surr = summary["wilcoxon"][nn_surr_key]
                surr_nn = summary["wilcoxon"][f"surrogate_vs_nn_k{k}"]
                line += (
                    f", NN<Surrogate p={nn_surr['p']:.6g}, "
                    f"Surrogate<NN p={surr_nn['p']:.6g}"
                )
            lines.append(line)
        lines.append("")
        lines.append("Pairwise NN differences")
        for k in k_values:
            ftk_stats = summary["pairwise"][f"nn_minus_fixed_topk_k{k}"]
            rand_stats = summary["pairwise"][f"nn_minus_random_k{k}"]
            lines.append(
                f"  K={k:<2d} NN-FixedTopK mean={ftk_stats['mean_diff'] * 100:+.2f}pp, "
                f"win={ftk_stats['win_rate'] * 100:.1f}%, "
                f"tie={ftk_stats['tie_rate'] * 100:.1f}%, "
                f"loss={ftk_stats['loss_rate'] * 100:.1f}%"
            )
            lines.append(
                f"       NN-Random    mean={rand_stats['mean_diff'] * 100:+.2f}pp, "
                f"win={rand_stats['win_rate'] * 100:.1f}%, "
                f"tie={rand_stats['tie_rate'] * 100:.1f}%, "
                f"loss={rand_stats['loss_rate'] * 100:.1f}%"
            )
            surr_key = f"nn_minus_surrogate_k{k}"
            if surr_key in summary["pairwise"]:
                surr_stats = summary["pairwise"][surr_key]
                lines.append(
                    f"       NN-Surrogate mean={surr_stats['mean_diff'] * 100:+.2f}pp, "
                    f"win={surr_stats['win_rate'] * 100:.1f}%, "
                    f"tie={surr_stats['tie_rate'] * 100:.1f}%, "
                    f"loss={surr_stats['loss_rate'] * 100:.1f}%"
                )
        lines.append("")

    (output_dir / "nn_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a nearest-neighbor preprocessing shortlist baseline."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--distance-metric", default="euclidean",
                        help="Metric accepted by sklearn.metrics.pairwise_distances.")
    parser.add_argument("--compare-metrics-json", default=None,
                        help="Optional surrogate metrics JSON for aggregate comparison.")
    parser.add_argument("--compare-per-page-path", default=None,
                        help=("Optional surrogate per-page CSV or directory containing "
                              "per_page_results_<engine>.csv files for paired NN-vs-surrogate tests."))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    feature_cols = available_feature_cols(df)
    engines = infer_engines(df, args.engines)
    k_values = sorted(set(args.k_values))
    page_df = build_page_features(df, feature_cols)
    combo_maps = build_combo_cer_maps(df, engines)
    compare_metrics = load_compare_metrics(args.compare_metrics_json)
    compare_per_page = load_compare_per_page(args.compare_per_page_path, engines, k_values)

    print(f"Data          : {data_path}")
    print(f"Output        : {output_dir}")
    print(f"Images        : {page_df[GROUP_COL].nunique()}")
    print(f"Engines       : {engines}")
    print(f"K values      : {k_values}")
    print(f"Features      : {len(feature_cols)}")
    print(f"Distance      : {args.distance_metric}")
    if compare_per_page:
        print(f"Paired compare: surrogate per-page for {sorted(compare_per_page.keys())}")

    all_per_image = []
    summaries = {}
    for engine in engines:
        print(f"\nEvaluating NN baseline for {engine}")
        per_image, summary = evaluate_engine(
            df=df,
            page_df=page_df,
            combo_maps=combo_maps,
            engine=engine,
            feature_cols=feature_cols,
            k_values=k_values,
            n_folds=args.n_folds,
            random_trials=args.random_trials,
            distance_metric=args.distance_metric,
            compare_per_page=compare_per_page.get(engine),
        )
        summaries[engine] = summary
        all_per_image.append(per_image)
        for k in k_values:
            nn = summary["means"][f"nn_k{k}_cer"]
            ftk = summary["means"][f"fixed_topk_k{k}_cer"]
            rand = summary["means"][f"random_k{k}_cer"]
            surr = summary["means"].get(f"surr_k{k}_cer")
            if surr is None:
                print(f"  K={k:<2d} NN={nn:.4f} FixedTopK={ftk:.4f} Random={rand:.4f}")
            else:
                print(
                    f"  K={k:<2d} NN={nn:.4f} FixedTopK={ftk:.4f} "
                    f"Random={rand:.4f} Surr={surr:.4f}"
                )

    per_image_df = pd.concat(all_per_image, ignore_index=True)
    per_image_path = output_dir / "nn_per_image.csv"
    metrics_path = output_dir / "nn_metrics.json"
    per_image_df.to_csv(per_image_path, index=False)
    metrics_path.write_text(json.dumps(to_jsonable(summaries), indent=2), encoding="utf-8")
    write_report(
        summaries=summaries,
        output_dir=output_dir,
        k_values=k_values,
        compare_metrics=compare_metrics,
        data_path=data_path,
        feature_cols=feature_cols,
        distance_metric=args.distance_metric,
        has_compare_per_page=bool(compare_per_page),
    )

    print(f"\nReport   : {output_dir / 'nn_report.txt'}")
    print(f"Per-image: {per_image_path}")
    print(f"Metrics  : {metrics_path}")


if __name__ == "__main__":
    main()
