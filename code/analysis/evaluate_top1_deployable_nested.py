#!/usr/bin/env python3
"""
Evaluate deployable top-1 surrogate selection with nested GroupKFold.

This script uses already collected preprocessing/OCR data. It does not rerun
OCR. For each outer test image, it selects the candidate with the lowest
out-of-fold predicted CER and evaluates the actual CER of that candidate.

Top1Pred is the deployable one-call policy:
  image -> features -> predict CER for the 50 candidate params -> pick rank #1

BestOfK remains a shortlist-quality upper bound:
  pick top-K by predicted CER -> use ground truth CER to choose the best within K
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from main.train_surrogate_multiengine_kfold import (  # noqa: E402
    GROUP_COL,
    PARAM_COLS,
    get_models,
    load_exact_nopre_maps,
    prepare_data,
    select_best_model,
    summarize_model_selection,
)


def _safe_wilcoxon(a: List[float], b: List[float], alternative: str = "less") -> Dict[str, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or np.allclose(x, y):
        return {"stat": float("nan"), "p": float("nan"), "n": int(len(x))}
    stat, p_value = wilcoxon(x, y, alternative=alternative)
    return {"stat": float(stat), "p": float(p_value), "n": int(len(x))}


def _mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr))


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pairwise_stats(a: List[float], b: List[float]) -> Dict[str, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = x[mask] - y[mask]
    if len(diff) == 0:
        return {
            "n": 0,
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "win_rate": float("nan"),
            "tie_rate": float("nan"),
            "loss_rate": float("nan"),
        }
    return {
        "n": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "win_rate": float(np.mean(diff < 0)),
        "tie_rate": float(np.mean(np.isclose(diff, 0.0))),
        "loss_rate": float(np.mean(diff > 0)),
    }


def _get_fixedbest_for_group(
    group: pd.DataFrame,
    best_combo_vals,
    target_col: str,
) -> float:
    mask = np.ones(len(group), dtype=bool)
    for col, val in zip(PARAM_COLS, best_combo_vals):
        mask &= np.isclose(group[col].values, val)
    if mask.sum() == 0:
        return float("nan")
    return float(group.loc[mask, target_col].iloc[0])


def summarize_per_image(
    per_image: pd.DataFrame,
    k_values: List[int],
    model_selection: str = "not available (loaded from per-image CSV)",
    model_selection_counts: Dict[str, int] = None,
    fold_metrics: List[Dict] = None,
) -> Dict:
    methods = [
        "nopre_cer",
        "fixedbest_actual_cer",
        "random1_actual_cer",
        "top1_actual_cer",
        "samporacle_cer",
    ]
    for k in k_values:
        col = f"bestof{k}_actual_cer"
        if col in per_image.columns:
            methods.append(col)

    means = {m: _mean(per_image[m].tolist()) for m in methods}
    tests = {
        "top1_vs_random1": _safe_wilcoxon(
            per_image["top1_actual_cer"].tolist(),
            per_image["random1_actual_cer"].tolist(),
            alternative="less",
        ),
        "top1_vs_fixedbest": _safe_wilcoxon(
            per_image["top1_actual_cer"].tolist(),
            per_image["fixedbest_actual_cer"].tolist(),
            alternative="less",
        ),
        "fixedbest_vs_top1": _safe_wilcoxon(
            per_image["fixedbest_actual_cer"].tolist(),
            per_image["top1_actual_cer"].tolist(),
            alternative="less",
        ),
        "top1_vs_fixedbest_two_sided": _safe_wilcoxon(
            per_image["top1_actual_cer"].tolist(),
            per_image["fixedbest_actual_cer"].tolist(),
            alternative="two-sided",
        ),
        "top1_vs_nopre": _safe_wilcoxon(
            per_image["top1_actual_cer"].tolist(),
            per_image["nopre_cer"].tolist(),
            alternative="less",
        ),
        "nopre_vs_top1": _safe_wilcoxon(
            per_image["nopre_cer"].tolist(),
            per_image["top1_actual_cer"].tolist(),
            alternative="less",
        ),
    }

    pairwise = {
        "top1_minus_random1": _pairwise_stats(
            per_image["top1_actual_cer"].tolist(),
            per_image["random1_actual_cer"].tolist(),
        ),
        "top1_minus_fixedbest": _pairwise_stats(
            per_image["top1_actual_cer"].tolist(),
            per_image["fixedbest_actual_cer"].tolist(),
        ),
        "top1_minus_nopre": _pairwise_stats(
            per_image["top1_actual_cer"].tolist(),
            per_image["nopre_cer"].tolist(),
        ),
    }

    gaps = {
        "top1_minus_samporacle": _mean(per_image["top1_minus_samporacle"].tolist()),
    }
    for k in k_values:
        col = f"top1_minus_bestof{k}"
        if col in per_image.columns:
            gaps[col] = _mean(per_image[col].tolist())

    return {
        "n_images": int(per_image["image_name"].nunique()),
        "fold_metrics": fold_metrics or [],
        "model_selection": model_selection,
        "model_selection_counts": model_selection_counts or {},
        "means": means,
        "wilcoxon": tests,
        "pairwise": pairwise,
        "gaps": gaps,
    }


def evaluate_engine(
    df: pd.DataFrame,
    engine: str,
    input_cols: List[str],
    exact_nopre_by_image: Dict[str, float],
    n_folds: int,
    k_values: List[int],
    random_trials: int,
    verbose: bool,
) -> Dict:
    target_col = f"cer_{engine}"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    models = get_models()
    gkf = GroupKFold(n_splits=n_folds)

    X_all = df[input_cols].values
    y_all = df[target_col].values
    groups = df[GROUP_COL].values

    fold_metrics = []
    selected_models = []
    per_image_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        if verbose:
            print(f"  Fold {fold_idx}/{n_folds}: selecting model")
        best_model_name = select_best_model(train_df, target_col, input_cols, verbose=False)
        selected_models.append(best_model_name)

        model = clone(models[best_model_name])
        model.fit(train_df[input_cols].values, train_df[target_col].values)
        y_pred = model.predict(test_df[input_cols].values)
        y_true = test_df[target_col].values

        spearman = stats.spearmanr(y_true, y_pred).correlation
        fold_metrics.append({
            "fold": fold_idx,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
            "selected_model": best_model_name,
        })

        test_eval = test_df.copy()
        test_eval["pred_cer"] = y_pred

        combo_train_mean = train_df.groupby(PARAM_COLS)[target_col].mean()
        best_combo_vals = combo_train_mean.idxmin()

        for image_name, group in test_eval.groupby(GROUP_COL):
            if image_name not in exact_nopre_by_image:
                raise KeyError(f"Missing exact NoPre CER for image '{image_name}'")

            ranked = group.sort_values("pred_cer", ascending=True)
            top1 = ranked.iloc[0]

            row = {
                "engine": engine,
                "fold": fold_idx,
                "image_name": image_name,
                "top1_pred_cer": float(top1["pred_cer"]),
                "top1_actual_cer": float(top1[target_col]),
                "random1_actual_cer": float(np.mean([
                    group.sample(1, random_state=s)[target_col].iloc[0]
                    for s in range(random_trials)
                ])),
                "fixedbest_actual_cer": _get_fixedbest_for_group(group, best_combo_vals, target_col),
                "nopre_cer": float(exact_nopre_by_image[image_name]),
                "samporacle_cer": float(group[target_col].min()),
            }

            for param in PARAM_COLS:
                row[f"top1_{param}"] = top1[param]

            for k in k_values:
                top_k = ranked.head(min(k, len(ranked)))
                row[f"bestof{k}_actual_cer"] = float(top_k[target_col].min())
                row[f"top1_minus_bestof{k}"] = (
                    row["top1_actual_cer"] - row[f"bestof{k}_actual_cer"]
                )

            row["top1_minus_samporacle"] = row["top1_actual_cer"] - row["samporacle_cer"]
            per_image_rows.append(row)

    per_image = pd.DataFrame(per_image_rows)

    summary = summarize_per_image(
        per_image,
        k_values=k_values,
        model_selection=summarize_model_selection(selected_models),
        model_selection_counts=dict(Counter(selected_models)),
        fold_metrics=fold_metrics,
    )
    summary.update({
        "engine": engine,
        "per_image": per_image,
    })
    return summary


def write_report(results: Dict[str, Dict], output_dir: Path, k_values: List[int]) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("TOP-1 DEPLOYABLE SURROGATE EVALUATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Top1Pred: candidate with lowest out-of-fold predicted CER; no ground truth")
    lines.append("BestOfK: best actual CER within top-K predicted shortlist; upper bound")
    lines.append("Random1: one random candidate averaged over reproducible trials per image")
    lines.append("")

    for engine, result in results.items():
        means = result["means"]
        lines.append("-" * 80)
        lines.append(f"Engine: {engine}")
        lines.append(f"Images: {result['n_images']}")
        lines.append(f"Model selection: {result['model_selection']}")
        lines.append("")
        lines.append("Mean CER")
        lines.append(f"  NoPre       : {_format_pct(means['nopre_cer'])}")
        lines.append(f"  FixedBest   : {_format_pct(means['fixedbest_actual_cer'])}")
        lines.append(f"  Random1     : {_format_pct(means['random1_actual_cer'])}")
        lines.append(f"  Top1Pred    : {_format_pct(means['top1_actual_cer'])}")
        for k in k_values:
            lines.append(f"  BestOf{k:<2d}   : {_format_pct(means[f'bestof{k}_actual_cer'])}")
        lines.append(f"  SampOracle  : {_format_pct(means['samporacle_cer'])}")
        lines.append("")
        lines.append("Gaps")
        for k in k_values:
            gap_key = f"top1_minus_bestof{k}"
            if gap_key not in result["gaps"]:
                continue
            lines.append(
                f"  Top1Pred - BestOf{k:<2d}: "
                f"{result['gaps'][gap_key] * 100:.2f} pp"
            )
        lines.append(
            f"  Top1Pred - SampOracle: "
            f"{result['gaps']['top1_minus_samporacle'] * 100:.2f} pp"
        )
        lines.append("")
        lines.append("Wilcoxon one-sided tests (alternative: Top1Pred < baseline)")
        for name in ["top1_vs_random1", "top1_vs_fixedbest", "top1_vs_nopre"]:
            test = result["wilcoxon"][name]
            lines.append(
                f"  {name:18s}: p={test['p']:.6g}, "
                f"stat={test['stat']:.3f}, n={test['n']}"
            )
        lines.append("")
        lines.append("Reverse / two-sided checks for FixedBest")
        for name in ["fixedbest_vs_top1", "top1_vs_fixedbest_two_sided"]:
            test = result["wilcoxon"][name]
            lines.append(
                f"  {name:28s}: p={test['p']:.6g}, "
                f"stat={test['stat']:.3f}, n={test['n']}"
            )
        lines.append("")
        lines.append("Pairwise Top1Pred differences and win rates")
        lines.append("  (diff = Top1Pred CER - baseline CER; win = Top1Pred lower CER)")
        for name, stats_dict in result["pairwise"].items():
            baseline = name.replace("top1_minus_", "")
            lines.append(
                f"  vs {baseline:10s}: mean={stats_dict['mean_diff'] * 100:.2f} pp, "
                f"median={stats_dict['median_diff'] * 100:.2f} pp, "
                f"win={stats_dict['win_rate'] * 100:.1f}%, "
                f"tie={stats_dict['tie_rate'] * 100:.1f}%, "
                f"loss={stats_dict['loss_rate'] * 100:.1f}%"
            )
        lines.append("")

    report_path = output_dir / "top1_deployable_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_image_path:
        per_image_df = pd.read_csv(args.per_image_path)
        k_values = [
            k for k in args.k_values
            if f"bestof{k}_actual_cer" in per_image_df.columns
        ]
        metrics_path = output_dir / "top1_deployable_metrics.json"
        existing_results = {}
        if metrics_path.exists():
            existing_results = json.loads(metrics_path.read_text(encoding="utf-8"))

        results = {}
        for engine, group in per_image_df.groupby("engine"):
            existing = existing_results.get(engine, {})
            results[engine] = summarize_per_image(
                group.copy(),
                k_values=k_values,
                model_selection=existing.get(
                    "model_selection",
                    "not available (loaded from per-image CSV)",
                ),
                model_selection_counts=existing.get("model_selection_counts", {}),
                fold_metrics=existing.get("fold_metrics", []),
            )
            results[engine]["engine"] = engine

        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_report(results, output_dir, k_values)
        print(f"\nLoaded per-image results: {args.per_image_path}")
        print(f"Report: {output_dir / 'top1_deployable_report.txt'}")
        print(f"Metrics: {metrics_path}")
        return

    if not args.data_path or not args.nopre_path:
        raise ValueError("--data-path and --nopre-path are required unless --per-image-path is used")

    df = pd.read_csv(args.data_path)
    input_cols = prepare_data(df)

    if args.engines:
        engines = args.engines
    else:
        engines = sorted(c.replace("cer_", "") for c in df.columns if c.startswith("cer_"))

    exact_maps = load_exact_nopre_maps(Path(args.nopre_path), engines)

    results = {}
    all_per_image = []
    for engine in engines:
        print(f"\nEvaluating {engine}")
        result = evaluate_engine(
            df=df,
            engine=engine,
            input_cols=input_cols,
            exact_nopre_by_image=exact_maps[engine],
            n_folds=args.n_folds,
            k_values=args.k_values,
            random_trials=args.random_trials,
            verbose=args.verbose,
        )
        results[engine] = {k: v for k, v in result.items() if k != "per_image"}
        all_per_image.append(result["per_image"])

    per_image_df = pd.concat(all_per_image, ignore_index=True)
    per_image_df.to_csv(output_dir / "top1_per_image.csv", index=False)

    selected_cols = ["engine", "image_name"] + [f"top1_{p}" for p in PARAM_COLS]
    per_image_df[selected_cols].to_csv(output_dir / "top1_selected_params.csv", index=False)

    metrics_path = output_dir / "top1_deployable_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, output_dir, args.k_values)

    print(f"\nReport: {output_dir / 'top1_deployable_report.txt'}")
    print(f"Per-image: {output_dir / 'top1_per_image.csv'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested GroupKFold deployable top-1 surrogate evaluation")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--nopre-path", default=None)
    parser.add_argument("--per-image-path", default=None,
                        help="Optional existing top1_per_image.csv to summarize without rerunning models")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10, 15])
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
