#!/usr/bin/env python3
"""
Analyze why a static FixedTopK shortlist may be competitive in some datasets
and less competitive in others.

The script is intended as a paper-supporting diagnostic, not as a training
pipeline. For each dataset it reports:
  - Feature variability from OCR-free page features.
  - How concentrated the per-page oracle-best preprocessing candidates are.
  - How well fold-specific FixedTopK shortlists approach the sampled-pool oracle.
  - Optional aggregate Surrogate-vs-FixedTopK results from metrics.json.

Example:
    python main/analyze_shortlist_heterogeneity.py \
      --dataset raw=outputs/multiengine_data_level1_fix_raw_g03757_nested/preprocessing_data.csv \
      --dataset contour=outputs_contour/multiengine_data_level1_fix_contour_g13757_nested/preprocessing_data.csv \
      --dataset sroie=SROIRE_COMPARISON/output_sroire/lhs_data_sroie_level1_fix_all626_g1/preprocessing_data.csv \
      --metrics raw=outputs/surrogate_multiengine_kfold_level1_fix_raw_g03757_nested_fixedtopk/metrics.json \
      --metrics contour=outputs_contour/surrogate_multiengine_kfold_level1_fix_contour_g13757_nested_fixedtopk/metrics.json \
      --metrics sroie=SROIRE_COMPARISON/output_sroire/surrogate_kfold_sroie_level1_fix_all626_g1_nested_fixedtopk/metrics.json \
      --output-dir outputs/shortlist_heterogeneity_analysis
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


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

DEFAULT_KS = [3, 5, 10, 15]


def parse_named_paths(items: Iterable[str]) -> Dict[str, Path]:
    paths = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty dataset name in: {item}")
        paths[name] = Path(path)
    return paths


def detect_engines(df: pd.DataFrame, requested: List[str] = None) -> List[str]:
    available = [c.replace("cer_", "") for c in df.columns if c.startswith("cer_")]
    if not available and "cer" in df.columns:
        df["cer_vietocr"] = df["cer"]
        available = ["vietocr"]
    if requested:
        return [engine for engine in requested if engine in available]
    return available


def add_combo_id(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PARAM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing preprocessing parameter columns: {missing}")

    out = df.copy()
    rounded = out[PARAM_COLS].copy()
    for col in rounded.columns:
        rounded[col] = rounded[col].map(lambda x: f"{float(x):.8g}")
    out["combo_id"] = rounded.agg("|".join, axis=1)
    return out


def entropy_from_counts(counts: np.ndarray) -> Tuple[float, float]:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0, 0.0
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return entropy, normalized


def page_feature_table(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in FEATURE_COLS_BASE + FEATURE_COLS_NEW if c in df.columns]
    if not feature_cols:
        return pd.DataFrame()

    # Page features are repeated for each candidate; keep one row per image.
    page_df = df[[GROUP_COL] + feature_cols].drop_duplicates(subset=[GROUP_COL])
    rows = []
    for col in feature_cols:
        vals = page_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        cv = float(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
        q25, q75 = vals.quantile([0.25, 0.75])
        rows.append({
            "dataset": dataset_name,
            "feature": col,
            "mean": mean,
            "std": std,
            "cv": cv,
            "iqr": float(q75 - q25),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    return pd.DataFrame(rows)


def oracle_distribution(
    dataset_name: str,
    df: pd.DataFrame,
    engine: str,
    ks: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"cer_{engine}"
    if target_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Stable tie handling: sort by target and combo_id, then take first per image.
    best_rows = (
        df[[GROUP_COL, "combo_id", target_col] + PARAM_COLS]
        .sort_values([GROUP_COL, target_col, "combo_id"])
        .groupby(GROUP_COL, as_index=False)
        .first()
    )
    counts = best_rows["combo_id"].value_counts()
    entropy, norm_entropy = entropy_from_counts(counts.values.astype(float))

    global_order = (
        df.groupby("combo_id")[target_col]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    rows = []
    for k in ks:
        topk_global = set(global_order[:k])
        topk_freq = set(counts.index[:k])
        rows.append({
            "dataset": dataset_name,
            "engine": engine,
            "k": k,
            "global_topk_covers_oracle_best_rate": float(best_rows["combo_id"].isin(topk_global).mean()),
            "freq_topk_covers_oracle_best_rate": float(best_rows["combo_id"].isin(topk_freq).mean()),
        })

    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "engine": engine,
        "n_images": int(best_rows[GROUP_COL].nunique()),
        "n_unique_oracle_best_combos": int(counts.size),
        "oracle_best_entropy_bits": entropy,
        "oracle_best_entropy_normalized": norm_entropy,
        "top1_oracle_best_share": float(counts.iloc[0] / counts.sum()) if len(counts) else float("nan"),
        "top3_oracle_best_share": float(counts.iloc[:3].sum() / counts.sum()) if len(counts) else float("nan"),
        "top5_oracle_best_share": float(counts.iloc[:5].sum() / counts.sum()) if len(counts) else float("nan"),
        "top10_oracle_best_share": float(counts.iloc[:10].sum() / counts.sum()) if len(counts) else float("nan"),
    }])
    coverage = pd.DataFrame(rows)
    return summary, coverage


def fixed_topk_gap(
    dataset_name: str,
    df: pd.DataFrame,
    engine: str,
    ks: List[int],
    n_folds: int,
) -> pd.DataFrame:
    target_col = f"cer_{engine}"
    if target_col not in df.columns:
        return pd.DataFrame()

    groups = df[GROUP_COL].values
    dummy_y = df[target_col].values
    splitter = GroupKFold(n_splits=n_folds)

    pooled_oracle = {}
    pooled_fixed_topk = {k: {} for k in ks}

    for train_idx, test_idx in splitter.split(df, dummy_y, groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_combo_mean = train_df.groupby("combo_id")[target_col].mean().sort_values()
        for img_name, group in test_df.groupby(GROUP_COL):
            pooled_oracle[img_name] = float(group[target_col].min())

        for k in ks:
            selected_combos = set(train_combo_mean.index[:k])
            matched = test_df[test_df["combo_id"].isin(selected_combos)]
            for img_name, group in matched.groupby(GROUP_COL):
                pooled_fixed_topk[k][img_name] = float(group[target_col].min())

    rows = []
    image_names = sorted(pooled_oracle)
    oracle = np.array([pooled_oracle[i] for i in image_names])
    for k in ks:
        missing = sorted(set(image_names) - set(pooled_fixed_topk[k]))
        if missing:
            raise ValueError(
                f"{dataset_name}/{engine}: FixedTopK K={k} missing "
                f"{len(missing)} images; first few: {missing[:3]}"
            )
        fixed = np.array([pooled_fixed_topk[k][i] for i in image_names])
        rows.append({
            "dataset": dataset_name,
            "engine": engine,
            "k": k,
            "oracle_mean": float(oracle.mean()),
            "fixed_topk_mean": float(fixed.mean()),
            "fixed_topk_minus_oracle": float((fixed - oracle).mean()),
            "fixed_topk_gap_relative_to_oracle": float((fixed.mean() - oracle.mean()) / oracle.mean()) if oracle.mean() else float("nan"),
        })
    return pd.DataFrame(rows)


def load_metrics_summary(dataset_name: str, metrics_path: Path) -> pd.DataFrame:
    if not metrics_path or not metrics_path.exists():
        return pd.DataFrame()
    with open(metrics_path, "r") as fp:
        metrics = json.load(fp)
    rows = []
    for engine, result in metrics.items():
        for k_str, fixed_val in result.get("fixed_topk_results", {}).items():
            k = int(k_str)
            surr = float(result["filter_results"][k_str])
            rand = float(result["random_results"][k_str])
            wil = result.get("wilcoxon", {}).get(k_str, {})
            rows.append({
                "dataset": dataset_name,
                "engine": engine,
                "k": k,
                "surrogate_mean": surr,
                "random_mean": rand,
                "fixed_topk_mean": float(fixed_val),
                "surrogate_minus_random": surr - rand,
                "surrogate_minus_fixed_topk": surr - float(fixed_val),
                "p_vs_random": float(wil.get("p_value", float("nan"))),
                "p_vs_fixed_topk": float(wil.get("p_vs_fixed_topk", float("nan"))),
            })
    return pd.DataFrame(rows)


def format_markdown_value(value, floatfmt: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), floatfmt)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a small dataframe as a GitHub-style markdown table without tabulate."""
    if df.empty:
        return ""

    display = df.copy()
    display.columns = [str(col) for col in display.columns]
    rows = [
        [format_markdown_value(value, floatfmt) for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    headers = list(display.columns)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def render_row(values: List[str]) -> str:
        return "| " + " | ".join(
            values[i].ljust(widths[i]) for i in range(len(values))
        ) + " |"

    header = render_row(headers)
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator] + body)


def write_report(
    output_dir: Path,
    feature_summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    oracle_coverage: pd.DataFrame,
    fixed_gap: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> None:
    report_path = output_dir / "report.md"
    with open(report_path, "w") as fp:
        fp.write("# Shortlist Heterogeneity Analysis\n\n")

        if not oracle_summary.empty:
            fp.write("## Oracle-Best Candidate Concentration\n\n")
            cols = [
                "dataset",
                "engine",
                "n_unique_oracle_best_combos",
                "oracle_best_entropy_normalized",
                "top3_oracle_best_share",
                "top5_oracle_best_share",
                "top10_oracle_best_share",
            ]
            fp.write(dataframe_to_markdown(oracle_summary[cols], floatfmt=".4f"))
            fp.write("\n\n")

        if not oracle_coverage.empty:
            fp.write("## Global Top-K Coverage of Per-Page Oracle Best\n\n")
            pivot = oracle_coverage.pivot_table(
                index=["dataset", "engine"],
                columns="k",
                values="global_topk_covers_oracle_best_rate",
            ).reset_index()
            fp.write(dataframe_to_markdown(pivot, floatfmt=".4f"))
            fp.write("\n\n")

        if not fixed_gap.empty:
            fp.write("## FixedTopK Gap to Sampled Oracle\n\n")
            display = fixed_gap.copy()
            display["oracle_mean_pct"] = display["oracle_mean"] * 100
            display["fixed_topk_mean_pct"] = display["fixed_topk_mean"] * 100
            display["fixed_topk_minus_oracle_pp"] = display["fixed_topk_minus_oracle"] * 100
            cols = [
                "dataset",
                "engine",
                "k",
                "oracle_mean_pct",
                "fixed_topk_mean_pct",
                "fixed_topk_minus_oracle_pp",
            ]
            fp.write(dataframe_to_markdown(display[cols], floatfmt=".3f"))
            fp.write("\n\n")

        if not metrics_summary.empty:
            fp.write("## Surrogate vs FixedTopK From Metrics\n\n")
            display = metrics_summary.copy()
            display["surrogate_mean_pct"] = display["surrogate_mean"] * 100
            display["fixed_topk_mean_pct"] = display["fixed_topk_mean"] * 100
            display["surrogate_minus_fixed_topk_pp"] = display["surrogate_minus_fixed_topk"] * 100
            cols = [
                "dataset",
                "engine",
                "k",
                "surrogate_mean_pct",
                "fixed_topk_mean_pct",
                "surrogate_minus_fixed_topk_pp",
                "p_vs_fixed_topk",
            ]
            fp.write(dataframe_to_markdown(display[cols], floatfmt=".4f"))
            fp.write("\n\n")

        if not feature_summary.empty:
            fp.write("## Page-Feature Variability Summary\n\n")
            cv_summary = (
                feature_summary.groupby("dataset")["cv"]
                .agg(["mean", "median", "max"])
                .reset_index()
                .rename(columns={"mean": "mean_feature_cv", "median": "median_feature_cv", "max": "max_feature_cv"})
            )
            fp.write(dataframe_to_markdown(cv_summary, floatfmt=".4f"))
            fp.write("\n\n")

    print(f"Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze preprocessing-shortlist heterogeneity and FixedTopK strength."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset CSV as NAME=PATH. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Optional metrics JSON as NAME=PATH. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_KS)
    args = parser.parse_args()

    dataset_paths = parse_named_paths(args.dataset)
    metrics_paths = parse_named_paths(args.metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tables = []
    oracle_summaries = []
    oracle_coverages = []
    fixed_gaps = []
    metrics_summaries = []

    for name, csv_path in dataset_paths.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        print(f"Loading {name}: {csv_path}")
        df = add_combo_id(pd.read_csv(csv_path))
        engines = detect_engines(df, args.engines)
        print(f"  rows={len(df):,}, images={df[GROUP_COL].nunique()}, engines={engines}")

        feature_tables.append(page_feature_table(name, df))

        for engine in engines:
            summary, coverage = oracle_distribution(name, df, engine, args.k_values)
            oracle_summaries.append(summary)
            oracle_coverages.append(coverage)
            fixed_gaps.append(fixed_topk_gap(name, df, engine, args.k_values, args.n_folds))

        if name in metrics_paths:
            metrics_summaries.append(load_metrics_summary(name, metrics_paths[name]))

    feature_summary = pd.concat([x for x in feature_tables if not x.empty], ignore_index=True) if feature_tables else pd.DataFrame()
    oracle_summary = pd.concat([x for x in oracle_summaries if not x.empty], ignore_index=True) if oracle_summaries else pd.DataFrame()
    oracle_coverage = pd.concat([x for x in oracle_coverages if not x.empty], ignore_index=True) if oracle_coverages else pd.DataFrame()
    fixed_gap = pd.concat([x for x in fixed_gaps if not x.empty], ignore_index=True) if fixed_gaps else pd.DataFrame()
    metrics_summary = pd.concat([x for x in metrics_summaries if not x.empty], ignore_index=True) if metrics_summaries else pd.DataFrame()

    outputs = {
        "feature_variability.csv": feature_summary,
        "oracle_best_distribution.csv": oracle_summary,
        "oracle_best_topk_coverage.csv": oracle_coverage,
        "fixedtopk_gap_to_oracle.csv": fixed_gap,
        "surrogate_vs_fixedtopk_from_metrics.csv": metrics_summary,
    }
    for filename, table in outputs.items():
        if not table.empty:
            path = output_dir / filename
            table.to_csv(path, index=False)
            print(f"Wrote {path}")

    write_report(output_dir, feature_summary, oracle_summary, oracle_coverage, fixed_gap, metrics_summary)


if __name__ == "__main__":
    main()
