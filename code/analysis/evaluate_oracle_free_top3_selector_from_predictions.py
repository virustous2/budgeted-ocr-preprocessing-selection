#!/usr/bin/env python3
"""
Oracle-free selector for surrogate-ranked top-3 OCR outputs.

This version reconstructs each recognizer's top-3 shortlist from out-of-fold
candidate-level surrogate predictions, then looks up cached OCR text for those
exact preprocessing candidates. It avoids relying on the first three rows of a
shared OCR cache, which may contain a union of top-3 candidates from multiple
recognizers.

The selector is trained only on training folds. Reference CER is used to create
training labels and for final evaluation, not for test-time selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_oracle_free_top3_selector import (  # type: ignore
    PARAM_COLS,
    char_features,
    fmt_pct,
    fmt_pp,
    norm_text,
    normalized_edit_distance,
    paired_wilcoxon_two_sided,
)


GROUP_COL = "image_name"
K = 3


ComboKey = Tuple[str, int, int, float, float, float]


def parse_engines(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def round_params(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gaussian_ksize"] = out["gaussian_ksize"].astype(int)
    out["bilateral_d"] = out["bilateral_d"].astype(int)
    for col in ["clahe_clip", "gamma", "sharpen"]:
        out[col] = out[col].astype(float).round(10)
    return out


def make_key(row: pd.Series) -> ComboKey:
    return (
        str(row[GROUP_COL]),
        int(row["gaussian_ksize"]),
        int(row["bilateral_d"]),
        round(float(row["clahe_clip"]), 10),
        round(float(row["gamma"]), 10),
        round(float(row["sharpen"]), 10),
    )


def load_cache(cache_path: Path) -> Dict[ComboKey, Dict[str, object]]:
    cache_df = round_params(pd.read_csv(cache_path, keep_default_na=False))
    return {make_key(row): row.to_dict() for _, row in cache_df.iterrows()}


def make_estimator(kind: str, seed: int):
    if kind == "rf_cls":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "logreg_cls":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                ),
            ]
        )
    if kind == "rf_reg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "gbr_reg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=8,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "ridge_reg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    raise ValueError(f"Unknown selector kind: {kind}")


def is_classifier(kind: str) -> bool:
    return kind.endswith("_cls")


def candidate_feature_columns(frame: pd.DataFrame) -> List[str]:
    cols = [
        "rank",
        "pred_cer",
        "pred_delta_best",
        "pred_delta_prev",
        "pred_delta_next",
        "pred_rank_norm",
        "n_detected_crops",
        "n_chars",
        *PARAM_COLS,
        "text_len",
        "letter_ratio",
        "digit_ratio",
        "space_ratio",
        "punct_ratio",
        "upper_ratio",
        "non_ascii_ratio",
        "vietnamese_mark_ratio",
        "weird_ratio",
        "admin_keyword_count",
        "date_like_count",
        "law_like_count",
        "repeated_run_count",
        "mean_dist_to_others",
        "min_dist_to_others",
        "max_dist_to_others",
        "text_len_minus_median",
        "text_len_minus_mean",
        "text_len_z",
        "detected_crops_minus_median",
        "is_rank1",
        "is_rank2",
        "is_rank3",
    ]
    return [col for col in cols if col in frame.columns]


def build_candidates(
    candidate_predictions_path: Path,
    cache_path: Path,
    per_image_path: Path,
    engines: List[str],
) -> pd.DataFrame:
    pred = round_params(pd.read_csv(candidate_predictions_path))
    cache = load_cache(cache_path)
    per_image = pd.read_csv(per_image_path)

    rows = []
    missing_cache = 0
    for engine in engines:
        engine_pred = pred[pred["engine"].astype(str) == engine].copy()
        if engine_pred.empty:
            raise ValueError(f"No candidate predictions for engine {engine}")
        engine_per = per_image[per_image["engine"].astype(str) == engine].copy()
        per_cols = [
            "engine",
            "fold",
            GROUP_COL,
            "fixedbest_actual_cer",
            "nopre_cer",
            "samporacle_cer",
            "top3_consensus_actual_cer",
        ]
        per_cols = [col for col in per_cols if col in engine_per.columns]
        per_lookup = engine_per[per_cols].drop_duplicates(subset=[GROUP_COL])

        for image_name, group in engine_pred.groupby(GROUP_COL, sort=False):
            top = group.sort_values("pred_cer", kind="mergesort").head(K).copy()
            if len(top) < K:
                continue
            top = top.reset_index(drop=True)
            pred_values = top["pred_cer"].astype(float).values
            texts = []
            cache_records = []
            for _, row in top.iterrows():
                cached = cache.get(make_key(row))
                if cached is None:
                    missing_cache += 1
                    cached = {}
                cache_records.append(cached)
                texts.append(str(cached.get(f"text_{engine}", "") or ""))

            lengths = np.array([len(norm_text(text)) for text in texts], dtype=float)
            len_median = float(np.median(lengths))
            len_mean = float(np.mean(lengths))
            len_std = float(np.std(lengths))
            dist_matrix = np.zeros((len(top), len(top)), dtype=float)
            for i in range(len(top)):
                for j in range(len(top)):
                    if i != j:
                        dist_matrix[i, j] = normalized_edit_distance(texts[i], texts[j])

            for idx, (_, row) in enumerate(top.iterrows()):
                cached = cache_records[idx]
                dists = [dist_matrix[idx, j] for j in range(len(top)) if j != idx]
                rec = {
                    "engine": engine,
                    GROUP_COL: str(image_name),
                    "fold": int(row["outer_fold"]),
                    "rank": idx + 1,
                    "pred_cer": float(row["pred_cer"]),
                    "actual_cer": float(row["actual_cer"]),
                    "ocr_text": texts[idx],
                    "n_detected_crops": float(cached.get("n_detected_crops", 0) or 0),
                    "n_chars": float(cached.get(f"n_chars_{engine}", len(texts[idx])) or 0),
                }
                for param in PARAM_COLS:
                    rec[param] = row[param]
                rec.update(char_features(texts[idx]))
                rec["mean_dist_to_others"] = float(np.mean(dists)) if dists else 0.0
                rec["min_dist_to_others"] = float(np.min(dists)) if dists else 0.0
                rec["max_dist_to_others"] = float(np.max(dists)) if dists else 0.0
                rec["text_len_minus_median"] = float(lengths[idx] - len_median)
                rec["text_len_minus_mean"] = float(lengths[idx] - len_mean)
                rec["text_len_z"] = float((lengths[idx] - len_mean) / (len_std + 1e-8))
                rec["detected_crops_minus_median"] = float(
                    rec["n_detected_crops"]
                    - np.median([float(c.get("n_detected_crops", 0) or 0) for c in cache_records])
                )
                rec["pred_delta_best"] = float(pred_values[idx] - pred_values[0])
                rec["pred_delta_prev"] = float(pred_values[idx] - pred_values[idx - 1]) if idx > 0 else 0.0
                rec["pred_delta_next"] = float(pred_values[idx + 1] - pred_values[idx]) if idx < len(top) - 1 else 0.0
                rec["pred_rank_norm"] = float(idx / max(len(top) - 1, 1))
                rec["is_rank1"] = float(idx == 0)
                rec["is_rank2"] = float(idx == 1)
                rec["is_rank3"] = float(idx == 2)
                rows.append(rec)

    candidates = pd.DataFrame(rows)
    candidates["best_in_top3"] = (
        candidates["actual_cer"]
        <= candidates.groupby(["engine", GROUP_COL])["actual_cer"].transform("min") + 1e-12
    ).astype(int)

    # Agreement baseline on these exact reconstructed top-3 candidates.
    candidates["agreement_selected"] = False
    candidates["agreement_score"] = np.nan
    for (_engine, _image_name), group in candidates.groupby(["engine", GROUP_COL], sort=False):
        scores = group["mean_dist_to_others"].astype(float)
        chosen = scores.sort_values(kind="mergesort").index[0]
        candidates.loc[group.index, "agreement_score"] = scores.values
        candidates.loc[chosen, "agreement_selected"] = True

    # Add per-page baselines.
    candidates = candidates.merge(
        per_image[[col for col in [
            "engine",
            GROUP_COL,
            "fixedbest_actual_cer",
            "nopre_cer",
            "samporacle_cer",
            "top3_consensus_actual_cer",
        ] if col in per_image.columns]].drop_duplicates(subset=["engine", GROUP_COL]),
        on=["engine", GROUP_COL],
        how="left",
        validate="many_to_one",
    )
    candidates.attrs["missing_cache_rows"] = missing_cache
    return candidates


def select_by_model(data: pd.DataFrame, kind: str, seed: int) -> Tuple[pd.DataFrame, Dict]:
    feats = candidate_feature_columns(data)
    data = data.copy()
    data["selector_score"] = np.nan
    data["selected_by_model"] = False

    for fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != fold].copy()
        test = data[data["fold"] == fold].copy()
        model = make_estimator(kind, seed + int(fold))
        if is_classifier(kind):
            y_train = train["best_in_top3"].astype(int).values
            if len(set(y_train)) < 2:
                scores = -test["rank"].astype(float).values
            else:
                model.fit(train[feats], y_train)
                scores = model.predict_proba(test[feats])[:, 1]
        else:
            y_train = train["actual_cer"].astype(float).values
            model.fit(train[feats], y_train)
            scores = -model.predict(test[feats])
        data.loc[test.index, "selector_score"] = scores

    selected_rows = []
    for (_engine, _image_name), group in data.groupby(["engine", GROUP_COL], sort=False):
        chosen = group.sort_values(["selector_score", "rank"], ascending=[False, True]).index[0]
        data.loc[chosen, "selected_by_model"] = True
        selected_rows.append(data.loc[chosen].to_dict())
    selected = pd.DataFrame(selected_rows)
    summary = summarize_selection(selected, data, kind, len(feats))
    return data, summary


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr))


def summarize_selection(selected: pd.DataFrame, all_rows: pd.DataFrame, kind: str, n_features: int) -> Dict:
    top1 = (
        all_rows[all_rows["rank"] == 1]
        .sort_values(["engine", GROUP_COL])
        .set_index(["engine", GROUP_COL])["actual_cer"]
    )
    agreement = (
        all_rows[all_rows["agreement_selected"]]
        .sort_values(["engine", GROUP_COL])
        .set_index(["engine", GROUP_COL])["actual_cer"]
    )
    selected_index = selected.set_index(["engine", GROUP_COL])
    selector = selected_index["actual_cer"]
    bestof3 = all_rows.groupby(["engine", GROUP_COL])["actual_cer"].min()

    # Align all page-level series.
    idx = selector.index
    top1 = top1.loc[idx]
    agreement = agreement.loc[idx]
    bestof3 = bestof3.loc[idx]
    fixed = selected_index["fixedbest_actual_cer"].astype(float)
    nopre = selected_index["nopre_cer"].astype(float)

    return {
        "selector": kind,
        "n_images": int(len(selected)),
        "n_features": int(n_features),
        "selected_rank_counts": dict(Counter(selected["rank"].astype(int).astype(str))),
        "selector_mean": safe_mean(selector),
        "top1_mean": safe_mean(top1),
        "agreement_mean": safe_mean(agreement),
        "fixedbest_mean": safe_mean(fixed),
        "nopre_mean": safe_mean(nopre),
        "bestof3_mean": safe_mean(bestof3),
        "selector_minus_top1": safe_mean(selector - top1),
        "selector_minus_agreement": safe_mean(selector - agreement),
        "selector_minus_fixedbest": safe_mean(selector - fixed),
        "selector_minus_bestof3": safe_mean(selector - bestof3),
        "p_selector_vs_top1": paired_wilcoxon_two_sided(selector, top1),
        "p_top1_vs_selector": paired_wilcoxon_two_sided(top1, selector),
        "p_selector_vs_agreement": paired_wilcoxon_two_sided(selector, agreement),
        "p_agreement_vs_selector": paired_wilcoxon_two_sided(agreement, selector),
        "p_selector_vs_fixedbest": paired_wilcoxon_two_sided(selector, fixed),
        "p_selector_vs_nopre": paired_wilcoxon_two_sided(selector, nopre),
        "selected_best_accuracy": float(np.mean(selected["best_in_top3"].astype(int).values)),
    }


def summarize_by_engine(selected_all: pd.DataFrame, scored_all: pd.DataFrame, kind: str, n_features: int) -> List[Dict]:
    summaries = []
    for engine, selected in selected_all.groupby("engine", sort=True):
        scored = scored_all[scored_all["engine"] == engine].copy()
        summary = summarize_selection(selected, scored, kind, n_features)
        summary["engine"] = engine
        summaries.append(summary)
    return summaries


def format_report(summaries: Sequence[Dict], missing_cache_rows: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("ORACLE-FREE TOP-3 SELECTOR FROM CANDIDATE PREDICTIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Missing OCR cache rows: {missing_cache_rows}")
    lines.append("")
    header = (
        "Engine      Selector   Chosen   Top1   Agree  Fixed  NoPre  Best3  "
        "Sel-Top1 Sel-Agree Sel-Fixed pTop1 pAgree pFixed"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{s['engine']:<11s} "
            f"{s['selector']:<10s} "
            f"{fmt_pct(s['selector_mean']):>7s} "
            f"{fmt_pct(s['top1_mean']):>6s} "
            f"{fmt_pct(s['agreement_mean']):>6s} "
            f"{fmt_pct(s['fixedbest_mean']):>6s} "
            f"{fmt_pct(s['nopre_mean']):>6s} "
            f"{fmt_pct(s['bestof3_mean']):>6s} "
            f"{fmt_pp(s['selector_minus_top1']):>8s} "
            f"{fmt_pp(s['selector_minus_agreement']):>9s} "
            f"{fmt_pp(s['selector_minus_fixedbest']):>9s} "
            f"{s['p_selector_vs_top1']:.3g} "
            f"{s['p_selector_vs_agreement']:.3g} "
            f"{s['p_selector_vs_fixedbest']:.3g}"
        )
    lines.append("")
    lines.append("Rank counts")
    for s in summaries:
        lines.append(f"  {s['engine']} {s['selector']}: {s['selected_rank_counts']}")
    return "\n".join(lines) + "\n"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--per-image-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engines", default="vietocr,tesseract,easyocr")
    parser.add_argument(
        "--selectors",
        default="rf_cls,logreg_cls,rf_reg,gbr_reg,ridge_reg",
        help="Comma-separated selector kinds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engines = parse_engines(args.engines)
    selectors = parse_engines(args.selectors)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(
        candidate_predictions_path=args.candidate_predictions,
        cache_path=args.cache_path,
        per_image_path=args.per_image_path,
        engines=engines,
    )
    missing_cache_rows = int(candidates.attrs.get("missing_cache_rows", 0))

    all_summaries = []
    all_scored = []
    for selector in selectors:
        print(f"\nSelector: {selector}")
        scored_pieces = []
        selected_pieces = []
        for engine in engines:
            data = candidates[candidates["engine"] == engine].copy().reset_index(drop=True)
            scored, summary = select_by_model(data, selector, args.seed)
            summary["engine"] = engine
            scored_pieces.append(scored)
            selected_pieces.append(scored[scored["selected_by_model"]].copy())
            all_summaries.append(summary)
            print(
                f"  {engine}: selector={fmt_pct(summary['selector_mean'])}, "
                f"top1={fmt_pct(summary['top1_mean'])}, "
                f"agreement={fmt_pct(summary['agreement_mean'])}, "
                f"fixed={fmt_pct(summary['fixedbest_mean'])}, "
                f"best3={fmt_pct(summary['bestof3_mean'])}"
            )
        scored_all = pd.concat(scored_pieces, ignore_index=True)
        scored_all["selector_kind"] = selector
        all_scored.append(scored_all)

    summary_df = pd.DataFrame(all_summaries)
    scored_df = pd.concat(all_scored, ignore_index=True)
    summary_df.to_csv(args.output_dir / "selector_summary.csv", index=False)
    scored_df.to_csv(args.output_dir / "selector_candidate_scores.csv", index=False)
    (args.output_dir / "selector_summary.json").write_text(
        json.dumps(to_jsonable(all_summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "selector_report.txt").write_text(
        format_report(all_summaries, missing_cache_rows),
        encoding="utf-8",
    )
    print(f"\nWrote: {args.output_dir / 'selector_report.txt'}")


if __name__ == "__main__":
    main()
