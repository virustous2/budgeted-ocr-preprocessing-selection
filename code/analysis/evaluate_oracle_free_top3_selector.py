#!/usr/bin/env python3
"""
Exploratory oracle-free selector for surrogate top-3 OCR outputs.

The selector chooses one OCR output from the surrogate-ranked top-3 candidates
without using reference text at selection time. Candidate labels and CER values
are used only for supervised training on training folds and for final evaluation.

Inputs expected from the existing raw top-3 consensus run:
  - top3_consensus_ocr_cache.csv: OCR text for the top-3 candidates
  - top3_consensus_per_image.csv: fold IDs and deployable baselines
  - preprocessing_data.csv: actual candidate CER labels for evaluation/training
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rapidfuzz.distance import Levenshtein
except Exception:  # pragma: no cover
    Levenshtein = None


PARAM_COLS = ["gaussian_ksize", "bilateral_d", "clahe_clip", "gamma", "sharpen"]
K = 3

ADMIN_KEYWORDS = [
    "căn cứ",
    "điều",
    "nghị định",
    "quyết định",
    "thông tư",
    "cộng hòa",
    "độc lập",
    "hạnh phúc",
    "chính phủ",
    "quốc hội",
    "ủy ban",
    "uỷ ban",
    "tòa án",
    "toà án",
    "bộ trưởng",
    "ngày",
    "tháng",
    "năm",
]


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt_pp(x: float) -> str:
    return f"{100.0 * x:+.2f}pp"


def norm_text(text: object) -> str:
    return " ".join(str(text or "").split()).lower()


def edit_distance(a: str, b: str) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(a, b))
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    na = norm_text(a)
    nb = norm_text(b)
    denom = max(len(na), len(nb), 1)
    return edit_distance(na, nb) / denom


def paired_wilcoxon_two_sided(a: Sequence[float], b: Sequence[float]) -> float:
    """Return a two-sided paired Wilcoxon p-value."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if not np.any(arr_a != arr_b):
        return 1.0
    return float(wilcoxon(arr_a, arr_b, alternative="two-sided").pvalue)


def make_classifier(kind: str, seed: int):
    if kind == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
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
    if kind == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown classifier: {kind}")


def char_features(text: str) -> Dict[str, float]:
    raw = str(text or "")
    compact = norm_text(raw)
    n = max(len(compact), 1)
    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    spaces = sum(ch.isspace() for ch in compact)
    punct = sum(unicodedata.category(ch).startswith("P") for ch in compact)
    upper = sum(ch.isupper() for ch in raw)
    non_ascii = sum(ord(ch) > 127 for ch in compact)
    vietnamese_marks = sum(ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for ch in compact)
    weird = sum(ch in "�□■●¤¬" for ch in compact)
    keyword_count = sum(compact.count(kw) for kw in ADMIN_KEYWORDS)
    date_like = len(re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", compact))
    law_like = len(re.findall(r"\b\d+/\d{4}/[A-Za-zÀ-ỹ0-9-]+\b", compact))
    repeated = len(re.findall(r"(.)\1{3,}", compact))

    return {
        "text_len": float(len(compact)),
        "letter_ratio": letters / n,
        "digit_ratio": digits / n,
        "space_ratio": spaces / n,
        "punct_ratio": punct / n,
        "upper_ratio": upper / max(len(raw), 1),
        "non_ascii_ratio": non_ascii / n,
        "vietnamese_mark_ratio": vietnamese_marks / n,
        "weird_ratio": weird / n,
        "admin_keyword_count": float(keyword_count),
        "date_like_count": float(date_like),
        "law_like_count": float(law_like),
        "repeated_run_count": float(repeated),
    }


def rounded_params(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gaussian_ksize"] = out["gaussian_ksize"].astype(int)
    out["bilateral_d"] = out["bilateral_d"].astype(int)
    for col in ["clahe_clip", "gamma", "sharpen"]:
        out[col] = out[col].astype(float).round(10)
    return out


def load_candidate_rows(
    cache_path: Path,
    data_path: Path,
    per_image_path: Path,
    engines: List[str],
) -> pd.DataFrame:
    cache = rounded_params(pd.read_csv(cache_path, keep_default_na=False))
    cache["_rank"] = cache.groupby("image_name").cumcount() + 1
    cache = cache[cache["_rank"] <= K].copy()

    data = rounded_params(pd.read_csv(data_path))
    join_cols = ["image_name"] + PARAM_COLS
    keep_cols = join_cols + [f"cer_{engine}" for engine in engines]
    data = data[keep_cols].copy()

    merged = cache.merge(data, on=join_cols, how="left", validate="many_to_one")
    per_image = pd.read_csv(per_image_path)

    rows = []
    for engine in engines:
        engine_per = per_image[per_image["engine"] == engine].copy()
        engine_cols = [
            "fold",
            "image_name",
            "top1_pred_cer",
            "top1_actual_cer",
            "random1_actual_cer",
            "fixedbest_actual_cer",
            "nopre_cer",
            "samporacle_cer",
            "bestof3_actual_cer",
            "top3_consensus_actual_cer",
            "top3_consensus_selected_rank",
        ]
        engine_cols = [c for c in engine_cols if c in engine_per.columns]
        temp = merged.merge(engine_per[engine_cols], on="image_name", how="inner", validate="many_to_one")
        temp["engine"] = engine
        temp["ocr_text"] = temp[f"text_{engine}"].fillna("").astype(str)
        temp["n_chars"] = temp.get(f"n_chars_{engine}", temp["ocr_text"].str.len()).astype(float)
        temp["actual_cer"] = temp[f"cer_{engine}"].astype(float)
        temp["rank"] = temp["_rank"].astype(int)
        rows.append(temp)

    out = pd.concat(rows, ignore_index=True)
    if out["actual_cer"].isna().any():
        missing = int(out["actual_cer"].isna().sum())
        raise ValueError(f"Missing actual CER for {missing} candidate rows")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (engine, image_name), group in df.groupby(["engine", "image_name"], sort=False):
        group = group.sort_values("rank").copy()
        texts = group["ocr_text"].tolist()
        lengths = np.array([len(norm_text(t)) for t in texts], dtype=float)
        len_median = float(np.median(lengths)) if len(lengths) else 0.0
        len_mean = float(np.mean(lengths)) if len(lengths) else 0.0
        len_std = float(np.std(lengths)) if len(lengths) else 0.0
        dist_matrix = np.zeros((len(group), len(group)), dtype=float)
        for i in range(len(group)):
            for j in range(len(group)):
                if i != j:
                    dist_matrix[i, j] = normalized_edit_distance(texts[i], texts[j])
        for idx, (_, row) in enumerate(group.iterrows()):
            rec = row.to_dict()
            dists = [dist_matrix[idx, j] for j in range(len(group)) if j != idx]
            rec.update(char_features(row["ocr_text"]))
            rec["mean_dist_to_others"] = float(np.mean(dists)) if dists else 0.0
            rec["min_dist_to_others"] = float(np.min(dists)) if dists else 0.0
            rec["max_dist_to_others"] = float(np.max(dists)) if dists else 0.0
            rec["text_len_minus_median"] = float(lengths[idx] - len_median)
            rec["text_len_minus_mean"] = float(lengths[idx] - len_mean)
            rec["text_len_z"] = float((lengths[idx] - len_mean) / (len_std + 1e-8))
            rec["detected_crops_minus_median"] = float(
                row["n_detected_crops"] - group["n_detected_crops"].median()
            )
            rec["is_rank1"] = float(row["rank"] == 1)
            rec["is_rank2"] = float(row["rank"] == 2)
            rec["is_rank3"] = float(row["rank"] == 3)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["best_in_top3"] = (
        out["actual_cer"]
        <= out.groupby(["engine", "image_name"])["actual_cer"].transform("min") + 1e-12
    ).astype(int)
    return out


def feature_cols(df: pd.DataFrame) -> List[str]:
    base = [
        "rank",
        "n_detected_crops",
        "n_chars",
        "gaussian_ksize",
        "bilateral_d",
        "clahe_clip",
        "gamma",
        "sharpen",
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
    return [c for c in base if c in df.columns]


def evaluate_engine(df: pd.DataFrame, engine: str, classifier: str, seed: int) -> tuple[pd.DataFrame, Dict]:
    data = df[df["engine"] == engine].copy().reset_index(drop=True)
    feats = feature_cols(data)
    data["selector_score"] = np.nan
    data["selected_by_model"] = False

    fold_details = []
    for fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != fold].copy()
        test = data[data["fold"] == fold].copy()
        model = make_classifier(classifier, seed)
        y_train = train["best_in_top3"].values
        if len(set(y_train)) < 2:
            # Degenerate fallback: rank 1.
            scores = -test["rank"].astype(float).values
        else:
            model.fit(train[feats], y_train)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(test[feats])[:, 1]
            else:
                scores = model.decision_function(test[feats])
        data.loc[test.index, "selector_score"] = scores
        fold_details.append(
            {
                "fold": int(fold),
                "train_positive_rate": float(np.mean(y_train)),
                "test_images": int(test["image_name"].nunique()),
            }
        )

    selected_rows = []
    for (_engine, image_name), group in data.groupby(["engine", "image_name"], sort=False):
        chosen_idx = group.sort_values(
            ["selector_score", "rank"],
            ascending=[False, True],
        ).index[0]
        data.loc[chosen_idx, "selected_by_model"] = True
        selected_rows.append(data.loc[chosen_idx].to_dict())
    selected = pd.DataFrame(selected_rows)

    # One row per page baselines from selected dataframe.
    selector = selected["actual_cer"].values
    top1 = selected["top1_actual_cer"].values
    fixed = selected["fixedbest_actual_cer"].values
    nopre = selected["nopre_cer"].values
    bestof3 = selected["bestof3_actual_cer"].values
    consensus = selected["top3_consensus_actual_cer"].values
    random1 = selected["random1_actual_cer"].values

    labels = data.groupby("image_name")["best_in_top3"].max()
    selected_is_best = selected["best_in_top3"].values
    summary = {
        "engine": engine,
        "classifier": classifier,
        "n_images": int(len(selected)),
        "feature_count": int(len(feats)),
        "selected_rank_counts": dict(Counter(selected["rank"].astype(int).astype(str))),
        "selector_mean": float(np.mean(selector)),
        "top1_mean": float(np.mean(top1)),
        "fixedbest_mean": float(np.mean(fixed)),
        "nopre_mean": float(np.mean(nopre)),
        "bestof3_mean": float(np.mean(bestof3)),
        "consensus_mean": float(np.mean(consensus)),
        "random1_mean": float(np.mean(random1)),
        "selector_minus_top1": float(np.mean(selector - top1)),
        "selector_minus_fixedbest": float(np.mean(selector - fixed)),
        "selector_minus_nopre": float(np.mean(selector - nopre)),
        "selector_minus_consensus": float(np.mean(selector - consensus)),
        "selector_minus_bestof3": float(np.mean(selector - bestof3)),
        "p_selector_vs_top1": paired_wilcoxon_two_sided(selector, top1),
        "p_selector_vs_fixedbest": paired_wilcoxon_two_sided(selector, fixed),
        "p_selector_vs_nopre": paired_wilcoxon_two_sided(selector, nopre),
        "p_selector_vs_consensus": paired_wilcoxon_two_sided(selector, consensus),
        "p_top1_vs_selector": paired_wilcoxon_two_sided(top1, selector),
        "selected_best_accuracy": float(np.mean(selected_is_best)),
        "candidate_balanced_accuracy": float(balanced_accuracy_score(data["best_in_top3"], data["selector_score"] >= 0.5)),
        "fold_details": fold_details,
    }
    return data, summary


def write_report(summaries: List[Dict], output_dir: Path) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("ORACLE-FREE TOP-3 OUTPUT SELECTOR EXPLORATORY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Selector chooses one OCR output from surrogate top-3 using text/rank/agreement features.")
    lines.append("Reference text/CER are used only to train labels on training folds and to evaluate test folds.")
    lines.append("")
    header = (
        "Engine      Selector  Top1Pred  FixedBest  NoPre    Consensus  BestOf3  "
        "Sel-Top1  Sel-Fixed  pSelTop1 pSelFixed RankCounts"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{s['engine']:<11s} "
            f"{fmt_pct(s['selector_mean']):>8s} "
            f"{fmt_pct(s['top1_mean']):>9s} "
            f"{fmt_pct(s['fixedbest_mean']):>9s} "
            f"{fmt_pct(s['nopre_mean']):>7s} "
            f"{fmt_pct(s['consensus_mean']):>10s} "
            f"{fmt_pct(s['bestof3_mean']):>8s} "
            f"{fmt_pp(s['selector_minus_top1']):>8s} "
            f"{fmt_pp(s['selector_minus_fixedbest']):>9s} "
            f"{s['p_selector_vs_top1']:>7.3g} "
            f"{s['p_selector_vs_fixedbest']:>8.3g} "
            f"{s['selected_rank_counts']}"
        )
    lines.append("")
    lines.append("Additional two-sided paired p-values")
    lines.append("Engine      pSelNoPre  pSelConsensus pTop1Selector GapToBestOf3")
    lines.append("-" * 66)
    for s in summaries:
        lines.append(
            f"{s['engine']:<11s} "
            f"{s['p_selector_vs_nopre']:>9.3g} "
            f"{s['p_selector_vs_consensus']:>12.3g} "
            f"{s['p_top1_vs_selector']:>14.3g} "
            f"{fmt_pp(s['selector_minus_bestof3']):>12s}"
        )
    (output_dir / "selector_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--per-image-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engines", default="vietocr,tesseract,easyocr")
    parser.add_argument("--classifier", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_rows(
        cache_path=args.cache_path,
        data_path=args.data_path,
        per_image_path=args.per_image_path,
        engines=engines,
    )
    candidates = add_features(candidates)

    all_candidate_rows = []
    summaries = []
    for engine in engines:
        scored, summary = evaluate_engine(candidates, engine, args.classifier, args.seed)
        all_candidate_rows.append(scored)
        summaries.append(summary)
        print(
            f"{engine}: selector={fmt_pct(summary['selector_mean'])}, "
            f"top1={fmt_pct(summary['top1_mean'])}, fixed={fmt_pct(summary['fixedbest_mean'])}, "
            f"bestof3={fmt_pct(summary['bestof3_mean'])}",
            flush=True,
        )

    pd.concat(all_candidate_rows, ignore_index=True).to_csv(
        args.output_dir / "selector_candidate_scores.csv",
        index=False,
    )
    pd.DataFrame(summaries).to_csv(args.output_dir / "selector_summary.csv", index=False)
    (args.output_dir / "selector_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summaries, args.output_dir)
    print(f"\nWrote: {args.output_dir / 'selector_report.txt'}")


if __name__ == "__main__":
    main()
