#!/usr/bin/env python3
"""
Legacy pooled-OOF selector; not an end-to-end nested submission evaluation.

The upstream predictions used for selector training can depend on the selector
test fold. Use evaluate_top3_selector_strict_nested.py for corrected evaluation.

The exploratory selector showed that text/agreement features can choose a
deployable output from the top-3 shortlist. This script removes the main
exploratory bias: selector-family choice is made only inside each outer
training fold, then the selected family is refit on the outer training pages
and evaluated on held-out pages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluate_oracle_free_top3_selector import (  # type: ignore
    fmt_pct,
    fmt_pp,
    paired_wilcoxon_two_sided,
)
from evaluate_oracle_free_top3_selector_from_predictions import (  # type: ignore
    GROUP_COL,
    build_candidates,
    candidate_feature_columns,
    is_classifier,
    make_estimator,
    parse_engines,
)


DEFAULT_SELECTORS = ["rf_cls", "logreg_cls", "rf_reg", "gbr_reg", "ridge_reg"]


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr))


def train_scores(train: pd.DataFrame, test: pd.DataFrame, features: List[str], selector: str, seed: int) -> np.ndarray:
    model = make_estimator(selector, seed)
    if is_classifier(selector):
        y_train = train["best_in_top3"].astype(int).values
        if len(set(y_train)) < 2:
            return -test["rank"].astype(float).values
        model.fit(train[features], y_train)
        return model.predict_proba(test[features])[:, 1]

    y_train = train["actual_cer"].astype(float).values
    model.fit(train[features], y_train)
    return -model.predict(test[features])


def select_one_per_page(scored: pd.DataFrame, score_col: str = "selector_score") -> pd.DataFrame:
    rows = []
    for (_engine, _image_name), group in scored.groupby(["engine", GROUP_COL], sort=False):
        chosen = group.sort_values([score_col, "rank"], ascending=[False, True]).iloc[0]
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def validation_mean_cer(train: pd.DataFrame, selector: str, features: List[str], seed: int) -> float:
    fold_scores = []
    for inner_fold in sorted(train["fold"].unique()):
        inner_train = train[train["fold"] != inner_fold].copy()
        inner_valid = train[train["fold"] == inner_fold].copy()
        if inner_train.empty or inner_valid.empty:
            continue
        inner_valid["selector_score"] = train_scores(
            inner_train,
            inner_valid,
            features,
            selector,
            seed + int(inner_fold),
        )
        selected = select_one_per_page(inner_valid)
        fold_scores.append(float(selected["actual_cer"].mean()))
    return safe_mean(fold_scores)


def evaluate_engine(
    candidates: pd.DataFrame,
    engine: str,
    selectors: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    data = candidates[candidates["engine"] == engine].copy().reset_index(drop=True)
    features = candidate_feature_columns(data)
    scored_pieces = []
    selected_rows = []
    selector_choices = []
    inner_rows = []

    for outer_fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != outer_fold].copy()
        test = data[data["fold"] == outer_fold].copy()

        inner_scores = []
        for selector in selectors:
            score = validation_mean_cer(train, selector, features, seed + 1000 * int(outer_fold))
            inner_scores.append((score, selector))
            inner_rows.append({
                "engine": engine,
                "outer_fold": int(outer_fold),
                "selector": selector,
                "inner_valid_mean_cer": float(score),
            })
        inner_scores.sort(key=lambda item: (item[0], selectors.index(item[1])))
        chosen_selector = inner_scores[0][1]
        selector_choices.append(chosen_selector)

        test = test.copy()
        test["selector_score"] = train_scores(
            train,
            test,
            features,
            chosen_selector,
            seed + 2000 * int(outer_fold),
        )
        test["selected_selector_family"] = chosen_selector
        test["selected_by_nested_model"] = False
        selected = select_one_per_page(test)
        selected_idx = set(selected.index.tolist())
        # selected index values from selected are not original after dict conversion,
        # so mark by deterministic page/rank matching instead.
        for _, row in selected.iterrows():
            mask = (
                (test[GROUP_COL].astype(str) == str(row[GROUP_COL]))
                & (test["rank"].astype(int) == int(row["rank"]))
            )
            test.loc[mask, "selected_by_nested_model"] = True
        scored_pieces.append(test)
        selected_rows.extend(selected.to_dict("records"))

    scored = pd.concat(scored_pieces, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    summary = summarize_selected(selected, scored, engine, features, selector_choices)
    summary["inner_validation"] = inner_rows
    return scored, summary


def summarize_selected(
    selected: pd.DataFrame,
    scored: pd.DataFrame,
    engine: str,
    features: List[str],
    selector_choices: List[str],
) -> Dict:
    selected_idx = selected.set_index([GROUP_COL])
    top1 = scored[scored["rank"] == 1].set_index(GROUP_COL)["actual_cer"].loc[selected_idx.index]
    agreement = scored[scored["agreement_selected"]].set_index(GROUP_COL)["actual_cer"].loc[selected_idx.index]
    bestof3 = scored.groupby(GROUP_COL)["actual_cer"].min().loc[selected_idx.index]
    selector = selected_idx["actual_cer"]
    fixed = selected_idx["fixedbest_actual_cer"].astype(float)
    nopre = selected_idx["nopre_cer"].astype(float)

    return {
        "engine": engine,
        "n_images": int(len(selected)),
        "n_features": int(len(features)),
        "selector_choice_counts": dict(Counter(selector_choices)),
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


def write_report(summaries: Sequence[Dict], output_dir: Path, missing_cache_rows: int, selectors: List[str]) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("NESTED ORACLE-FREE TOP-3 SELECTOR")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Selector families: {', '.join(selectors)}")
    lines.append(f"Missing OCR cache rows: {missing_cache_rows}")
    lines.append("")
    header = (
        "Engine      Chosen  Top1   Agree  Fixed  NoPre  Best3  "
        "Sel-Top1 Sel-Agree Sel-Fixed pTop1 pAgree pFixed Choices"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{s['engine']:<11s} "
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
            f"{s['p_selector_vs_fixedbest']:.3g} "
            f"{s['selector_choice_counts']}"
        )
    lines.append("")
    lines.append("Selected rank counts")
    for s in summaries:
        lines.append(f"  {s['engine']}: {s['selected_rank_counts']}")
    (output_dir / "nested_selector_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--selectors", default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-legacy-pooled-oof", action="store_true",
                        help="Historical reproduction only; NOT submission-safe nested evaluation")
    args = parser.parse_args()
    if not args.allow_legacy_pooled_oof:
        parser.error("Pooled-OOF upstream leakage: use evaluate_top3_selector_strict_nested.py. "
                     "Historical reproduction requires --allow-legacy-pooled-oof.")

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

    all_scored = []
    summaries = []
    inner_rows = []
    for engine in engines:
        print(f"\nEngine: {engine}")
        scored, summary = evaluate_engine(candidates, engine, selectors, args.seed)
        all_scored.append(scored)
        inner_rows.extend(summary.pop("inner_validation"))
        summaries.append(summary)
        print(
            f"  selector={fmt_pct(summary['selector_mean'])}, "
            f"top1={fmt_pct(summary['top1_mean'])}, "
            f"agreement={fmt_pct(summary['agreement_mean'])}, "
            f"fixed={fmt_pct(summary['fixedbest_mean'])}, "
            f"best3={fmt_pct(summary['bestof3_mean'])}, "
            f"choices={summary['selector_choice_counts']}"
        )

    scored_all = pd.concat(all_scored, ignore_index=True)
    scored_all.to_csv(args.output_dir / "nested_selector_candidate_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "nested_selector_summary.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.output_dir / "nested_selector_inner_validation.csv", index=False)
    (args.output_dir / "nested_selector_summary.json").write_text(
        json.dumps(to_jsonable(summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summaries, args.output_dir, missing_cache_rows, selectors)
    print(f"\nWrote: {args.output_dir / 'nested_selector_report.txt'}")


if __name__ == "__main__":
    main()
