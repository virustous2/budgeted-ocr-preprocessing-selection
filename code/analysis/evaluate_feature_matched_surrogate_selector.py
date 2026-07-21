#!/usr/bin/env python3
"""Feature-matched ablation of the surrogate top-3 nested selector.

Confound being tested: in the policy-matched comparison, the selector on the
Surrogate top-3 receives *real* predicted-CER features (pred_cer, pred_delta_*,
pred_rank_norm), whereas the NN-K / FixedTopK selectors receive only degenerate
rank-based values (because those policies have no CER prediction). A win for the
Surrogate selector could therefore come from the richer features rather than a
better shortlist.

This script removes that advantage: it keeps the *exact same* Surrogate top-3
shortlist and OCR outputs, but replaces pred_cer with the within-page rank
(1, 2, 3), so build_candidates regenerates the identical degenerate pred_*
features that NN-K / FixedTopK selectors get. The nested selector is then refit
on the same outer folds. Comparing this feature-matched selector against NN-K
removes the prediction-feature advantage and makes the policy comparison more
directly attributable to differences in shortlist content and ordering.

No OCR is run. It reuses the saved Surrogate candidate scores (for the top-3
identity and params), the existing OCR-text cache, and the per-image baseline
file, then calls the unmodified build_candidates / evaluate_engine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "analysis"))

from evaluate_nn_shortlist_baseline import GROUP_COL, PARAM_COLS  # type: ignore  # noqa: E402
from evaluate_oracle_free_top3_selector_from_predictions import (  # type: ignore  # noqa: E402
    build_candidates,
    parse_engines,
)
from evaluate_oracle_free_top3_selector_nested import (  # type: ignore  # noqa: E402
    evaluate_engine,
    to_jsonable,
    write_report,
)

K = 3


def build_rank_predictions(scores_path: Path, engines: List[str]) -> pd.DataFrame:
    """From saved Surrogate candidate scores, emit a predictions table whose
    pred_cer equals the within-page rank so the shortlist is preserved but the
    prediction features become degenerate (matched to NN-K / FixedTopK)."""
    scores = pd.read_csv(scores_path)
    needed = {"engine", GROUP_COL, "fold", "rank", "actual_cer", *PARAM_COLS}
    missing = needed.difference(scores.columns)
    if missing:
        raise ValueError(f"{scores_path}: missing columns {sorted(missing)}")
    scores = scores[scores["engine"].astype(str).isin(engines)].copy()

    # Each page must have exactly the K top-3 rows with ranks 1..K.
    sizes = scores.groupby(["engine", GROUP_COL]).size()
    if not (sizes == K).all():
        bad = sizes[sizes != K].head(5).to_dict()
        raise ValueError(f"{scores_path}: expected {K} rows per engine/page; examples {bad}")
    ranks = scores.groupby(["engine", GROUP_COL])["rank"].apply(lambda r: sorted(r.tolist()))
    if not ranks.apply(lambda r: r == list(range(1, K + 1))).all():
        raise ValueError(f"{scores_path}: ranks are not exactly 1..{K} per page")

    pred = pd.DataFrame({
        "engine": scores["engine"].astype(str),
        GROUP_COL: scores[GROUP_COL].astype(str),
        "outer_fold": scores["fold"].astype(int),   # build_candidates expects 'outer_fold'
        "pred_cer": scores["rank"].astype(float),    # rank as the (degenerate) prediction
        "actual_cer": scores["actual_cer"].astype(float),
    })
    for p in PARAM_COLS:
        pred[p] = scores[p].values
    return pred.sort_values(["engine", GROUP_COL, "pred_cer"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate-scores", required=True, type=Path,
                    help="Existing surrogate nested_selector_candidate_scores.csv (source of the top-3 identity).")
    ap.add_argument("--cache-path", required=True, type=Path,
                    help="OCR-text cache used by the original surrogate selector run.")
    ap.add_argument("--per-image-path", required=True, type=Path,
                    help="Per-image baseline CSV (fixedbest/nopre/samporacle).")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--selectors", default="rf_cls,logreg_cls,rf_reg,gbr_reg,ridge_reg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    engines = parse_engines(args.engines)
    selectors = parse_engines(args.selectors)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = build_rank_predictions(args.surrogate_scores, engines)
    pred_path = args.output_dir / "feature_matched_predictions.csv"
    pred.to_csv(pred_path, index=False)

    candidates = build_candidates(
        candidate_predictions_path=pred_path,
        cache_path=args.cache_path,
        per_image_path=args.per_image_path,
        engines=engines,
    )
    missing = int(candidates.attrs.get("missing_cache_rows", 0))
    if missing:
        raise RuntimeError(f"{missing} OCR-cache rows missing; the surrogate cache should already cover its own top-3")

    # Guard: the regenerated prediction features must be degenerate (rank-based),
    # i.e. identical to what NN-K / FixedTopK selectors receive.
    per_page_pred = candidates.groupby([GROUP_COL, "engine"])['pred_cer'].apply(lambda s: sorted(s.tolist()))
    if not per_page_pred.apply(lambda s: s == [1.0, 2.0, 3.0]).all():
        raise AssertionError("pred_cer is not the degenerate rank 1/2/3 after rebuild")

    all_scored, summaries, inner_rows = [], [], []
    for engine in engines:
        scored, summary = evaluate_engine(candidates, engine, selectors, args.seed)
        all_scored.append(scored)
        inner_rows.extend(summary.pop("inner_validation"))
        summary["policy"] = "surrogate_feature_matched"
        summaries.append(summary)
        print(f"  {engine}: selector={summary['selector_mean']*100:.3f}%  "
              f"fixedbest={summary['fixedbest_mean']*100:.3f}%")

    pd.concat(all_scored, ignore_index=True).to_csv(
        args.output_dir / "nested_selector_candidate_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "nested_selector_summary.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.output_dir / "nested_selector_inner_validation.csv", index=False)
    (args.output_dir / "nested_selector_summary.json").write_text(
        json.dumps(to_jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summaries, args.output_dir, 0, selectors)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "policy": "surrogate_feature_matched",
                "k": K,
                "surrogate_scores": str(args.surrogate_scores),
                "cache_path": str(args.cache_path),
                "per_image_path": str(args.per_image_path),
                "engines": engines,
                "selector_families": selectors,
                "seed": args.seed,
                "candidate_rows": int(len(pred)),
                "missing_cache_rows": missing,
                "prediction_feature_transform": (
                    "Replace surrogate pred_cer with within-page rank 1/2/3; "
                    "build_candidates then regenerates rank-derived pred_delta_* "
                    "and pred_rank_norm features."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
