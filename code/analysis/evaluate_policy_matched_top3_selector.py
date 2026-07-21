#!/usr/bin/env python3
"""Evaluate the nested oracle-free selector on NN-K or FixedTopK top-3.

This script supplies the budget-matched control missing from the current
surrogate-top-3 selector result.  It reconstructs each policy's fold-specific
top-3 preprocessing shortlist from the candidate-response table and the saved
NN-K per-page output, then applies the *same* nested selector protocol used
for the surrogate shortlist.

The default ``--dry-run`` never initializes OCR.  It writes the exact
policy-specific candidate list plus a cache-coverage report, allowing the
required OCR work to be inspected before any new OCR calls are made.  With
``--fill-missing-cache``, OCR is run only for missing page/candidate/engine
outputs and written to a new working cache.  The source cache is never
modified by this script.

For NN-K, the historical ``nn_per_image.csv`` must contain at least three
unique transferred candidate combinations among its stored ``nn1``--``nn15``
columns. The script stops rather than inventing an approximation when that
provenance is incomplete.

Examples
--------
Raw NN-K coverage only, using a local dataset and OCR-text cache not distributed with this release::

    python code/analysis/evaluate_policy_matched_top3_selector.py \
      --policy nn-k \
      --data-path /path/to/preprocessing_data.csv \
      --nn-per-image-path /path/to/nn_per_image.csv \
      --baseline-per-image-path /path/to/per_page_baselines.csv \
      --source-cache-path /path/to/ocr_text_cache.csv \
      --working-cache-path /path/to/policy_matched_ocr_cache.csv \
      --output-dir /path/to/policy_matched_selector/nn_k \
      --engines vietocr,tesseract,easyocr \
      --setting raw \
      --dry-run

After reviewing ``cache_coverage.csv`` and ``missing_ocr_candidates.csv``,
replace ``--dry-run`` with ``--fill-missing-cache --device cuda``.  The latter
fills the new working cache and then runs the nested selector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "analysis"))

from evaluate_nn_shortlist_baseline import (  # type: ignore  # noqa: E402
    GROUP_COL,
    PARAM_COLS,
    build_combo_cer_maps,
    fixed_topk_combos,
    unique_preserve_order,
)
from evaluate_oracle_free_top3_selector_from_predictions import (  # type: ignore  # noqa: E402
    K,
    build_candidates,
    load_cache,
    make_key,
    parse_engines,
    round_params,
)
from evaluate_oracle_free_top3_selector_nested import (  # type: ignore  # noqa: E402
    DEFAULT_SELECTORS,
    evaluate_engine,
    to_jsonable,
    write_report,
)
from evaluate_topk_consensus_deployable import (  # type: ignore  # noqa: E402
    build_image_index,
    default_images_folder,
    ensure_ocr_texts,
    infer_setting,
    init_ocr,
    save_cache,
)


Combo = Tuple[int, int, float, float, float]


def combo_key(image_name: str, combo: Combo) -> Tuple[str, int, int, float, float, float]:
    return (
        str(image_name),
        int(combo[0]),
        int(combo[1]),
        round(float(combo[2]), 10),
        round(float(combo[3]), 10),
        round(float(combo[4]), 10),
    )


def combo_values(row: pd.Series, prefix: str) -> Combo:
    return (
        int(row[f"{prefix}_gaussian_ksize"]),
        int(row[f"{prefix}_bilateral_d"]),
        round(float(row[f"{prefix}_clahe_clip"]), 10),
        round(float(row[f"{prefix}_gamma"]), 10),
        round(float(row[f"{prefix}_sharpen"]), 10),
    )


def add_combo(record: Dict[str, object], combo: Combo) -> None:
    for column, value in zip(PARAM_COLS, combo):
        record[column] = value


def load_cache_if_present(path: Path) -> Dict:
    return load_cache(path) if path.exists() else {}


def merged_cache(source_path: Path | None, working_path: Path) -> Dict:
    """Load immutable source cache, then overwrite with working-cache records."""
    cache: Dict = {}
    if source_path is not None:
        cache.update(load_cache_if_present(source_path))
    cache.update(load_cache_if_present(working_path))
    return cache


def policy_fold_map(nn_per_image: pd.DataFrame, engine: str) -> Dict[str, int]:
    rows = nn_per_image[nn_per_image["engine"].astype(str) == engine].copy()
    if rows.empty:
        raise ValueError(f"No NN-K rows found for engine '{engine}'")
    rows[GROUP_COL] = rows[GROUP_COL].astype(str)
    if rows[GROUP_COL].duplicated().any():
        raise ValueError(f"Duplicate NN-K per-image rows for engine '{engine}'")
    return dict(zip(rows[GROUP_COL], rows["fold"].astype(int)))


def validate_baseline_folds(
    baseline_per_image: pd.DataFrame,
    nn_per_image: pd.DataFrame,
    engines: Sequence[str],
) -> None:
    """Ensure policy shortlists and selector baselines use the same outer folds."""
    required = {"engine", GROUP_COL, "fold", "fixedbest_actual_cer", "nopre_cer"}
    missing = required.difference(baseline_per_image.columns)
    if missing:
        raise ValueError(
            "Baseline per-image file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    for engine in engines:
        left = nn_per_image[nn_per_image["engine"].astype(str) == engine][
            [GROUP_COL, "fold"]
        ].copy()
        right = baseline_per_image[
            baseline_per_image["engine"].astype(str) == engine
        ][[GROUP_COL, "fold"]].copy()
        left[GROUP_COL] = left[GROUP_COL].astype(str)
        right[GROUP_COL] = right[GROUP_COL].astype(str)
        merged = left.merge(right, on=GROUP_COL, how="outer", suffixes=("_policy", "_baseline"), indicator=True)
        if (merged["_merge"] != "both").any():
            raise ValueError(f"Page mismatch between policy and baseline rows for engine '{engine}'")
        if (merged["fold_policy"].astype(int) != merged["fold_baseline"].astype(int)).any():
            raise ValueError(f"Outer-fold mismatch between policy and baseline rows for engine '{engine}'")


def verify_shortlist_cer(
    combo_cer: Dict[Combo, float],
    combos: Sequence[Combo],
    expected: float,
    label: str,
) -> None:
    actual = min(combo_cer[combo] for combo in combos)
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label} reconstruction does not match saved per-image CER: "
            f"reconstructed={actual:.12g}, saved={float(expected):.12g}"
        )


def reconstruct_policy_predictions(
    data: pd.DataFrame,
    nn_per_image: pd.DataFrame,
    engines: Sequence[str],
    policy: str,
) -> pd.DataFrame:
    """Build one exact, ordered K=3 shortlist for every policy page/engine.

    NN-K is recomputed inside the saved outer-fold partition rather than taking
    only ``nn1``--``nn3`` columns.  This preserves duplicate removal and handles
    the rare case in which the first three unique transferred candidates appear
    after the third nearest neighbor.
    """
    data = round_params(data)
    data[GROUP_COL] = data[GROUP_COL].astype(str)
    nn_per_image = nn_per_image.copy()
    nn_per_image[GROUP_COL] = nn_per_image[GROUP_COL].astype(str)
    combo_maps = build_combo_cer_maps(data, list(engines))
    rows: List[Dict[str, object]] = []
    data_page_names = set(data[GROUP_COL].astype(str))

    for engine in engines:
        target_col = f"cer_{engine}"
        if target_col not in data.columns:
            raise ValueError(f"Candidate-response table lacks {target_col}")
        saved_engine = nn_per_image[nn_per_image["engine"].astype(str) == engine].copy()
        if saved_engine.empty:
            raise ValueError(f"No saved NN-K rows for engine '{engine}'")
        saved_engine = saved_engine.set_index(GROUP_COL, verify_integrity=True)
        fold_map = policy_fold_map(nn_per_image, engine)
        if set(fold_map) != data_page_names:
            raise ValueError(f"Saved NN-K pages do not match candidate-response pages for engine '{engine}'")

        for fold in sorted(set(fold_map.values())):
            test_names = [name for name, page_fold in fold_map.items() if page_fold == fold]
            train_names = [name for name, page_fold in fold_map.items() if page_fold != fold]
            train_df = data[data[GROUP_COL].isin(train_names)].copy()
            fixed_combos = fixed_topk_combos(train_df, target_col, K)

            for image_name in test_names:
                if policy == "fixed-topk":
                    shortlist = list(fixed_combos)
                    expected_col = "fixed_topk_k3_cer"
                else:
                    saved_row = saved_engine.loc[image_name]
                    transferred = []
                    for neighbor_rank in range(1, 16):
                        prefix = f"nn{neighbor_rank}"
                        needed = [f"{prefix}_{column}" for column in PARAM_COLS]
                        if any(column not in saved_row.index or pd.isna(saved_row[column]) for column in needed):
                            continue
                        transferred.append(combo_values(saved_row, prefix))
                    shortlist = unique_preserve_order(transferred)
                    if len(shortlist) < K:
                        raise ValueError(
                            "Cannot reconstruct the exact NN-K top-3 shortlist for "
                            f"{engine}/{image_name}: only {len(shortlist)} unique candidates are "
                            "stored among nn1--nn15. Recreate NN-K with explicit final shortlist "
                            "columns before evaluating this row."
                        )
                    shortlist = shortlist[:K]
                    expected_col = "nn_k3_cer"

                if len(shortlist) != K:
                    raise ValueError(f"{policy} produced {len(shortlist)} candidates for {engine}/{image_name}")
                saved_row = saved_engine.loc[image_name]
                verify_shortlist_cer(
                    combo_maps[engine][image_name],
                    shortlist,
                    float(saved_row[expected_col]),
                    f"{policy} {engine}/{image_name}",
                )
                for rank, combo in enumerate(shortlist, start=1):
                    record: Dict[str, object] = {
                        "policy": policy,
                        "engine": engine,
                        GROUP_COL: image_name,
                        "outer_fold": int(fold),
                        # Existing selector code expects an ascending score named pred_cer.
                        # For these non-surrogate policies it is the policy shortlist order.
                        "pred_cer": float(rank),
                        "policy_rank_score": float(rank),
                        "actual_cer": float(combo_maps[engine][image_name][combo]),
                        "rank_within_policy": int(rank),
                    }
                    add_combo(record, combo)
                    rows.append(record)

    predictions = pd.DataFrame(rows)
    expected_rows = len(data_page_names) * len(engines) * K
    if len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} policy candidate rows, found {len(predictions)}")
    return predictions.sort_values(["engine", GROUP_COL, "policy_rank_score"], kind="mergesort").reset_index(drop=True)


def cache_coverage(
    predictions: pd.DataFrame,
    cache: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple, Set[str]]]:
    """Return per-engine coverage, missing rows, and OCR work grouped by key."""
    coverage_rows = []
    missing_rows = []
    missing_by_key: Dict[Tuple, Set[str]] = defaultdict(set)

    for engine, group in predictions.groupby("engine", sort=True):
        total = len(group)
        complete = 0
        for _, row in group.iterrows():
            key = make_key(row)
            cached = cache.get(key, {})
            has_text = isinstance(cached.get(f"text_{engine}"), str)
            has_metadata = "n_detected_crops" in cached
            if has_text and has_metadata:
                complete += 1
                continue
            missing_by_key[key].add(str(engine))
            record = row.to_dict()
            record["missing_engine"] = str(engine)
            record["has_cached_text"] = bool(has_text)
            record["has_detection_metadata"] = bool(has_metadata)
            missing_rows.append(record)
        coverage_rows.append({
            "engine": str(engine),
            "required_candidate_engine_rows": int(total),
            "complete_cached_rows": int(complete),
            "missing_candidate_engine_rows": int(total - complete),
            "coverage_pct": 100.0 * complete / total if total else float("nan"),
        })

    missing = pd.DataFrame(missing_rows)
    if not missing.empty:
        missing = missing.sort_values(["missing_engine", GROUP_COL, "policy_rank_score"], kind="mergesort")
    return pd.DataFrame(coverage_rows), missing, missing_by_key


def fill_cache(
    missing_by_key: Dict[Tuple, Set[str]],
    predictions: pd.DataFrame,
    cache: Dict,
    setting: str,
    images_folder: Path,
    device: str,
    save_every: int,
    working_cache_path: Path,
) -> int:
    if not missing_by_key:
        return 0

    image_index = build_image_index(images_folder)
    all_engines = sorted({engine for engines in missing_by_key.values() for engine in engines})
    preprocessor, runner = init_ocr(setting, device, all_engines)
    by_key = {make_key(row): row for _, row in predictions.iterrows()}
    total = len(missing_by_key)

    for index, (key, engines) in enumerate(sorted(missing_by_key.items(), key=lambda item: item[0]), start=1):
        row = by_key[key]
        ensure_ocr_texts(
            row=row,
            engines=sorted(engines),
            image_index=image_index,
            preprocessor=preprocessor,
            runner=runner,
            cache=cache,
        )
        if save_every > 0 and index % save_every == 0:
            save_cache(cache, working_cache_path)
            print(f"  OCR cache: {index}/{total} unique candidates completed", flush=True)

    save_cache(cache, working_cache_path)
    return total


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    coverage: pd.DataFrame,
    ocr_unique_candidates: int,
) -> None:
    metadata = {
        "policy": args.policy,
        "k": K,
        "data_path": str(args.data_path),
        "nn_per_image_path": str(args.nn_per_image_path),
        "baseline_per_image_path": str(args.baseline_per_image_path),
        "source_cache_path": str(args.source_cache_path) if args.source_cache_path else None,
        "working_cache_path": str(args.working_cache_path),
        "setting": args.setting,
        "engines": parse_engines(args.engines),
        "selector_families": parse_engines(args.selectors),
        "candidate_rows": int(len(predictions)),
        "unique_policy_candidates": int(predictions.apply(make_key, axis=1).nunique()),
        "unique_candidates_needing_ocr": int(ocr_unique_candidates),
        "cache_coverage": coverage.to_dict(orient="records"),
        "note": (
            "pred_cer is policy order (1, 2, 3) for NN-K/FixedTopK, not a CER prediction. "
            "The nested selector is refit independently for this policy."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["nn-k", "fixed-topk"], required=True)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--nn-per-image-path", required=True, type=Path)
    parser.add_argument(
        "--baseline-per-image-path",
        required=True,
        type=Path,
        help="Surrogate top-3 per-image CSV providing fold-matched FixedBest and NoPre baselines.",
    )
    parser.add_argument(
        "--source-cache-path",
        default=None,
        type=Path,
        help="Read-only existing OCR-text cache used to seed the working cache.",
    )
    parser.add_argument(
        "--working-cache-path",
        required=True,
        type=Path,
        help="New or reusable cache written only when --fill-missing-cache is supplied.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engines", required=True, help="Comma-separated recognizers.")
    parser.add_argument("--selectors", default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--setting", choices=["auto", "raw", "contour", "sroie"], default="auto")
    parser.add_argument("--images-folder", default=None, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-save-every", type=int, default=25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate and cache-coverage artifacts, then exit before OCR or selector fitting.",
    )
    parser.add_argument(
        "--fill-missing-cache",
        action="store_true",
        help="Run OCR only for missing policy candidate outputs, write the working cache, then fit/evaluate the nested selector.",
    )
    parser.add_argument(
        "--fill-only",
        action="store_true",
        help="With --fill-missing-cache, stop after writing OCR cache and coverage artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.fill_missing_cache:
        raise ValueError("Choose either --dry-run or --fill-missing-cache, not both")
    if args.fill_only and not args.fill_missing_cache:
        raise ValueError("--fill-only requires --fill-missing-cache")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    engines = parse_engines(args.engines)
    selectors = parse_engines(args.selectors)
    data = pd.read_csv(args.data_path)
    nn_per_image = pd.read_csv(args.nn_per_image_path)
    baseline_per_image = pd.read_csv(args.baseline_per_image_path)
    validate_baseline_folds(baseline_per_image, nn_per_image, engines)

    print(f"Policy             : {args.policy}")
    print(f"Candidate data     : {args.data_path}")
    print(f"NN-K per-image     : {args.nn_per_image_path}")
    print(f"Baseline per-image : {args.baseline_per_image_path}")
    print(f"Working OCR cache  : {args.working_cache_path}")
    print(f"Engines            : {engines}")

    predictions = reconstruct_policy_predictions(data, nn_per_image, engines, args.policy)
    predictions_path = args.output_dir / "policy_top3_candidates.csv"
    predictions.to_csv(predictions_path, index=False)

    cache = merged_cache(args.source_cache_path, args.working_cache_path)
    coverage, missing, missing_by_key = cache_coverage(predictions, cache)
    coverage_path = args.output_dir / "cache_coverage.csv"
    missing_path = args.output_dir / "missing_ocr_candidates.csv"
    coverage.to_csv(coverage_path, index=False)
    missing.to_csv(missing_path, index=False)
    write_run_metadata(args.output_dir, args, predictions, coverage, len(missing_by_key))

    print("\nOCR cache coverage")
    print(coverage.to_string(index=False))
    print(f"Unique candidates requiring OCR: {len(missing_by_key)}")
    print(f"Candidate list : {predictions_path}")
    print(f"Coverage report: {coverage_path}")
    print(f"Missing rows   : {missing_path}")

    if args.dry_run:
        print("\nDry run complete. No OCR was initialized and no cache was modified.")
        return

    if missing_by_key and not args.fill_missing_cache:
        raise RuntimeError(
            "OCR cache is incomplete. Review missing_ocr_candidates.csv, then rerun with "
            "--fill-missing-cache --device <device>."
        )

    if missing_by_key:
        setting = infer_setting(args.data_path, args.setting)
        images_folder = args.images_folder or default_images_folder(setting)
        print(f"\nFilling {len(missing_by_key)} unique OCR candidates on {args.device}...")
        fill_cache(
            missing_by_key=missing_by_key,
            predictions=predictions,
            cache=cache,
            setting=setting,
            images_folder=Path(images_folder),
            device=args.device,
            save_every=args.cache_save_every,
            working_cache_path=args.working_cache_path,
        )
        coverage, missing, missing_by_key = cache_coverage(predictions, cache)
        coverage.to_csv(coverage_path, index=False)
        missing.to_csv(missing_path, index=False)
        write_run_metadata(args.output_dir, args, predictions, coverage, len(missing_by_key))
        if missing_by_key:
            raise RuntimeError("OCR cache remains incomplete after fill; inspect missing_ocr_candidates.csv")

    if args.fill_only:
        print("\nCache fill complete. --fill-only requested; nested selector was not run.")
        return

    # build_candidates uses the standardized candidate/feature implementation
    # shared with the existing surrogate selector.  It requires a CSV cache.
    cache_for_evaluation = args.working_cache_path if args.working_cache_path.exists() else args.source_cache_path
    if cache_for_evaluation is None:
        raise RuntimeError("No OCR cache available for selector evaluation")
    candidates = build_candidates(
        candidate_predictions_path=predictions_path,
        cache_path=cache_for_evaluation,
        per_image_path=args.baseline_per_image_path,
        engines=engines,
    )
    missing_cache_rows = int(candidates.attrs.get("missing_cache_rows", 0))
    if missing_cache_rows:
        raise RuntimeError(f"Selector candidate build still reports {missing_cache_rows} missing OCR cache rows")

    all_scored = []
    summaries = []
    inner_rows = []
    for engine in engines:
        print(f"\nNested selector: {args.policy} / {engine}")
        scored, summary = evaluate_engine(candidates, engine, selectors, args.seed)
        all_scored.append(scored)
        inner_rows.extend(summary.pop("inner_validation"))
        summary["policy"] = args.policy
        summaries.append(summary)
        print(
            f"  selector={summary['selector_mean'] * 100:.2f}%, "
            f"top1={summary['top1_mean'] * 100:.2f}%, "
            f"agreement={summary['agreement_mean'] * 100:.2f}%, "
            f"fixed={summary['fixedbest_mean'] * 100:.2f}%"
        )

    scored_all = pd.concat(all_scored, ignore_index=True)
    scored_all.to_csv(args.output_dir / "nested_selector_candidate_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "nested_selector_summary.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.output_dir / "nested_selector_inner_validation.csv", index=False)
    (args.output_dir / "nested_selector_summary.json").write_text(
        json.dumps(to_jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(summaries, args.output_dir, 0, selectors)
    print(f"\nWrote: {args.output_dir / 'nested_selector_report.txt'}")


if __name__ == "__main__":
    main()
