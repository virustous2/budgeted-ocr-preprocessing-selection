#!/usr/bin/env python3
"""
Evaluate an oracle-free Top-K consensus output selector.

This is a deployable check that converts a surrogate-ranked shortlist into one
OCR output without using reference text or CER during selection:

  page -> surrogate ranks the 50 candidates -> OCR top-K candidates
       -> choose the output with the smallest mean normalized edit distance
          to the other top-K outputs.

The actual CER of the consensus-selected candidate is then looked up from the
already collected candidate-response table for evaluation only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

RELEASE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

GROUP_COL = "image_name"
PARAM_COLS = ["gaussian_ksize", "bilateral_d", "clahe_clip", "gamma", "sharpen"]


def combo_mask(frame: pd.DataFrame, combo_vals) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for col, val in zip(PARAM_COLS, combo_vals):
        mask &= np.isclose(frame[col].values, val)
    return mask


def load_training_helpers():
    """Import surrogate training helpers lazily so --help does not need ML deps."""
    from main.train_surrogate_multiengine_kfold import (  # noqa: E402
        get_models,
        load_exact_nopre_maps,
        prepare_data,
        resolve_exact_nopre_csv,
        select_best_model,
        summarize_model_selection,
    )

    return {
        "get_models": get_models,
        "load_exact_nopre_maps": load_exact_nopre_maps,
        "prepare_data": prepare_data,
        "resolve_exact_nopre_csv": resolve_exact_nopre_csv,
        "select_best_model": select_best_model,
        "summarize_model_selection": summarize_model_selection,
    }


def _safe_wilcoxon(a: Iterable[float], b: Iterable[float], alternative: str = "less") -> Dict:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or np.allclose(x, y):
        return {"stat": float("nan"), "p": float("nan"), "n": int(len(x))}
    stat, p_value = wilcoxon(x, y, alternative=alternative)
    return {"stat": float(stat), "p": float(p_value), "n": int(len(x))}


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr))


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pairwise_stats(a: Iterable[float], b: Iterable[float]) -> Dict:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
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


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _levenshtein_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:
        pass

    try:
        import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:
        pass

    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    na = _norm_text(a)
    nb = _norm_text(b)
    denom = max(len(na), len(nb), 1)
    return _levenshtein_distance(na, nb) / denom


def consensus_select(rows: List[Dict]) -> Tuple[Dict, List[float]]:
    """Return the row with smallest mean distance to other non-empty outputs.

    Empty OCR outputs are ignored when at least one non-empty output exists in
    the shortlist. This keeps a batch of blank strings from winning consensus
    by tying at zero distance. Ties are resolved by rank order, so the lower
    predicted-CER candidate wins if consensus scores are equal.
    """
    if not rows:
        raise ValueError("Cannot select consensus from an empty shortlist")
    if len(rows) == 1:
        return rows[0], [0.0]

    active_indices = [
        idx for idx, row in enumerate(rows)
        if len(_norm_text(row.get("ocr_text", ""))) > 0
    ]
    if not active_indices:
        active_indices = list(range(len(rows)))

    scores = [float("inf")] * len(rows)
    for i in active_indices:
        row_i = rows[i]
        dists = []
        for j in active_indices:
            if i == j:
                continue
            row_j = rows[j]
            dists.append(normalized_edit_distance(row_i["ocr_text"], row_j["ocr_text"]))
        scores[i] = float(np.mean(dists)) if dists else 0.0

    best_i = min(range(len(rows)), key=lambda idx: (scores[idx], rows[idx]["rank"]))
    return rows[best_i], scores


def infer_setting(data_path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    lower = str(data_path).lower()
    if "sroie" in lower:
        return "sroie"
    if "contour" in lower:
        return "contour"
    return "raw"


def default_images_folder(setting: str) -> Path:
    source_root = RELEASE_ROOT / "data" / "source_documents_not_included"
    if setting == "sroie":
        return source_root / "sroie" / "images"
    if setting == "contour":
        return source_root / "vietnamese_contour" / "images"
    return source_root / "vietnamese" / "images"


def build_image_index(images_folder: Path) -> Dict[str, Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    index: Dict[str, Path] = {}
    for path in images_folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stem = path.stem
        if stem in index:
            raise ValueError(f"Duplicate image stem '{stem}': {index[stem]} and {path}")
        index[stem] = path
    if not index:
        raise FileNotFoundError(f"No image files found under {images_folder}")
    return index


def make_param_key(row: pd.Series) -> Tuple:
    return (
        str(row[GROUP_COL]),
        int(row["gaussian_ksize"]),
        int(row["bilateral_d"]),
        round(float(row["clahe_clip"]), 10),
        round(float(row["gamma"]), 10),
        round(float(row["sharpen"]), 10),
    )


def record_from_key(key: Tuple) -> Dict:
    return {
        GROUP_COL: key[0],
        "gaussian_ksize": key[1],
        "bilateral_d": key[2],
        "clahe_clip": key[3],
        "gamma": key[4],
        "sharpen": key[5],
    }


def load_cache(cache_path: Path) -> Dict[Tuple, Dict]:
    if not cache_path.exists():
        return {}
    cache_df = pd.read_csv(cache_path, keep_default_na=False)
    cache: Dict[Tuple, Dict] = {}
    for _, row in cache_df.iterrows():
        key = make_param_key(row)
        cache[key] = row.to_dict()
    return cache


def save_cache(cache: Dict[Tuple, Dict], cache_path: Path) -> None:
    if not cache:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [cache[key] for key in sorted(cache.keys(), key=lambda k: str(k))]
    pd.DataFrame(rows).to_csv(cache_path, index=False)


def init_ocr(setting: str, device: str, engines: List[str]):
    if setting == "sroie":
        from SROIRE_COMPARISON.sroie_scripts.collect_lhs_sroie import (
            ImagePreprocessor,
            SROIEOCRRunner,
        )

        return ImagePreprocessor(), SROIEOCRRunner(device=device, engines=engines)

    from main.collect_multiengine_data import ImagePreprocessor, MultiEngineOCRRunner

    return ImagePreprocessor(), MultiEngineOCRRunner(device=device, engines=engines)


def run_ocr_with_metadata(runner, image: np.ndarray, engines: List[str]) -> Tuple[Dict[str, str], Dict]:
    """Run shared detection once and return OCR text plus lightweight metadata."""
    outputs = {engine: "" for engine in engines}
    metadata = {"n_detected_crops": 0}

    try:
        crops = runner._detect(image)
    except Exception as exc:
        metadata["detect_error"] = repr(exc)
        return outputs, metadata

    metadata["n_detected_crops"] = int(len(crops))
    if not crops:
        return outputs, metadata

    recognizers = getattr(runner, "recognizers", {})
    for engine_name in engines:
        if engine_name not in recognizers:
            metadata[f"error_{engine_name}"] = "engine_not_loaded"
            continue
        try:
            if engine_name == "vietocr":
                outputs[engine_name] = runner._recognize_vietocr(crops)
            elif engine_name == "tesseract":
                outputs[engine_name] = runner._recognize_tesseract(crops)
            elif engine_name == "easyocr":
                outputs[engine_name] = runner._recognize_easyocr(crops)
            else:
                metadata[f"error_{engine_name}"] = "unknown_engine"
        except Exception as exc:
            metadata[f"error_{engine_name}"] = repr(exc)
            outputs[engine_name] = ""

    return outputs, metadata


def ensure_ocr_texts(
    row: pd.Series,
    engines: List[str],
    image_index: Dict[str, Path],
    preprocessor,
    runner,
    cache: Dict[Tuple, Dict],
) -> Dict:
    key = make_param_key(row)
    cached = cache.get(key)
    if (
        cached
        and "n_detected_crops" in cached
        and all(f"text_{engine}" in cached and isinstance(cached[f"text_{engine}"], str)
                for engine in engines)
    ):
        return cached

    image_name = str(row[GROUP_COL])
    if image_name not in image_index:
        raise FileNotFoundError(f"Missing image for '{image_name}' in image folder")

    image = cv2.imread(str(image_index[image_name]))
    if image is None:
        raise ValueError(f"Could not read image: {image_index[image_name]}")

    params = {col: row[col] for col in PARAM_COLS}
    processed = preprocessor.preprocess(image, params)
    ocr_outputs, metadata = run_ocr_with_metadata(runner, processed, engines)

    record = record_from_key(key)
    if cached:
        record.update(cached)
    record["n_detected_crops"] = int(metadata.get("n_detected_crops", 0))
    for key_name, value in metadata.items():
        if key_name != "n_detected_crops":
            record[key_name] = value
    for engine in engines:
        text = str(ocr_outputs.get(engine, "") or "")
        record[f"text_{engine}"] = text
        record[f"n_chars_{engine}"] = len(_norm_text(text))

    cache[key] = record
    return record


def get_fixedbest_for_group(group: pd.DataFrame, best_combo_vals, target_col: str) -> float:
    mask = combo_mask(group, best_combo_vals)
    if mask.sum() == 0:
        return float("nan")
    return float(group.loc[mask, target_col].iloc[0])


def summarize_per_image(per_image: pd.DataFrame, k: int, selected_models: List[str],
                        fold_metrics: List[Dict]) -> Dict:
    means = {
        "nopre_cer": _mean(per_image["nopre_cer"]),
        "fixedbest_actual_cer": _mean(per_image["fixedbest_actual_cer"]),
        "random1_actual_cer": _mean(per_image["random1_actual_cer"]),
        "top1_actual_cer": _mean(per_image["top1_actual_cer"]),
        f"bestof{k}_actual_cer": _mean(per_image[f"bestof{k}_actual_cer"]),
        f"top{k}_consensus_actual_cer": _mean(per_image[f"top{k}_consensus_actual_cer"]),
        "samporacle_cer": _mean(per_image["samporacle_cer"]),
    }
    tests = {
        "consensus_vs_top1": _safe_wilcoxon(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["top1_actual_cer"],
        ),
        "consensus_vs_fixedbest": _safe_wilcoxon(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["fixedbest_actual_cer"],
        ),
        "consensus_vs_nopre": _safe_wilcoxon(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["nopre_cer"],
        ),
        "consensus_vs_random1": _safe_wilcoxon(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["random1_actual_cer"],
        ),
        "top1_vs_fixedbest": _safe_wilcoxon(
            per_image["top1_actual_cer"],
            per_image["fixedbest_actual_cer"],
        ),
    }
    pairwise = {
        "consensus_minus_top1": _pairwise_stats(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["top1_actual_cer"],
        ),
        "consensus_minus_fixedbest": _pairwise_stats(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["fixedbest_actual_cer"],
        ),
        "consensus_minus_nopre": _pairwise_stats(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["nopre_cer"],
        ),
        "consensus_minus_random1": _pairwise_stats(
            per_image[f"top{k}_consensus_actual_cer"],
            per_image["random1_actual_cer"],
        ),
    }
    gaps = {
        f"consensus_minus_bestof{k}": (
            means[f"top{k}_consensus_actual_cer"] - means[f"bestof{k}_actual_cer"]
        ),
        "consensus_minus_samporacle": (
            means[f"top{k}_consensus_actual_cer"] - means["samporacle_cer"]
        ),
    }
    diagnostics = {
        f"mean_top{k}_nonempty_count": _mean(per_image[f"top{k}_nonempty_count"]),
        f"mean_top{k}_min_detected_crops": _mean(per_image[f"top{k}_min_detected_crops"]),
        f"mean_top{k}_max_detected_crops": _mean(per_image[f"top{k}_max_detected_crops"]),
        f"mean_consensus_detected_crops": _mean(per_image[f"top{k}_consensus_n_detected_crops"]),
        f"mean_consensus_text_len": _mean(per_image[f"top{k}_consensus_text_len"]),
    }
    return {
        "n_images": int(per_image[GROUP_COL].nunique()),
        "means": means,
        "wilcoxon": tests,
        "pairwise": pairwise,
        "gaps": gaps,
        "diagnostics": diagnostics,
        "model_selection": summarize_model_selection(selected_models),
        "model_selection_counts": dict(Counter(selected_models)),
        "fold_metrics": fold_metrics,
        "selected_rank_counts": {
            str(int(rank)): int(count)
            for rank, count in per_image[f"top{k}_consensus_selected_rank"].value_counts().sort_index().items()
        },
    }


def evaluate_engine(
    df: pd.DataFrame,
    engine: str,
    ocr_engines: List[str],
    input_cols: List[str],
    exact_nopre_by_image: Dict[str, float],
    image_index: Dict[str, Path],
    preprocessor,
    runner,
    cache: Dict[Tuple, Dict],
    cache_path: Path,
    k: int,
    n_folds: int,
    random_trials: int,
    cache_save_every: int,
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

    selected_models = []
    fold_metrics = []
    per_image_rows = []
    new_ocr_count = 0

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        if verbose:
            print(f"  {engine}: fold {fold_idx}/{n_folds} selecting model")

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
        best_combo_vals = train_df.groupby(PARAM_COLS)[target_col].mean().idxmin()

        for image_name, group in test_eval.groupby(GROUP_COL):
            if image_name not in exact_nopre_by_image:
                raise KeyError(f"Missing exact NoPre CER for image '{image_name}'")

            ranked = group.sort_values("pred_cer", ascending=True)
            top1 = ranked.iloc[0]
            top_k = ranked.head(min(k, len(ranked)))

            consensus_rows = []
            for rank, (_, candidate) in enumerate(top_k.iterrows(), start=1):
                before_cached = make_param_key(candidate) in cache
                ocr_record = ensure_ocr_texts(
                    candidate,
                    engines=ocr_engines,
                    image_index=image_index,
                    preprocessor=preprocessor,
                    runner=runner,
                    cache=cache,
                )
                if not before_cached:
                    new_ocr_count += 1
                    if cache_save_every > 0 and new_ocr_count % cache_save_every == 0:
                        save_cache(cache, cache_path)

                consensus_rows.append({
                    "rank": rank,
                    "ocr_text": str(ocr_record.get(f"text_{engine}", "") or ""),
                    "text_len": int(ocr_record.get(f"n_chars_{engine}", 0) or 0),
                    "n_detected_crops": int(ocr_record.get("n_detected_crops", 0) or 0),
                    "pred_cer": float(candidate["pred_cer"]),
                    "actual_cer": float(candidate[target_col]),
                    "params": {param: candidate[param] for param in PARAM_COLS},
                })

            selected, scores = consensus_select(consensus_rows)
            row = {
                "engine": engine,
                "fold": fold_idx,
                GROUP_COL: image_name,
                "top1_pred_cer": float(top1["pred_cer"]),
                "top1_actual_cer": float(top1[target_col]),
                "random1_actual_cer": float(np.mean([
                    group.sample(1, random_state=s)[target_col].iloc[0]
                    for s in range(random_trials)
                ])),
                "fixedbest_actual_cer": get_fixedbest_for_group(group, best_combo_vals, target_col),
                "nopre_cer": float(exact_nopre_by_image[image_name]),
                "samporacle_cer": float(group[target_col].min()),
                f"bestof{k}_actual_cer": float(top_k[target_col].min()),
                f"top{k}_consensus_actual_cer": float(selected["actual_cer"]),
                f"top{k}_consensus_pred_cer": float(selected["pred_cer"]),
                f"top{k}_consensus_selected_rank": int(selected["rank"]),
                f"top{k}_consensus_score": float(scores[int(selected["rank"]) - 1]),
                f"top{k}_consensus_text_len": int(selected["text_len"]),
                f"top{k}_consensus_n_detected_crops": int(selected["n_detected_crops"]),
                f"top{k}_nonempty_count": int(sum(r["text_len"] > 0 for r in consensus_rows)),
                f"top{k}_min_detected_crops": int(min(r["n_detected_crops"] for r in consensus_rows)),
                f"top{k}_max_detected_crops": int(max(r["n_detected_crops"] for r in consensus_rows)),
            }

            for param in PARAM_COLS:
                row[f"top1_{param}"] = top1[param]
                row[f"top{k}_consensus_{param}"] = selected["params"][param]

            per_image_rows.append(row)

    save_cache(cache, cache_path)
    per_image = pd.DataFrame(per_image_rows)
    summary = summarize_per_image(per_image, k, selected_models, fold_metrics)
    summary.update({
        "engine": engine,
        "per_image": per_image,
    })
    return summary


def write_report(results: Dict[str, Dict], output_dir: Path, k: int) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append(f"TOP-{k} CONSENSUS ORACLE-FREE DEPLOYABLE EVALUATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Top{k}Consensus: select the surrogate-ranked top-{k} OCR output with")
    lines.append("the smallest mean normalized edit distance to the other shortlisted outputs.")
    lines.append("Selection uses OCR outputs only; reference text and CER are used only for evaluation.")
    lines.append("")

    for engine, result in results.items():
        means = result["means"]
        diagnostics = result["diagnostics"]
        lines.append("-" * 80)
        lines.append(f"Engine: {engine}")
        lines.append(f"Images: {result['n_images']}")
        lines.append(f"Model selection: {result['model_selection']}")
        lines.append(f"Consensus selected-rank counts: {result['selected_rank_counts']}")
        lines.append("")
        lines.append("Mean CER")
        lines.append(f"  NoPre            : {_format_pct(means['nopre_cer'])}")
        lines.append(f"  FixedBest        : {_format_pct(means['fixedbest_actual_cer'])}")
        lines.append(f"  Random1          : {_format_pct(means['random1_actual_cer'])}")
        lines.append(f"  Top1Pred         : {_format_pct(means['top1_actual_cer'])}")
        lines.append(f"  Top{k}Consensus   : {_format_pct(means[f'top{k}_consensus_actual_cer'])}")
        lines.append(f"  BestOf{k} upper bd: {_format_pct(means[f'bestof{k}_actual_cer'])}")
        lines.append(f"  SampOracle       : {_format_pct(means['samporacle_cer'])}")
        lines.append("")
        lines.append("Gaps")
        lines.append(
            f"  Top{k}Consensus - BestOf{k}: "
            f"{result['gaps'][f'consensus_minus_bestof{k}'] * 100:.2f} pp"
        )
        lines.append(
            f"  Top{k}Consensus - SampOracle: "
            f"{result['gaps']['consensus_minus_samporacle'] * 100:.2f} pp"
        )
        lines.append("")
        lines.append("OCR shortlist diagnostics")
        lines.append(
            f"  Mean non-empty outputs in top-{k}: "
            f"{diagnostics[f'mean_top{k}_nonempty_count']:.2f}/{k}"
        )
        lines.append(
            f"  Mean detected crops in selected output: "
            f"{diagnostics[f'mean_consensus_detected_crops']:.2f}"
        )
        lines.append(
            f"  Mean selected text length: "
            f"{diagnostics[f'mean_consensus_text_len']:.1f} normalized chars"
        )
        lines.append("")
        lines.append(f"Wilcoxon one-sided tests (alternative: Top{k}Consensus < baseline)")
        for name in [
            "consensus_vs_top1",
            "consensus_vs_fixedbest",
            "consensus_vs_nopre",
            "consensus_vs_random1",
        ]:
            test = result["wilcoxon"][name]
            lines.append(
                f"  {name:24s}: p={test['p']:.6g}, "
                f"stat={test['stat']:.3f}, n={test['n']}"
            )
        lines.append("")
        lines.append(f"Pairwise Top{k}Consensus differences and win rates")
        lines.append(f"  (diff = Top{k}Consensus CER - baseline CER; win = consensus lower CER)")
        for name, stats_dict in result["pairwise"].items():
            baseline = name.replace("consensus_minus_", "")
            lines.append(
                f"  vs {baseline:10s}: mean={stats_dict['mean_diff'] * 100:.2f} pp, "
                f"median={stats_dict['median_diff'] * 100:.2f} pp, "
                f"win={stats_dict['win_rate'] * 100:.1f}%, "
                f"tie={stats_dict['tie_rate'] * 100:.1f}%, "
                f"loss={stats_dict['loss_rate'] * 100:.1f}%"
            )
        lines.append("")

    (output_dir / f"top{k}_consensus_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def to_jsonable_results(results: Dict[str, Dict]) -> Dict:
    serializable = {}
    for engine, result in results.items():
        serializable[engine] = {k: v for k, v in result.items() if k != "per_image"}
    return serializable


def main(args: argparse.Namespace) -> None:
    global get_models
    global load_exact_nopre_maps
    global prepare_data
    global resolve_exact_nopre_csv
    global select_best_model
    global summarize_model_selection

    helpers = load_training_helpers()
    get_models = helpers["get_models"]
    load_exact_nopre_maps = helpers["load_exact_nopre_maps"]
    prepare_data = helpers["prepare_data"]
    resolve_exact_nopre_csv = helpers["resolve_exact_nopre_csv"]
    select_best_model = helpers["select_best_model"]
    summarize_model_selection = helpers["summarize_model_selection"]

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting = infer_setting(data_path, args.setting)
    images_folder = Path(args.images_folder) if args.images_folder else default_images_folder(setting)
    cache_path = Path(args.ocr_cache) if args.ocr_cache else output_dir / f"top{args.k}_consensus_ocr_cache.csv"

    df = pd.read_csv(data_path)
    if args.limit_images:
        keep = sorted(df[GROUP_COL].unique())[:args.limit_images]
        df = df[df[GROUP_COL].isin(keep)].copy()

    input_cols = prepare_data(df)
    engines = args.engines or sorted(c.replace("cer_", "") for c in df.columns if c.startswith("cer_"))
    if setting == "sroie":
        engines = [e for e in engines if e in {"tesseract", "easyocr"}]
    if not engines:
        raise ValueError("No engines to evaluate")

    nopre_path = Path(args.nopre_path) if args.nopre_path else resolve_exact_nopre_csv(str(data_path))
    exact_maps = load_exact_nopre_maps(nopre_path, engines)

    print(f"Setting      : {setting}")
    print(f"Data         : {data_path}")
    print(f"NoPre        : {nopre_path}")
    print(f"Images       : {images_folder}")
    print(f"Engines      : {engines}")
    print(f"K            : {args.k}")
    print(f"Output       : {output_dir}")
    print(f"OCR cache    : {cache_path}")

    image_index = build_image_index(images_folder)
    cache = load_cache(cache_path)
    preprocessor, runner = init_ocr(setting, args.device, engines)

    results = {}
    all_per_image = []
    for engine in engines:
        print(f"\nEvaluating Top{args.k}Consensus for {engine}")
        result = evaluate_engine(
            df=df,
            engine=engine,
            ocr_engines=engines,
            input_cols=input_cols,
            exact_nopre_by_image=exact_maps[engine],
            image_index=image_index,
            preprocessor=preprocessor,
            runner=runner,
            cache=cache,
            cache_path=cache_path,
            k=args.k,
            n_folds=args.n_folds,
            random_trials=args.random_trials,
            cache_save_every=args.cache_save_every,
            verbose=args.verbose,
        )
        results[engine] = result
        all_per_image.append(result["per_image"])

    per_image_df = pd.concat(all_per_image, ignore_index=True)
    per_image_path = output_dir / f"top{args.k}_consensus_per_image.csv"
    metrics_path = output_dir / f"top{args.k}_consensus_metrics.json"

    per_image_df.to_csv(per_image_path, index=False)
    metrics_path.write_text(
        json.dumps(to_jsonable_results(results), indent=2),
        encoding="utf-8",
    )
    write_report(to_jsonable_results(results), output_dir, args.k)
    save_cache(cache, cache_path)

    print(f"\nReport   : {output_dir / f'top{args.k}_consensus_report.txt'}")
    print(f"Per-image: {per_image_path}")
    print(f"Metrics  : {metrics_path}")
    print(f"OCR cache: {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate oracle-free Top-K consensus selection under nested GroupKFold."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--nopre-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setting", choices=["auto", "raw", "contour", "sroie"], default="auto")
    parser.add_argument("--images-folder", default=None)
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-cache", default=None)
    parser.add_argument("--cache-save-every", type=int, default=25)
    parser.add_argument("--limit-images", type=int, default=None,
                        help="Optional smoke-test limit using the first N image names.")
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
