#!/usr/bin/env python3
"""End-to-end nested surrogate -> OCR-output selector evaluation.

For outer test fold A, final selector training rows in fold B are generated
by a surrogate fitted without A or B. For selector-family validation fold C,
inner training rows in B are generated without A, C, or B; validation rows in
C are generated without A or C. Surrogate family selection is also restricted
to that surrogate's training pages. No global pooled-OOF selector training.

Stages: plan (read-only), prepare (surrogate fitting), cache (OCR only for
missing candidates), evaluate (selector fitting, full + rank-only sensitivity).
Completed surrogate jobs and selector outer folds have hash-checked checkpoints.
An interrupted active fit restarts; there is no mid-estimator checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import signal
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUP = "image_name"
PARAMS = ["gaussian_ksize", "bilateral_d", "clahe_clip", "gamma", "sharpen"]
PROTOCOL = "strict_nested_surrogate_selector_v1"
SELECTORS = ["rf_cls", "logreg_cls", "rf_reg", "gbr_reg", "ridge_reg"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    required = {"data_path", "nopre_path", "images_folder", "seed_cache", "output_dir",
                "engines", "setting", "n_folds", "seed"}
    if set(cfg) != required:
        raise ValueError(f"Config keys must be exactly {sorted(required)}")
    if cfg["n_folds"] != 5 or cfg["seed"] != 42:
        raise ValueError("This version preserves the project's five outer folds and seed 42")
    if cfg["setting"] not in {"raw", "contour", "sroie"}:
        raise ValueError("Unknown setting")
    allowed = {"tesseract", "easyocr"} if cfg["setting"] == "sroie" else {"vietocr", "tesseract", "easyocr"}
    if not cfg["engines"] or len(set(cfg["engines"])) != len(cfg["engines"]) or not set(cfg["engines"]) <= allowed:
        raise ValueError("Invalid engines")
    for key in ["data_path", "nopre_path", "images_folder", "seed_cache", "output_dir"]:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    out = Path(cfg["output_dir"])
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError("Output must be a dedicated project subdirectory")
    for key in ["data_path", "nopre_path", "seed_cache"]:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    return cfg


def assign_folds(df, n_folds=5):
    if df[GROUP].isna().any():
        raise ValueError("Missing page ID")
    out = df.copy()
    out[GROUP] = out[GROUP].astype(str)
    if out.duplicated([GROUP, *PARAMS]).any():
        raise ValueError("Duplicate page/configuration rows")
    if not (out.groupby(GROUP).size() == 50).all():
        raise ValueError("Expected exactly 50 candidates per page")
    out["candidate_id"] = np.arange(len(out))
    out["fold"] = 0
    for fold, (_, test) in enumerate(GroupKFold(n_splits=n_folds).split(out, groups=out[GROUP]), 1):
        out.loc[out.index[test], "fold"] = fold
    if not (out.groupby(GROUP)["fold"].nunique() == 1).all():
        raise AssertionError("Page split across folds")
    return out


def exclusion_sets(folds):
    return [tuple(c) for size in (1, 2, 3) for c in itertools.combinations(sorted(map(int, folds)), size)]


def job_name(engine, excluded):
    return engine + "_exclude_" + "-".join(map(str, sorted(excluded)))


def job_frames(df, excluded):
    excluded = set(excluded)
    if not excluded or not excluded < set(df.fold.unique()):
        raise ValueError("Invalid excluded folds")
    train = df[~df.fold.isin(excluded)].copy()
    test = df[df.fold.isin(excluded)].copy()
    if set(train[GROUP]) & set(test[GROUP]):
        raise AssertionError("Surrogate train/test overlap")
    if train[GROUP].nunique() < 5:
        raise ValueError("Too few training pages for surrogate family CV")
    return train, test


def train_surrogate(train, test_features, features, target):
    from sklearn.base import clone
    from main.train_surrogate_multiengine_kfold import get_models, select_best_model
    # Family CV, scalers and final fit all receive only this job's training data.
    family = select_best_model(train, target, features, verbose=True)
    model = clone(get_models()[family])
    model.fit(train[features].values, train[target].values)
    # Serial RF prediction avoids thread summation order affecting near-ties.
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=1)
    return family, model.predict(test_features[features].values)


def make_predictions(df, engine, excluded, features, nopre, fitter=train_surrogate):
    train, test = job_frames(df, excluded)
    target = "cer_" + engine
    # The fitter receives no labels or page metadata for predicted pages.
    family, predictions = fitter(train, test[features].copy(), features, target)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid surrogate predictions")
    test["pred_cer"] = predictions
    fixed = train.groupby(PARAMS)[target].mean().idxmin()
    rows, baselines = [], []
    for page, group in test.groupby(GROUP, sort=False):
        top = group.sort_values("pred_cer", kind="mergesort").head(3)
        fixed_mask = np.ones(len(group), dtype=bool)
        for key, value in zip(PARAMS, fixed):
            fixed_mask &= np.isclose(group[key].to_numpy(), value, rtol=0, atol=1e-10)
        if fixed_mask.sum() != 1:
            raise ValueError(f"FixedBest not uniquely in pool: {page}")
        for row in top.to_dict("records"):
            rows.append({"engine": engine, GROUP: page, "outer_fold": int(row["fold"]),
                         "candidate_id": int(row["candidate_id"]), "pred_cer": row["pred_cer"],
                         "actual_cer": row[target], **{p: row[p] for p in PARAMS}})
        baselines.append({"engine": engine, GROUP: page, "fold": int(group.fold.iloc[0]),
                          "fixedbest_actual_cer": float(group.loc[fixed_mask, target].iloc[0]),
                          "nopre_cer": float(nopre[page]), "samporacle_cer": float(group[target].min())})
    meta = {"engine": engine, "excluded_folds": sorted(excluded), "selected_surrogate": family,
            "train_pages": sorted(train[GROUP].unique()), "predicted_pages": sorted(test[GROUP].unique()),
            "train_folds": sorted(map(int, train.fold.unique()))}
    return pd.DataFrame(rows), pd.DataFrame(baselines), meta


def input_manifest(cfg, df):
    from evaluate_topk_consensus_deployable import build_image_index
    sources = [Path(__file__), ROOT / "main/train_surrogate_multiengine_kfold.py",
               ROOT / "main/random_baseline.py"]
    sources += [ROOT / "analysis" / name for name in [
        "evaluate_topk_consensus_deployable.py", "evaluate_oracle_free_top3_selector.py",
        "evaluate_oracle_free_top3_selector_from_predictions.py", "evaluate_oracle_free_top3_selector_nested.py"]]
    sources += [ROOT / "main/preprocessing_sampling.py"]
    if cfg["setting"] == "sroie":
        sources.append(ROOT / "SROIRE_COMPARISON/sroie_scripts/collect_lhs_sroie.py")
    else:
        sources += [ROOT / "main/collect_multiengine_data.py", ROOT / "main/vietocr_offline.py",
                    ROOT / "config/vietocr_vgg_transformer_offline.yml"]
    packages = {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"]}
    for package in ["rapidfuzz", "torch", "paddleocr", "paddlepaddle", "paddlepaddle-gpu", "easyocr", "vietocr", "pytesseract", "opencv-python"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    images = build_image_index(Path(cfg["images_folder"]))
    if set(images) != set(df[GROUP]):
        raise ValueError("Image folder and response table have different page IDs")
    return {"protocol": PROTOCOL, "config": cfg, "python": platform.python_version(),
            "packages": packages, "source_sha256": {str(p): sha(p) for p in sources},
            "input_sha256": {k: sha(cfg[k]) for k in ["data_path", "nopre_path", "seed_cache"]},
            "image_sha256": {p: sha(images[p]) for p in sorted(images)},
            "fold_pages": {str(f): sorted(g[GROUP].unique()) for f, g in df.groupby("fold")}}


def check_checkpoint(folder, manifest_id):
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    if meta["manifest_sha256"] != manifest_id:
        raise ValueError(f"Checkpoint belongs to different inputs/code: {folder}")
    for name, expected in meta["files"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {folder / name}")
    return meta


def finish_checkpoint(folder, manifest_id, names, extra=None):
    write_json(folder / "complete.json", {"manifest_sha256": manifest_id,
               "files": {name: sha(folder / name) for name in names}, **(extra or {})})


def prepare(cfg, df, out, manifest_id, max_new_jobs=None):
    from main.train_surrogate_multiengine_kfold import prepare_data, load_exact_nopre_maps
    features = prepare_data(df)
    maps = load_exact_nopre_maps(Path(cfg["nopre_path"]), cfg["engines"])
    completed_now = 0
    for engine in cfg["engines"]:
        if set(maps[engine]) != set(df[GROUP]):
            raise ValueError("NoPre and response table IDs differ")
        if not np.isfinite(list(maps[engine].values())).all():
            raise ValueError("Nonfinite NoPre CER")
        if not np.isfinite(df[features + ["cer_" + engine]].to_numpy()).all():
            raise ValueError("Nonfinite response/features")
        for excluded in exclusion_sets(df.fold.unique()):
            folder = out / "surrogate_jobs" / job_name(engine, excluded)
            if check_checkpoint(folder, manifest_id):
                print(f"RESUME: {folder.name}", flush=True)
                continue
            if max_new_jobs is not None and completed_now >= max_new_jobs:
                print("PAUSED at requested job limit; completed checkpoints are reusable.", flush=True)
                return False
            print(f"FIT: {folder.name}", flush=True)
            pred, per, meta = make_predictions(df, engine, excluded, features, maps[engine])
            write_csv(folder / "predictions.csv", pred)
            write_csv(folder / "per_image.csv", per)
            write_json(folder / "provenance.json", meta)
            finish_checkpoint(folder, manifest_id, ["predictions.csv", "per_image.csv", "provenance.json"])
            print(f"SAVED: {folder.name}", flush=True)
            completed_now += 1
    return True


def all_jobs(cfg, df, out, manifest_id):
    jobs = {}
    for engine in cfg["engines"]:
        for excluded in exclusion_sets(df.fold.unique()):
            folder = out / "surrogate_jobs" / job_name(engine, excluded)
            if not check_checkpoint(folder, manifest_id):
                raise RuntimeError(f"Run prepare first; incomplete job: {folder}")
            meta = json.loads((folder / "provenance.json").read_text())
            train, test = job_frames(df, excluded)
            if meta["excluded_folds"] != list(excluded) or set(meta["train_pages"]) != set(train[GROUP]) or set(meta["predicted_pages"]) != set(test[GROUP]):
                raise ValueError(f"Invalid train/test provenance: {folder}")
            pred = pd.read_csv(folder / "predictions.csv")
            if set(pred[GROUP]) != set(test[GROUP]) or not (pred.groupby(GROUP).size() == 3).all():
                raise ValueError(f"Incomplete predicted-page coverage: {folder}")
            if pred.duplicated([GROUP, "candidate_id"]).any() or set(pred.engine) != {engine}:
                raise ValueError(f"Duplicate candidates or wrong engine: {folder}")
            expected = df.set_index("candidate_id").loc[pred.candidate_id]
            if not np.array_equal(expected[GROUP].values, pred[GROUP].values) or not np.array_equal(expected.fold.values, pred.outer_fold.values):
                raise ValueError(f"Candidate/fold provenance mismatch: {folder}")
            np.testing.assert_allclose(expected["cer_" + engine], pred.actual_cer, rtol=0, atol=1e-12)
            np.testing.assert_allclose(expected[PARAMS], pred[PARAMS], rtol=0, atol=1e-12)
            jobs[engine, excluded] = folder
    return jobs


def cache_requirements(jobs):
    from evaluate_topk_consensus_deployable import make_param_key
    required = {}
    for (engine, _), folder in jobs.items():
        for _, row in pd.read_csv(folder / "predictions.csv").iterrows():
            required.setdefault(make_param_key(row), set()).add(engine)
    return required


def cache_complete(record, engines):
    if not record or "n_detected_crops" not in record:
        return False
    try:
        crops = float(record["n_detected_crops"])
    except (ValueError, TypeError):
        return False
    if not np.isfinite(crops) or crops < 0 or crops != int(crops):
        return False
    for name, value in record.items():
        if name.endswith("_error") and str(value).lower().strip() not in {"", "false", "0", "0.0", "none", "nan"}:
            return False
    return all(isinstance(record.get("text_" + engine), str) for engine in engines)


def save_ocr_cache(cache, path):
    # Atomic replacement leaves the previous complete cache intact if saving
    # itself is interrupted. A leftover .tmp file is never used for resume.
    if cache:
        write_csv(path, pd.DataFrame([cache[key] for key in sorted(cache, key=str)]))


def fill_cache(cfg, jobs, out):
    from evaluate_topk_consensus_deployable import (
        load_cache, init_ocr, build_image_index, ensure_ocr_texts, record_from_key)
    cache_path = out / "ocr_cache.csv"
    cache = load_cache(cache_path if cache_path.exists() else Path(cfg["seed_cache"]))
    required = cache_requirements(jobs)
    missing = [(key, sorted(engines)) for key, engines in required.items() if not cache_complete(cache.get(key), engines)]
    print(f"Required page/config keys: {len(required)}; missing: {len(missing)}", flush=True)
    if not missing:
        save_ocr_cache(cache, cache_path)
        return
    prep, runner = init_ocr(cfg["setting"], "cuda", cfg["engines"])
    # Imported collectors install handlers that do not stop this caller.
    # Restore interrupt behavior, and save the OCR cache even on interruption.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    images = build_image_index(Path(cfg["images_folder"]))
    try:
        for i, (key, engines) in enumerate(missing, 1):
            # ensure_ocr_texts treats a string-valued errored row as cached;
            # remove invalid rows in this new working cache so they are redone.
            if key in cache and not cache_complete(cache[key], engines):
                cache.pop(key)
            ensure_ocr_texts(pd.Series(record_from_key(key)), engines, images, prep, runner, cache)
            if i % 25 == 0:
                save_ocr_cache(cache, cache_path)
                print(f"OCR saved {i}/{len(missing)}", flush=True)
    finally:
        save_ocr_cache(cache, cache_path)


def load_bank(jobs, out, rank_only=False):
    from evaluate_oracle_free_top3_selector_from_predictions import build_candidates
    from evaluate_topk_consensus_deployable import load_cache
    cache_path = out / "ocr_cache.csv"
    if not cache_path.exists():
        raise RuntimeError("Run cache stage first")
    cache = load_cache(cache_path)
    for key, engines in cache_requirements(jobs).items():
        if not cache_complete(cache.get(key), engines):
            raise RuntimeError("Missing engine text/crops metadata; run cache stage first")
    bank = {}
    for (engine, excluded), folder in jobs.items():
        prediction_path = folder / "predictions.csv"
        if rank_only:
            pred = pd.read_csv(prediction_path).sort_values([GROUP, "pred_cer"], kind="mergesort")
            pred["pred_cer"] = pred.groupby(GROUP).cumcount() + 1.0
            prediction_path = out / "rank_only_inputs" / folder.name / "predictions.csv"
            write_csv(prediction_path, pred)
        candidates = build_candidates(prediction_path, cache_path, folder / "per_image.csv", [engine])
        if candidates.attrs.get("missing_cache_rows", 0):
            raise RuntimeError("Missing OCR cache")
        if not (candidates.groupby(GROUP).size() == 3).all():
            raise ValueError("Incomplete top-3")
        agreement = candidates[candidates.agreement_selected].set_index(GROUP).actual_cer
        candidates["top3_consensus_actual_cer"] = candidates[GROUP].map(agreement)
        bank[engine, excluded] = candidates
    return bank


def evaluation_candidates(bank, engine, excluded, fold):
    excluded = tuple(sorted(excluded))
    if fold not in excluded:
        raise AssertionError("Prediction fold must be excluded from upstream fitting")
    frame = bank[engine, excluded]
    result = frame[frame.fold == fold].copy()
    if result.empty:
        raise ValueError("Missing evaluation fold")
    return result


def training_candidates(bank, engine, folds, blocked):
    blocked = set(blocked)
    pieces = [evaluation_candidates(bank, engine, blocked | {fold}, fold)
              for fold in folds if fold not in blocked]
    result = pd.concat(pieces, ignore_index=True)
    if set(result.fold) & blocked:
        raise AssertionError("Blocked fold in selector training")
    return result


def score_outer(bank, engine, folds, outer, seed=42, scorer=None):
    from evaluate_oracle_free_top3_selector_from_predictions import candidate_feature_columns
    from evaluate_oracle_free_top3_selector_nested import train_scores, select_one_per_page
    scorer = scorer or train_scores
    train = training_candidates(bank, engine, folds, {outer})
    test = evaluation_candidates(bank, engine, {outer}, outer)
    features = candidate_feature_columns(train)
    inner = []
    for family in SELECTORS:
        fold_scores = []
        for valid in folds:
            if valid == outer:
                continue
            inner_train = training_candidates(bank, engine, folds, {outer, valid})
            inner_valid = evaluation_candidates(bank, engine, {outer, valid}, valid)
            if set(inner_train[GROUP]) & set(inner_valid[GROUP]):
                raise AssertionError("Selector inner train/validation overlap")
            inner_valid["selector_score"] = scorer(inner_train, inner_valid[features].copy(), features, family, seed + 1000 * outer + valid)
            if not np.isfinite(inner_valid.selector_score).all():
                raise ValueError("Nonfinite inner selector score")
            fold_scores.append(float(select_one_per_page(inner_valid).actual_cer.mean()))
        inner.append({"engine": engine, "outer_fold": outer, "selector": family,
                      "inner_valid_mean_cer": float(np.mean(fold_scores))})
    if not np.isfinite([x["inner_valid_mean_cer"] for x in inner]).all():
        raise ValueError("Nonfinite inner-validation score")
    chosen = min(inner, key=lambda x: (x["inner_valid_mean_cer"], SELECTORS.index(x["selector"])))["selector"]
    test["selector_score"] = scorer(train, test[features].copy(), features, chosen, seed + 2000 * outer)
    if not np.isfinite(test.selector_score).all():
        raise ValueError("Nonfinite selector score")
    test["selected_selector_family"] = chosen
    test["selected_by_nested_model"] = False
    indices = test.sort_values([GROUP, "selector_score", "rank"], ascending=[True, False, True]).groupby(GROUP).head(1).index
    test.loc[indices, "selected_by_nested_model"] = True
    return test, pd.DataFrame(inner)


def evaluate(cfg, df, jobs, out, manifest_id):
    from evaluate_oracle_free_top3_selector_nested import summarize_selected, write_report
    from evaluate_oracle_free_top3_selector_from_predictions import candidate_feature_columns
    folds = sorted(map(int, df.fold.unique()))
    cache_hash = sha(out / "ocr_cache.csv")
    checkpoint_id = hashlib.sha256((manifest_id + cache_hash).encode()).hexdigest()
    for variant in ["full", "rank_only"]:
        bank = load_bank(jobs, out, rank_only=variant == "rank_only")
        destination = out / variant
        all_scores, all_inner, summaries = [], [], []
        for engine in cfg["engines"]:
            scores, choices = [], []
            for outer in folds:
                folder = destination / "folds" / f"{engine}_{outer}"
                if not check_checkpoint(folder, checkpoint_id):
                    print(f"SELECTOR: {variant}/{engine}/outer={outer}", flush=True)
                    scored, inner = score_outer(bank, engine, folds, outer, cfg["seed"])
                    write_csv(folder / "scores.csv", scored)
                    write_csv(folder / "inner.csv", inner)
                    finish_checkpoint(folder, checkpoint_id, ["scores.csv", "inner.csv"])
                scored = pd.read_csv(folder / "scores.csv")
                scores.append(scored)
                choices.append(str(scored.selected_selector_family.iloc[0]))
                all_inner.append(pd.read_csv(folder / "inner.csv"))
            combined = pd.concat(scores, ignore_index=True)
            selected = combined[combined.selected_by_nested_model]
            if len(selected) != df[GROUP].nunique() or selected[GROUP].duplicated().any():
                raise AssertionError("Incomplete outer predictions")
            summary = summarize_selected(selected, combined, engine, candidate_feature_columns(combined), choices)
            summary["protocol"] = PROTOCOL
            summaries.append(summary)
            all_scores.append(combined)
            print(f"{variant}/{engine}: selector CER={summary['selector_mean']*100:.4f}%", flush=True)
        write_csv(destination / "nested_selector_candidate_scores.csv", pd.concat(all_scores, ignore_index=True))
        write_csv(destination / "nested_selector_inner_validation.csv", pd.concat(all_inner, ignore_index=True))
        write_csv(destination / "nested_selector_summary.csv", pd.DataFrame(summaries))
        write_json(destination / "nested_selector_summary.json", summaries)
        write_report(summaries, destination, 0, SELECTORS)
        finish_checkpoint(destination, checkpoint_id, ["nested_selector_candidate_scores.csv",
            "nested_selector_inner_validation.csv", "nested_selector_summary.csv",
            "nested_selector_summary.json", "nested_selector_report.txt"],
            {"protocol": PROTOCOL, "cache_sha256": cache_hash,
             "statistics_note": "Helper p-values are exploratory one-sided; submission requires separate two-sided/Holm tables."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["plan", "prepare", "cache", "evaluate"], required=True)
    parser.add_argument("--max-new-jobs", type=int, help="Prepare only: pause after N newly completed surrogate jobs")
    args = parser.parse_args()
    if args.max_new_jobs is not None and (args.stage != "prepare" or args.max_new_jobs < 1):
        parser.error("--max-new-jobs must be positive and is only valid for prepare")
    cfg = load_config(args.config)
    df = assign_folds(pd.read_csv(cfg["data_path"]), cfg["n_folds"])
    jobs_count = len(exclusion_sets(df.fold.unique())) * len(cfg["engines"])
    print(json.dumps({"protocol": PROTOCOL, "stage": args.stage, "images": df[GROUP].nunique(),
        "rows": len(df), "engines": cfg["engines"], "surrogate_jobs": jobs_count,
        "fits_per_job": "4 families x 5 CV folds + 1 refit = 21",
        "fold_pages": df.groupby("fold")[GROUP].nunique().to_dict(),
        "output": cfg["output_dir"]}, indent=2), flush=True)
    if args.stage == "plan":
        return
    out = Path(cfg["output_dir"])
    manifest = input_manifest(cfg, df)
    path = out / "input_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Inputs/code/environment changed: use a new output directory, do not mix checkpoints")
    else:
        if args.stage != "prepare":
            raise RuntimeError("Run prepare first")
        if out.exists() and any(out.iterdir()):
            raise ValueError("Refusing to use a nonempty output without a matching manifest")
        write_json(path, manifest)
        write_csv(out / "fold_assignments.csv", df[[GROUP, "fold"]].drop_duplicates())
    manifest_id = sha(path)
    complete = True
    if args.stage == "prepare":
        complete = prepare(cfg, df, out, manifest_id, args.max_new_jobs)
    else:
        jobs = all_jobs(cfg, df, out, manifest_id)
        if args.stage == "cache":
            fill_cache(cfg, jobs, out)
        else:
            evaluate(cfg, df, jobs, out, manifest_id)
    print(f"{'DONE' if complete else 'PAUSED'} stage={args.stage}: {out}", flush=True)


if __name__ == "__main__":
    main()
