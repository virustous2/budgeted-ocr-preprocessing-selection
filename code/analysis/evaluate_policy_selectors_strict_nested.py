#!/usr/bin/env python3
"""Strict nested NN-K and FixedTopK output-selection controls.

Uses the SAME blocked-fold schedule and selector-family selection as the frozen
strict surrogate runner. Policies are rebuilt from allowed training pages,
never from the incomplete neighbor columns of a pooled OOF output. NN-K uses
training-only standardized page features, all neighbors in distance order,
unique transferred training-oracle configs, then training FixedTopK fallback.
For each policy, pred_cer is a rank placeholder (1,2,3), not a CER prediction.

plan is read-only. prepare builds shortlists and fits only the NN feature scaler;
cache runs missing OCR; evaluate trains selectors. No surrogate fitting occurs.
The completed surrogate runs and their code are not modified by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

import evaluate_top3_selector_strict_nested as strict
import evaluate_nn_shortlist_baseline as nn

POLICIES = ("nn_k", "fixed_topk")
PROTOCOL = "strict_nested_policy_selectors_v2"
BASE_CACHE_COMPLETE = strict.cache_complete


def validate_data(df, engines):
    features = nn.available_feature_cols(df)
    if not np.isfinite(df[features + nn.PARAM_COLS + ["cer_" + e for e in engines]].to_numpy()).all():
        raise ValueError("Nonfinite page features, parameters or CER")
    if (df.groupby(nn.GROUP_COL)[features].nunique(dropna=False) != 1).any().any():
        raise ValueError("Page features vary across candidates of the same page")
    pools = df.groupby(nn.GROUP_COL)[nn.PARAM_COLS].apply(
        lambda g: frozenset(nn.combo_from_row(row) for _, row in g.iterrows()))
    if any(len(pool) != 50 for pool in pools) or len(set(pools)) != 1:
        raise ValueError("Expected the same 50 unique parameter keys on every page")
    return features


def make_policy_jobs(df, engine, excluded, features, nopre):
    train, test = strict.job_frames(df, excluded)
    target = "cer_" + engine
    train_pages = nn.build_page_features(train, features)
    test_pages = nn.build_page_features(test, features)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_pages[features].values)
    test_x = scaler.transform(test_pages[features].values)
    distances = pairwise_distances(test_x, train_x, metric="euclidean")
    train_names = train_pages.image_name.astype(str).tolist()
    best = nn.best_combo_by_page(train, target)
    # Preserve the baseline's max_k=15 construction, then take its first 3.
    fixed = nn.fixed_topk_combos(train, target, 15)
    fixed_best = nn.combo_from_values(train.groupby(nn.PARAM_COLS)[target].mean().idxmin())
    rows = {p: [] for p in POLICIES}
    per_image = []
    groups = dict(tuple(test.groupby("image_name", sort=False)))
    for i, page in enumerate(test_pages.image_name):
        # Same np.argsort default as the original NN baseline; ordered training
        # page IDs and NumPy version are frozen in the input manifest.
        order = np.argsort(distances[i])
        neighbor_names = [train_names[j] for j in order]
        transferred = nn.unique_preserve_order(best[name] for name in neighbor_names)
        shortlist = nn.unique_preserve_order(transferred + fixed)[:3]
        if len(shortlist) != 3:
            raise ValueError(f"Cannot form three unique NN candidates for {page}")
        origin = {}
        for neighbor_rank, position in enumerate(order, 1):
            name = train_names[position]
            origin.setdefault(best[name], (name, neighbor_rank, float(distances[i, position])))
        group = groups[page]
        lookup = {nn.combo_from_row(row): row for _, row in group.iterrows()}
        if len(lookup) != 50:
            raise ValueError(f"Duplicate parameter keys: {page}")
        per_image.append({"engine": engine, "image_name": page, "fold": int(group.fold.iloc[0]),
            "fixedbest_actual_cer": float(lookup[fixed_best][target]),
            "nopre_cer": float(nopre[page]), "samporacle_cer": float(group[target].min())})
        for policy, combos in [("nn_k", shortlist), ("fixed_topk", fixed[:3])]:
            for rank, combo in enumerate(combos, 1):
                candidate = lookup[combo]
                source = origin.get(combo) if policy == "nn_k" else None
                rows[policy].append({"engine": engine, "image_name": page,
                    "outer_fold": int(candidate.fold), "candidate_id": int(candidate.candidate_id),
                    "pred_cer": float(rank), "actual_cer": float(candidate[target]),
                    **{p: candidate[p] for p in nn.PARAM_COLS}, "policy": policy,
                    "origin": "neighbor" if source else "training_fixed",
                    "neighbor_page": source[0] if source else "",
                    "neighbor_rank": source[1] if source else 0,
                    "neighbor_distance": source[2] if source else 0.0})
    meta = {"engine": engine, "excluded_folds": list(map(int, excluded)),
            "train_folds": sorted(map(int, train.fold.unique())),
            "train_pages": sorted(train.image_name.unique()),
            "predicted_pages": sorted(test.image_name.unique()),
            "features": features, "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(), "distance_metric": "euclidean",
            "rank_features": "pred_cer is shortlist rank, not predicted CER"}
    return {policy: pd.DataFrame(records) for policy, records in rows.items()}, pd.DataFrame(per_image), meta


def job_folder(out, policy, engine, excluded):
    return out / policy / "policy_jobs" / strict.job_name(engine, excluded)


def prepare(cfg, df, out, manifest_id):
    from main.train_surrogate_multiengine_kfold import load_exact_nopre_maps
    features = validate_data(df, cfg["engines"])
    maps = load_exact_nopre_maps(Path(cfg["nopre_path"]), cfg["engines"])
    for engine in cfg["engines"]:
        if set(maps[engine]) != set(df.image_name) or not np.isfinite(list(maps[engine].values())).all():
            raise ValueError("Invalid NoPre page coverage/CER")
        for excluded in strict.exclusion_sets(df.fold.unique()):
            folders = {p: job_folder(out, p, engine, excluded) for p in POLICIES}
            complete = {p: strict.check_checkpoint(folder, manifest_id) for p, folder in folders.items()}
            if all(complete.values()):
                print(f"RESUME {engine} exclude={excluded}", flush=True)
                continue
            rows, per, meta = make_policy_jobs(df, engine, excluded, features, maps[engine])
            for policy, folder in folders.items():
                if complete[policy]:
                    continue
                strict.write_csv(folder / "predictions.csv", rows[policy])
                strict.write_csv(folder / "per_image.csv", per)
                strict.write_json(folder / "provenance.json", {**meta, "policy": policy})
                strict.finish_checkpoint(folder, manifest_id, ["predictions.csv", "per_image.csv", "provenance.json"])
            print(f"SAVED both policies: {engine} exclude={excluded}", flush=True)


def load_jobs(cfg, df, out, manifest_id):
    result = {p: {} for p in POLICIES}
    indexed = df.set_index("candidate_id")
    for engine in cfg["engines"]:
        for excluded in strict.exclusion_sets(df.fold.unique()):
            train, test = strict.job_frames(df, excluded)
            for policy in POLICIES:
                folder = job_folder(out, policy, engine, excluded)
                if not strict.check_checkpoint(folder, manifest_id):
                    raise RuntimeError(f"Run prepare first: {folder}")
                meta = json.loads((folder / "provenance.json").read_text())
                if meta["policy"] != policy or meta["engine"] != engine or meta["excluded_folds"] != list(excluded):
                    raise ValueError("Wrong policy/engine/exclusion provenance")
                if set(meta["train_pages"]) != set(train.image_name) or set(meta["predicted_pages"]) != set(test.image_name):
                    raise ValueError("Upstream train/test provenance mismatch")
                if meta["train_folds"] != sorted(map(int, train.fold.unique())):
                    raise ValueError("Upstream training-fold provenance mismatch")
                pred = pd.read_csv(folder / "predictions.csv", keep_default_na=False)
                if set(pred.image_name) != set(test.image_name) or set(pred.engine) != {engine} or set(pred.policy) != {policy}:
                    raise ValueError("Predicted page/engine coverage mismatch")
                if pred.duplicated(["image_name", "candidate_id"]).any():
                    raise ValueError("Duplicate shortlisted candidate")
                if not pred.groupby("image_name").pred_cer.apply(lambda x: list(x) == [1., 2., 3.]).all():
                    raise ValueError("Invalid rank-only policy scores")
                reference = indexed.loc[pred.candidate_id]
                np.testing.assert_array_equal(reference.image_name.values, pred.image_name.values)
                np.testing.assert_array_equal(reference.fold.values, pred.outer_fold.values)
                np.testing.assert_allclose(reference[nn.PARAM_COLS], pred[nn.PARAM_COLS], rtol=0, atol=1e-12)
                np.testing.assert_allclose(reference["cer_" + engine], pred.actual_cer, rtol=0, atol=1e-12)
                transferred = pred[pred.origin == "neighbor"]
                if not set(transferred.neighbor_page) <= set(train.image_name):
                    raise ValueError("Neighbor outside allowed training pages")
                if policy == "fixed_topk" and len(transferred):
                    raise ValueError("FixedTopK row has a neighbor origin")
                if not set(pred.origin) <= {"neighbor", "training_fixed"}:
                    raise ValueError("Unknown candidate origin")
                per = pd.read_csv(folder / "per_image.csv")
                if per.image_name.duplicated().any() or set(per.image_name) != set(test.image_name) or set(per.engine) != {engine}:
                    raise ValueError("Baseline page/engine coverage mismatch")
                expected_folds = test.groupby("image_name").fold.first().loc[per.image_name]
                np.testing.assert_array_equal(expected_folds, per.fold)
                if not np.isfinite(per[["fixedbest_actual_cer", "nopre_cer", "samporacle_cer"]].to_numpy()).all():
                    raise ValueError("Nonfinite baseline CER")
                result[policy][engine, excluded] = folder
    return result


def cache_jobs(jobs):
    # strict.cache_requirements only uses the engine part of each key. Preserve
    # both policies' jobs without overwriting same-engine/exclusion dictionary keys.
    return {(engine, (policy, *excluded)): folder for policy, group in jobs.items()
            for (engine, excluded), folder in group.items()}


def policy_cache_complete(record, engines):
    """Reject rectangular-CSV blanks for engines that were never actually run.

    A genuine empty OCR result has text="" AND n_chars=0. A missing engine in a
    heterogeneous cache is loaded by pandas as text="" AND n_chars=""; checking
    only that text is a string incorrectly treats the latter as complete.
    """
    from evaluate_topk_consensus_deployable import _norm_text
    if not BASE_CACHE_COMPLETE(record, engines):
        return False
    for engine in engines:
        try:
            n_chars = float(record.get("n_chars_" + engine))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(n_chars) or n_chars < 0 or n_chars != int(n_chars):
            return False
        if int(n_chars) != len(_norm_text(record["text_" + engine])):
            return False
    return True


def fill_cache(cfg, jobs, out):
    # Reuse the frozen implementation, replacing only its module-level validity
    # predicate for this call. Restore it even if OCR is interrupted.
    original = strict.cache_complete
    strict.cache_complete = policy_cache_complete
    try:
        strict.fill_cache(cfg, cache_jobs(jobs), out)
    finally:
        strict.cache_complete = original


def validate_cache(jobs, out):
    from evaluate_topk_consensus_deployable import load_cache
    cache_path = out / "ocr_cache.csv"
    if not cache_path.exists():
        raise RuntimeError("Run cache stage first")
    cache = load_cache(cache_path)
    missing = [(key, sorted(engines)) for key, engines in strict.cache_requirements(cache_jobs(jobs)).items()
               if not policy_cache_complete(cache.get(key), engines)]
    if missing:
        raise RuntimeError(f"Incomplete OCR cache ({len(missing)} keys); run cache stage")


def evaluate(cfg, df, jobs, out, manifest_id):
    from evaluate_oracle_free_top3_selector_from_predictions import candidate_feature_columns
    from evaluate_oracle_free_top3_selector_nested import summarize_selected, write_report
    validate_cache(jobs, out)
    cache_hash = strict.sha(out / "ocr_cache.csv")
    checkpoint_id = hashlib.sha256((manifest_id + cache_hash).encode()).hexdigest()
    folds = sorted(map(int, df.fold.unique()))
    for policy in POLICIES:
        # No second rank-only ablation: these policies already supply ranks.
        bank = strict.load_bank(jobs[policy], out, rank_only=False)
        dest = out / policy
        all_scores, all_inner, summaries = [], [], []
        for engine in cfg["engines"]:
            pieces, choices = [], []
            for fold in folds:
                folder = dest / "folds" / f"{engine}_{fold}"
                if not strict.check_checkpoint(folder, checkpoint_id):
                    print(f"SELECTOR {policy}/{engine}/outer={fold}", flush=True)
                    scores, inner = strict.score_outer(bank, engine, folds, fold, cfg["seed"])
                    strict.write_csv(folder / "scores.csv", scores)
                    strict.write_csv(folder / "inner.csv", inner)
                    strict.finish_checkpoint(folder, checkpoint_id, ["scores.csv", "inner.csv"])
                scores = pd.read_csv(folder / "scores.csv")
                pieces.append(scores)
                choices.append(str(scores.selected_selector_family.iloc[0]))
                all_inner.append(pd.read_csv(folder / "inner.csv"))
            combined = pd.concat(pieces, ignore_index=True)
            selected = combined[combined.selected_by_nested_model]
            if len(selected) != df.image_name.nunique() or selected.image_name.duplicated().any():
                raise ValueError("Incomplete final selector predictions")
            summary = summarize_selected(selected, combined, engine, candidate_feature_columns(combined), choices)
            summary.update(policy=policy, protocol=PROTOCOL)
            summaries.append(summary)
            all_scores.append(combined)
            print(f"{policy}/{engine}: CER={summary['selector_mean']*100:.4f}%", flush=True)
        strict.write_csv(dest / "nested_selector_candidate_scores.csv", pd.concat(all_scores, ignore_index=True))
        strict.write_csv(dest / "nested_selector_inner_validation.csv", pd.concat(all_inner, ignore_index=True))
        strict.write_csv(dest / "nested_selector_summary.csv", pd.DataFrame(summaries))
        strict.write_json(dest / "nested_selector_summary.json", summaries)
        write_report(summaries, dest, 0, strict.SELECTORS)
        strict.finish_checkpoint(dest, checkpoint_id, ["nested_selector_candidate_scores.csv",
            "nested_selector_inner_validation.csv", "nested_selector_summary.csv",
            "nested_selector_summary.json", "nested_selector_report.txt"],
            {"protocol": PROTOCOL, "cache_sha256": cache_hash,
             "statistics_note": "Summary one-sided p-values are not final submission two-sided/Holm results."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["plan", "prepare", "cache", "evaluate"], required=True)
    args = parser.parse_args()
    cfg = strict.load_config(args.config)
    df = strict.assign_folds(pd.read_csv(cfg["data_path"]))
    validate_data(df, cfg["engines"])
    out = Path(cfg["output_dir"])
    print(json.dumps({"protocol": PROTOCOL, "stage": args.stage, "images": df.image_name.nunique(),
        "engines": cfg["engines"], "policies": list(POLICIES),
        "upstream_policy_jobs": 2 * 25 * len(cfg["engines"]), "surrogate_fits": 0,
        "output": str(out)}, indent=2), flush=True)
    if args.stage == "plan":
        return
    manifest = strict.input_manifest(cfg, df)
    manifest.update(protocol=PROTOCOL, policies=list(POLICIES))
    manifest["source_sha256"].update({str(Path(__file__).resolve()): strict.sha(Path(__file__)),
                                    str(Path(nn.__file__).resolve()): strict.sha(nn.__file__)})
    path = out / "input_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Inputs/code/environment changed; choose a new output directory")
    else:
        if args.stage != "prepare":
            raise RuntimeError("Run prepare first")
        if out.exists() and any(out.iterdir()):
            raise ValueError("Refusing nonempty output directory without matching manifest")
        strict.write_json(path, manifest)
        strict.write_csv(out / "fold_assignments.csv", df[["image_name", "fold"]].drop_duplicates())
    manifest_id = strict.sha(path)
    if args.stage == "prepare":
        prepare(cfg, df, out, manifest_id)
    else:
        jobs = load_jobs(cfg, df, out, manifest_id)
        if args.stage == "cache":
            fill_cache(cfg, jobs, out)
        else:
            evaluate(cfg, df, jobs, out, manifest_id)
    print(f"DONE stage={args.stage}: {out}", flush=True)


if __name__ == "__main__":
    main()
