#!/usr/bin/env python3
"""Supplementary Full vs drop_orig_noise best-of-K; no OCR or selectors.

See NOISE_FEATURE_SENSITIVITY_PROTOCOL.md. Plan/prepare do not fit models.
Run is owner-operated, with completed-engine checkpoints and retained attempts.
Existing production runners and primary outputs are never modified.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from main import train_surrogate_multiengine_kfold as surrogate
import evaluate_nn_shortlist_baseline as nn
from export_bestk_paired_statistics import ENGINES, calculate
from export_sroie_casefold_sensitivity import KS, holm, paired_ci, pvalue, seed_for

CONDITIONS = ("full", "drop_orig_noise")
PROTOCOL = ROOT / "analysis/NOISE_FEATURE_SENSITIVITY_PROTOCOL.md"
DEPENDENCIES = (
    "analysis/run_noise_feature_sensitivity.py",
    "analysis/evaluate_nn_shortlist_baseline.py",
    "main/train_surrogate_multiengine_kfold.py", "main/random_baseline.py",
    "analysis/export_bestk_paired_statistics.py",
    "analysis/export_sroie_casefold_sensitivity.py",
    "analysis/NOISE_FEATURE_SENSITIVITY_PROTOCOL.md",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def feature_lists(df, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown fixed sensitivity condition")
    page = surrogate.FEATURE_COLS_BASE + surrogate.FEATURE_COLS_NEW
    required = page + surrogate.PARAM_COLS
    if any(c not in df for c in required):
        raise ValueError("Missing fixed input features/controls")
    prepared = df.copy()
    full = surrogate.prepare_data(prepared)
    if len(full) != 22 or len(page) != 12:
        raise ValueError("Production feature schema changed; review protocol")
    # Construct the unchanged interaction BEFORE excluding the standalone column.
    cols = full if condition == "full" else [c for c in full if c != "orig_noise"]
    nncols = page if condition == "full" else [c for c in page if c != "orig_noise"]
    np.testing.assert_array_equal(prepared.inter_noise_bilateral_d,
                                  prepared.orig_noise * prepared.bilateral_d)
    return prepared, cols, nncols


def validate_data(df, nopre, folds, engines, pages):
    required = ["image_name"] + surrogate.FEATURE_COLS_BASE + surrogate.FEATURE_COLS_NEW + surrogate.PARAM_COLS + [f"cer_{e}" for e in engines]
    if any(c not in df for c in required):
        raise ValueError("Missing fixed response schema")
    if len(df) != pages * 50 or df.image_name.nunique() != pages:
        raise ValueError("Wrong response page/row counts")
    if not df.groupby("image_name").size().eq(50).all():
        raise ValueError("Require 50 rows/page")
    if df.duplicated(["image_name"] + surrogate.PARAM_COLS).any():
        raise ValueError("Duplicate page/configuration")
    combos = df.groupby(surrogate.PARAM_COLS, dropna=False).size()
    if len(combos) != 50 or not combos.eq(pages).all():
        raise ValueError("Candidate pool must be identical on all pages")
    pagecols = surrogate.FEATURE_COLS_BASE + surrogate.FEATURE_COLS_NEW
    if not df.groupby("image_name")[pagecols].nunique(dropna=False).eq(1).all().all():
        raise ValueError("Features change between candidates of a page")
    targets = [f"cer_{e}" for e in engines]
    if not np.isfinite(df[pagecols + surrogate.PARAM_COLS + targets]).all().all():
        raise ValueError("Nonfinite inputs/targets")
    if (df[targets] < 0).any().any():
        raise ValueError("Negative CER")
    if folds.image_name.duplicated().any() or len(folds) != pages:
        raise ValueError("Wrong/duplicate frozen folds")
    ids = set(df.image_name)
    expected = folds.set_index("image_name").fold
    if set(expected.index) != ids or set(expected) != set(range(1, 6)):
        raise ValueError("Frozen cohort/folds mismatch")
    for e in engines:
        if set(nopre[e]) != ids or not np.isfinite(list(nopre[e].values())).all():
            raise ValueError("NoPre cohort/nonfinite mismatch")
        if min(nopre[e].values()) < 0:
            raise ValueError("Negative NoPre CER")
    np.testing.assert_allclose(df.orig_noise**2, df.orig_blur, atol=1e-12, rtol=1e-12)
    # Assert BOTH production group constructions, not only a sorted page split.
    for frame in (df, nn.build_page_features(df, pagecols)):
        groups = frame.image_name.to_numpy()
        actual = {}
        for fold, (tr, te) in enumerate(GroupKFold(5).split(frame, groups=groups), 1):
            train = set(groups[tr]); test = set(groups[te])
            if train & test:
                raise ValueError("Page-group leakage")
            actual.update({name: fold for name in test})
        if actual != expected.to_dict():
            raise ValueError("Production GroupKFold differs from frozen folds")


def distance_ties(df, cols):
    pages = nn.build_page_features(df, cols)
    ties = 0
    for tr, te in GroupKFold(5).split(pages, groups=pages.image_name):
        scaler = StandardScaler().fit(pages.iloc[tr][cols])
        distances = pairwise_distances(scaler.transform(pages.iloc[te][cols]),
                                      scaler.transform(pages.iloc[tr][cols]), metric="euclidean")
        ties += int((np.diff(np.sort(distances, axis=1), axis=1) == 0).sum())
    return ties


def load_block(config, block):
    paths = config["blocks"][block]
    df = pd.read_csv(ROOT / paths["data"])
    folds = pd.read_csv(ROOT / paths["folds"])
    nopre = surrogate.load_exact_nopre_maps(ROOT / paths["nopre"], list(ENGINES[block]))
    return df, folds, nopre


def build_manifest(config_path):
    config = json.loads(config_path.read_text())
    if set(config) != {"protocol", "output_dir", "primary_config", "primary_statistics", "blocks"}:
        raise ValueError("Unknown/missing protocol settings")
    if config["protocol"] != "standalone_noise_bestk_sensitivity_v1" or set(config["blocks"]) != set(ENGINES):
        raise ValueError("Require fixed v1 protocol and all three blocks")
    primary = json.loads((ROOT / config["primary_config"]).read_text())
    sources = {str(config_path.resolve()): sha(config_path),
               config["primary_config"]: sha(ROOT / config["primary_config"]),
               config["primary_statistics"]: sha(ROOT / config["primary_statistics"])}
    diagnostics = {}
    features = None
    for b in ENGINES:
        paths = config["blocks"][b]
        if set(paths) != {"data", "nopre", "folds"}:
            raise ValueError("Only explicit input paths configurable")
        if paths["folds"] != primary[b]["folds"]:
            raise ValueError("Not the primary frozen folds")
        df, folds, nopre = load_block(config, b)
        validate_data(df, nopre, folds, ENGINES[b], 616 if b == "sroie" else 300)
        details = {}
        for condition in CONDITIONS:
            _, scols, ncols = feature_lists(df, condition)
            count = distance_ties(df, ncols)
            if count:
                raise ValueError(f"{b}/{condition}: {count} exact NN distance ties; review ordering before fitting")
            details[condition] = {"surrogate": scols, "nn": ncols}
        if features is not None and features != details:
            raise ValueError("Feature lists differ across blocks")
        features = details
        diagnostics[b] = {"pages": len(folds), "response_rows": len(df),
            "frozen_fold_counts": folds.fold.value_counts().sort_index().to_dict(),
            "exact_nn_distance_ties": {c: 0 for c in CONDITIONS}}
        for p in paths.values(): sources[p] = sha(ROOT / p)
        for e in ENGINES[b]:
            p = str(Path(primary[b]["surrogate"]) / f"per_page_results_{e}.csv")
            sources[p] = sha(ROOT / p)
        for key, name in (("surrogate", "metrics.json"), ("nn", "nn_per_image.csv")):
            p = str(Path(primary[b][key]) / name); sources[p] = sha(ROOT / p)
    manifest = dict(protocol=config["protocol"], change="CHG-20261012-01", inputs=sources,
        code={p: sha(ROOT / p) for p in DEPENDENCIES}, features=features, preflight=diagnostics,
        versions={p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm")},
        python=platform.python_version(),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        model_definitions={k: clean(v.get_params(deep=True)) for k, v in surrogate.get_models().items()},
        job_count=16, expected_estimator_fits=2016, primary_results_unchanged=True,
        scope="post hoc best-of-K only; no OCR/strict selector/primary promotion")
    return config, clean(manifest)


def guard(root, manifest):
    stored = json.loads((root / "manifest.json").read_text())
    if stored != manifest:
        raise ValueError("Frozen manifest changed; use a new reviewed version, do not edit old manifests")


def validate_frozen_files(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for key in ("inputs", "code"):
        for path, digest in manifest[key].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"Frozen dependency changed during run: {path}")


@contextmanager
def writer_lock(root):
    path = root / ".writer.lock"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        yield
    finally:
        path.unlink()


def completed(job):
    marker = job / "complete.json"
    if not marker.exists(): return None
    data = json.loads(marker.read_text())
    for name, digest in data["files"].items():
        if sha(job / name) != digest:
            raise ValueError(f"Completed job hash mismatch: {job / name}")
    return job / data["attempt"]


class Tee:
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, s): self.a.write(s); self.b.write(s); self.b.flush(); return len(s)
    def flush(self): self.a.flush(); self.b.flush()


def validate_results(per, nper, folds, nopre, engine):
    ids = pd.Index(sorted(folds.image_name), name="image_name")
    for frame in (per, nper):
        if frame.image_name.duplicated().any() or len(frame) != len(ids):
            raise ValueError("Wrong result coverage")
        if not frame.set_index("image_name").sort_index().index.equals(ids):
            raise ValueError("Result cohort mismatch")
    s = per.set_index("image_name").sort_index(); n = nper.set_index("image_name").sort_index()
    np.testing.assert_array_equal(n.fold, folds.set_index("image_name").sort_index().fold)
    np.testing.assert_allclose(s.no_preprocessing, [nopre[engine][i] for i in ids], atol=1e-12, rtol=0)
    cols = ["oracle", "no_preprocessing", "fixed_best"]
    for k in KS:
        cols += [f"surr_k{k}", f"rand_k{k}", f"fixed_topk_k{k}"]
        for nc, sc in [(f"surr_k{k}_cer", f"surr_k{k}"), (f"random_k{k}_cer", f"rand_k{k}"),
                       (f"fixed_topk_k{k}_cer", f"fixed_topk_k{k}")]:
            np.testing.assert_allclose(n[nc], s[sc], atol=1e-12, rtol=0)
        if not np.isfinite(n[f"nn_k{k}_cer"]).all() or (n[f"nn_k{k}_cer"] < 0).any():
            raise ValueError("Invalid NN CER")
    if not np.isfinite(s[cols]).all().all() or (s[cols] < 0).any().any():
        raise ValueError("Invalid surrogate/baseline CER")


def run_job(config, root, block, condition, engine):
    validate_frozen_files(root)
    job = root / "jobs" / condition / block / engine
    if completed(job) is not None:
        print(f"SKIP verified complete: {condition}/{block}/{engine}", flush=True)
        return False
    df, folds, nopre = load_block(config, block)
    df, scols, ncols = feature_lists(df, condition)
    attempt = job / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    started = utc(); start = time.perf_counter()
    write_json(attempt / "started.json", dict(started_utc=started, block=block, condition=condition,
        engine=engine, surrogate_features=scols, nn_features=ncols, manifest_sha256=sha(root / "manifest.json")))
    print(f"START {condition}/{block}/{engine} {started}", flush=True)
    try:
        with (attempt / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)):
            fit_start = time.perf_counter()
            result = surrogate.train_one_engine_kfold(df, f"cer_{engine}", scols, nopre[engine], 5)
            fit_seconds = time.perf_counter() - fit_start
            per = result["per_page_results"]
            nn_start = time.perf_counter()
            nper, summary = nn.evaluate_engine(df, nn.build_page_features(df, ncols),
                nn.build_combo_cer_maps(df, [engine]), engine, ncols, list(KS), 5, 20, "euclidean", per)
            nn_seconds = time.perf_counter() - nn_start
            validate_results(per, nper, folds, nopre, engine)
            # No final model pickle or OCR/reference text in sensitivity artifacts.
            per.to_csv(attempt / f"per_page_results_{engine}.csv", index=False)
            nper.to_csv(attempt / "nn_per_image.csv", index=False)
            keys = ("best_model", "selected_models_per_fold", "model_selection_summary", "fold_metrics",
                    "cv_metrics", "baselines", "fixed_topk_results", "filter_results", "random_results", "wilcoxon")
            write_json(attempt / "metrics.json", {engine: {k: result[k] for k in keys}})
            write_json(attempt / "nn_metrics.json", {engine: summary})
            write_json(attempt / "timing.json", dict(started_utc=started, finished_utc=utc(),
                surrogate_seconds=fit_seconds, nn_seconds=nn_seconds, total_seconds=time.perf_counter()-start,
                expected_estimator_fits=126, condition=condition, block=block, engine=engine))
            validate_frozen_files(root)
    except BaseException as exc:
        write_json(attempt / "failed.json", dict(utc=utc(), exception_type=type(exc).__name__,
            message=str(exc), elapsed_seconds=time.perf_counter()-start,
            resume="Restart this job from a fresh attempt; completed other jobs remain reusable"))
        raise
    files = {str(p.relative_to(job)): sha(p) for p in attempt.iterdir() if p.is_file()}
    pending = job / (".complete_" + uuid.uuid4().hex + ".json")
    write_json(pending, dict(attempt=str(attempt.relative_to(job)), files=files))
    os.replace(pending, job / "complete.json")
    print(f"DONE {condition}/{block}/{engine}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    return True


def direct_statistics(paired):
    rows = []
    for (block, engine, k), group in paired.groupby(["block", "engine", "k"], sort=True):
        for policy in ("Surrogate", "NN"):
            p = group.pivot(index="image_name", columns="condition", values=policy).sort_index()
            if p.isna().any().any(): raise ValueError("Unpaired sensitivity pages")
            a = p.drop_orig_noise.to_numpy(); b = p.full.to_numpy(); diff = a-b
            seed = seed_for(f"{block}:{engine}", k, policy, "drop_orig_noise_vs_full")
            lo, hi = paired_ci(diff, seed)
            rows.append(dict(block=block, engine=engine, k=k, policy=policy, n=len(diff),
                difference_pp=100*diff.mean(), ci_low_pp=100*lo, ci_high_pp=100*hi,
                p_two_sided=pvalue(a, b), n_zero=int((diff == 0).sum()), bootstrap_seed=seed,
                bootstrap_replicates=10000, alternative="two-sided", zero_method="wilcox",
                ci_method="paired_page_percentile", ci_multiplicity_adjusted=False))
    stats = pd.DataFrame(rows)
    for (block, _), idx in stats.groupby(["block", "policy"]).groups.items():
        size = 8 if block == "sroie" else 12
        if len(idx) != size: raise ValueError("Incomplete direct Holm family")
        stats.loc[idx, "p_holm"] = holm(stats.loc[idx, "p_two_sided"])
        stats.loc[idx, "holm_family_size"] = size
    if len(stats) != 64: raise ValueError("Require 64 direct sensitivity comparisons")
    return stats


def significance_changes(a, b):
    keys = ["block", "engine", "k", "method_a", "method_b"]
    if a.duplicated(keys).any() or b.duplicated(keys).any(): raise ValueError("Duplicate comparison keys")
    x = a[keys+["difference_pp", "p_holm"]].merge(b[keys+["difference_pp", "p_holm"]],
        on=keys, how="outer", validate="one_to_one", suffixes=("_a", "_b"), indicator=True)
    if not x._merge.eq("both").all(): raise ValueError("Comparison coverage mismatch")
    for suffix in ("a", "b"):
        x[f"state_{suffix}"] = np.where(x[f"p_holm_{suffix}"] < .05,
                                        np.sign(x[f"difference_pp_{suffix}"]), 0)
    x["changed"] = x.state_a != x.state_b
    return x.drop(columns="_merge")


def analyze(config, root):
    # Validate completion before creating anything; no fit in this stage.
    attempts = {(c, b, e): completed(root / "jobs" / c / b / e)
                for c in CONDITIONS for b in ENGINES for e in ENGINES[b]}
    if any(v is None for v in attempts.values()): raise ValueError("All 16 jobs must complete before analysis")
    dest = root / "analysis_v1"
    dest.mkdir()  # Fail on existing analysis; preserve incomplete/failed history.
    primary = json.loads((ROOT / config["primary_config"]).read_text())
    tables = {}; frames = []; reproduction = []; models = []
    for c in CONDITIONS:
        cfg = {}
        for b in ENGINES:
            sdir = dest / c / b / "surrogate"; ndir = dest / c / b / "nn"
            sdir.mkdir(parents=True); ndir.mkdir(parents=True)
            nframes = []; metrics = {}
            for e in ENGINES[b]:
                attempt = attempts[c, b, e]
                s = pd.read_csv(attempt / f"per_page_results_{e}.csv")
                n = pd.read_csv(attempt / "nn_per_image.csv")
                s.to_csv(sdir / f"per_page_results_{e}.csv", index=False); nframes.append(n)
                metrics.update(json.loads((attempt / "metrics.json").read_text()))
                if c == "full":
                    old = pd.read_csv(ROOT / primary[b]["surrogate"] / f"per_page_results_{e}.csv")
                    oldn = pd.read_csv(ROOT / primary[b]["nn"] / "nn_per_image.csv")
                    oldn = oldn[oldn.engine.eq(e)]
                    for policy, now, before, columns in (
                        ("Surrogate", s, old, [f"surr_k{k}" for k in KS]),
                        ("NN", n, oldn, [f"nn_k{k}_cer" for k in KS])):
                        now = now.set_index("image_name").sort_index(); before = before.set_index("image_name").sort_index()
                        if not now.index.equals(before.index): raise ValueError("Primary replay cohort mismatch")
                        for k, col in zip(KS, columns):
                            diff = now[col].to_numpy()-before[col].to_numpy()
                            reproduction.append(dict(block=b, engine=e, policy=policy, k=k,
                                changed_pages_exact=int((diff != 0).sum()), mean_difference_pp=100*diff.mean(),
                                max_abs_difference_pp=100*np.abs(diff).max()))
                    olds = old.set_index("image_name").sort_index(); news = s.set_index("image_name").sort_index()
                    invariant = ["oracle", "no_preprocessing", "fixed_best"] + [f"{p}_k{k}" for p in ("rand", "fixed_topk") for k in KS]
                    np.testing.assert_allclose(news[invariant], olds[invariant], atol=1e-12, rtol=0)
                    oldm = json.loads((ROOT / primary[b]["surrogate"] / "metrics.json").read_text())[e]
                    for fold, (x, y) in enumerate(zip(oldm["selected_models_per_fold"], metrics[e]["selected_models_per_fold"]), 1):
                        models.append(dict(block=b, engine=e, fold=fold, primary=x, full_rerun=y, changed=x != y))
            pd.concat(nframes, ignore_index=True).to_csv(ndir / "nn_per_image.csv", index=False)
            write_json(sdir / "metrics.json", metrics)
            cfg[b] = dict(surrogate=str(sdir), nn=str(ndir), folds=str(ROOT / config["blocks"][b]["folds"]))
        write_json(dest / c / "bestk_sources.json", cfg)
        paired, stats, baselines, sources = calculate(cfg)
        tables[c] = stats; paired["condition"] = c; frames.append(paired)
        for name, table in (("statistics.csv", stats), ("paired_values.csv", paired), ("baseline_means.csv", baselines)):
            table.to_csv(dest / c / name, index=False)
        write_json(dest / c / "source_hashes.json", sources)
    paired = pd.concat(frames, ignore_index=True)
    paired.to_csv(dest / "paired_conditions.csv", index=False)
    direct_statistics(paired).to_csv(dest / "direct_drop_minus_full.csv", index=False)
    significance_changes(tables["full"], tables["drop_orig_noise"]).to_csv(dest / "holm_changes_drop_vs_full.csv", index=False)
    significance_changes(pd.read_csv(ROOT / config["primary_statistics"]), tables["full"]).to_csv(dest / "holm_changes_full_vs_primary.csv", index=False)
    pd.DataFrame(reproduction).to_csv(dest / "full_vs_primary_perpage_drift.csv", index=False)
    pd.DataFrame(models).to_csv(dest / "full_vs_primary_model_families.csv", index=False)
    write_json(dest / "manifest.json", dict(protocol="standalone_noise_bestk_sensitivity_v1", created_utc=utc(),
        prepared_manifest_sha256=sha(root / "manifest.json"),
        job_completion_hashes={f"{c}/{b}/{e}":sha(root / "jobs" / c / b / e / "complete.json") for c,b,e in attempts},
        files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()},
        comparisons=dict(direct=64, full=160, drop_orig_noise=160),
        scope="Supplementary best-of-K only; primary and strict selector unchanged; independent audit pending"))
    print(f"Analysis written to {dest}; independent verification required before ESM use")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=("plan", "prepare", "run", "analyze"), required=True)
    p.add_argument("--block", choices=tuple(ENGINES))
    p.add_argument("--engine", choices=("vietocr", "tesseract", "easyocr"))
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--max-new-jobs", type=int)
    args = p.parse_args()
    if args.max_new_jobs is not None and args.max_new_jobs < 1: p.error("--max-new-jobs must be positive")
    if args.stage != "run" and any((args.block, args.engine, args.condition, args.max_new_jobs)):
        p.error("Job filters allowed only for run")
    config, manifest = build_manifest(args.config)
    root = ROOT / config["output_dir"]
    jobs = [(c,b,e) for c in CONDITIONS for b in ENGINES for e in ENGINES[b]
            if (not args.block or b == args.block) and (not args.engine or e == args.engine)
            and (not args.condition or c == args.condition)]
    if not jobs: p.error("No matching jobs")
    if args.stage == "plan":
        print(json.dumps(dict(stage="plan", fits_executed=0, manifest=manifest, output=str(root)), indent=2)); return
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "manifest.json", manifest)
        write_json(root / "preparation.json", dict(created_utc=utc(), fits_executed=0, jobs=jobs))
        print(f"Prepared 16 jobs, no fits: {root}"); return
    guard(root, manifest)
    with writer_lock(root):
        if args.stage == "analyze": analyze(config, root); return
        new = 0
        for c,b,e in jobs:
            new += int(run_job(config, root, b, c, e))
            if args.max_new_jobs is not None and new >= args.max_new_jobs: break
        print(f"Finished {new} new jobs; rerun stage=run to resume; primary artifacts unchanged")


if __name__ == "__main__": main()
