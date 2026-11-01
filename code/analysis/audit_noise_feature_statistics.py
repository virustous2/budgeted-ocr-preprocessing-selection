#!/usr/bin/env python3
"""Independent arithmetic replay of A from original completed-job CER files.

Uses NumPy/SciPy primitives, not project p/CI/Holm/export helpers. Does not fit
models, validate CI coverage, prove equivalence or audit strict selectors.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {"raw": ("vietocr", "tesseract", "easyocr"),
           "contour": ("vietocr", "tesseract", "easyocr"),
           "sroie": ("tesseract", "easyocr")}
KS = (3, 5, 10, 15)
CONDITIONS = ("full", "drop_orig_noise")
CONTRASTS = (("Surrogate", "Random"), ("Surrogate", "FixedTopK"),
             ("NN", "Random"), ("NN", "FixedTopK"), ("NN", "Surrogate"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def independent_seed(engine, k, first, second):
    comparator = {"Random": "random", "FixedTopK": "fixed_topk"}.get(second, second.lower())
    if first != "Surrogate":
        comparator = first.lower() + "_vs_" + comparator
    token = f"{engine}:{k}:{comparator}".encode()
    return (12345 + sum((i+1)*v for i,v in enumerate(token))) % (2**32-1)


def independent_holm(p):
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid p-values")
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def check_row(row, x, y, seed):
    diff = x-y
    require(np.isfinite(diff).all(), "Nonfinite paired CER")
    require(row.n == len(diff) and row.n_zero == int((diff == 0).sum()), "Page/tie count mismatch")
    require(row.bootstrap_seed == seed and row.bootstrap_replicates == 10000, "Bootstrap configuration mismatch")
    require(row.alternative == "two-sided" and row.zero_method == "wilcox", "Test convention mismatch")
    require(row.ci_method == "paired_page_percentile" and not row.ci_multiplicity_adjusted,
            "CI convention mismatch")
    p = 1. if (diff == 0).all() else float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    rng = np.random.default_rng(seed)
    ci = 100*np.quantile(diff[rng.integers(len(diff), size=(10000,len(diff)))].mean(axis=1), [.025,.975])
    np.testing.assert_allclose(p, row.p_two_sided, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(100*diff.mean(), row.difference_pp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ci, [row.ci_low_pp, row.ci_high_pp], rtol=0, atol=1e-12)
    if hasattr(row, "mean_a_pct"):
        np.testing.assert_allclose([100*x.mean(),100*y.mean()], [row.mean_a_pct,row.mean_b_pct], rtol=0,atol=1e-12)
    return p


def coverage(table, keys, expected):
    require(not table.duplicated(keys).any(), "Duplicate statistical keys")
    require(set(table[keys].itertuples(index=False, name=None)) == expected, "Incomplete statistical coverage")


def audit_families(table, keys):
    count = 0
    for groupkey, g in table.groupby(keys, sort=False):
        block = groupkey[0]
        size = 8 if block == "sroie" else 12
        require(len(g) == size and g.holm_family_size.eq(size).all(), "Wrong Holm family")
        expected = independent_holm(g.replayed_p.to_numpy())
        np.testing.assert_allclose(expected, g.p_holm, rtol=1e-12, atol=1e-15)
        count += 1
    return count


def change_check(before, after, saved):
    keys = ["block", "engine", "k", "method_a", "method_b"]
    before = before.set_index(keys).sort_index()
    after = after.set_index(keys).sort_index()
    actual = saved.set_index(keys).sort_index()
    require(before.index.equals(after.index) and before.index.equals(actual.index), "Change-table alignment mismatch")
    states = []
    for suffix, source in (("a",before), ("b",after)):
        np.testing.assert_allclose(actual[f"difference_pp_{suffix}"], source.difference_pp, rtol=0,atol=1e-12)
        np.testing.assert_allclose(actual[f"p_holm_{suffix}"], source.p_holm, rtol=1e-12,atol=1e-15)
        state = np.where(source.p_holm.to_numpy() < .05, np.sign(source.difference_pp), 0)
        np.testing.assert_array_equal(actual[f"state_{suffix}"], state)
        states.append(state)
    np.testing.assert_array_equal(actual.changed, states[0] != states[1])
    return int((states[0] != states[1]).sum())


def verify(config_path):
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_dir"]
    require(not (root/".writer.lock").exists(), "Writer still active")
    dest = root / "analysis_v1"
    frozen = json.loads((root/"manifest.json").read_text())
    meta = json.loads((dest/"manifest.json").read_text())
    require(meta["prepared_manifest_sha256"] == sha(root/"manifest.json"), "Analysis provenance mismatch")
    for bucket in ("inputs", "code"):
        for path, digest in frozen[bucket].items():
            require(sha(ROOT/path) == digest, f"Frozen input/dependency changed: {path}")
    require(meta["comparisons"] == {"direct":64,"full":160,"drop_orig_noise":160}, "Comparison counts mismatch")
    for path, digest in meta["files"].items():
        require(sha(dest/path) == digest, f"Analysis file hash mismatch: {path}")
    frames, families_changed = {}, []
    for condition in CONDITIONS:
        for block, engines in ENGINES.items():
            folds = pd.read_csv(ROOT/config["blocks"][block]["folds"]).set_index("image_name").sort_index()
            for engine in engines:
                token = f"{condition}/{block}/{engine}"
                job = root/"jobs"/condition/block/engine
                require(sha(job/"complete.json") == meta["job_completion_hashes"][token], "Job marker changed")
                complete = json.loads((job/"complete.json").read_text())
                for path, digest in complete["files"].items():
                    require(sha(job/path) == digest, f"Job file hash mismatch: {path}")
                attempt = job/complete["attempt"]
                s = pd.read_csv(attempt/f"per_page_results_{engine}.csv").set_index("image_name").sort_index()
                n = pd.read_csv(attempt/"nn_per_image.csv").set_index("image_name").sort_index()
                require(s.index.is_unique and n.index.is_unique and s.index.equals(folds.index)
                        and n.index.equals(folds.index), "Original job cohort mismatch")
                np.testing.assert_array_equal(n.fold, folds.fold)
                for k in KS:
                    frames[condition,block,engine,k] = {
                        "Surrogate":s[f"surr_k{k}"].to_numpy(), "Random":s[f"rand_k{k}"].to_numpy(),
                        "FixedTopK":s[f"fixed_topk_k{k}"].to_numpy(), "NN":n[f"nn_k{k}_cer"].to_numpy()}
                    np.testing.assert_allclose(n[f"surr_k{k}_cer"],s[f"surr_k{k}"],rtol=0,atol=1e-12)
    stats = {}
    for condition in CONDITIONS:
        table = pd.read_csv(dest/condition/"statistics.csv")
        expected = {(b,e,k,x,y) for b,es in ENGINES.items() for e in es for k in KS for x,y in CONTRASTS}
        coverage(table,["block","engine","k","method_a","method_b"],expected)
        ps = []
        for row in table.itertuples():
            values = frames[condition,row.block,row.engine,row.k]
            ps.append(check_row(row,values[row.method_a],values[row.method_b],
                                independent_seed(row.engine,row.k,row.method_a,row.method_b)))
        table["replayed_p"] = ps
        require(audit_families(table,["block","method_a","method_b"]) == 15, "Wrong policy family count")
        stats[condition] = table
        print(f"PASS {condition}: 160 means/p-values/CIs, 15 Holm families", flush=True)
    direct = pd.read_csv(dest/"direct_drop_minus_full.csv")
    coverage(direct,["block","engine","k","policy"],
             {(b,e,k,p) for b,es in ENGINES.items() for e in es for k in KS for p in ("Surrogate","NN")})
    ps = []
    for row in direct.itertuples():
        x = frames["drop_orig_noise",row.block,row.engine,row.k][row.policy]
        y = frames["full",row.block,row.engine,row.k][row.policy]
        ps.append(check_row(row,x,y,independent_seed(f"{row.block}:{row.engine}",row.k,row.policy,"drop_orig_noise_vs_full")))
    direct["replayed_p"] = ps
    require(audit_families(direct,["block","policy"]) == 6, "Wrong direct family count")
    primary = pd.read_csv(ROOT/config["primary_statistics"])
    changed_full = change_check(primary,stats["full"],pd.read_csv(dest/"holm_changes_full_vs_primary.csv"))
    changed_drop = change_check(stats["full"],stats["drop_orig_noise"],pd.read_csv(dest/"holm_changes_drop_vs_full.csv"))
    significant = direct[direct.p_holm < .05].drop(columns="replayed_p").to_dict("records")
    ranges = direct.groupby(["block","policy"]).difference_pp.agg(["min","max"]).reset_index().to_dict("records")
    summary = dict(verified_utc=datetime.now(timezone.utc).isoformat(), comparisons_replayed=384,
        direct_comparisons=64, policy_comparisons_per_condition=160, holm_families_replayed=36,
        full_vs_primary_holm_changes=changed_full, drop_vs_full_holm_changes=changed_drop,
        direct_significant=significant, direct_mean_delta_ranges_pp=ranges,
        changed_policy_comparisons=pd.read_csv(dest/"holm_changes_drop_vs_full.csv").query("changed").to_dict("records"),
        hashes={"analysis_manifest":sha(dest/"manifest.json"), "prepared_manifest":sha(root/"manifest.json"),
                "verifier":sha(__file__)},
        scope="Independent arithmetic from original job CER using NumPy/SciPy; no project statistical helpers; "
              "not independent fitting/nominal CI coverage/equivalence/strict-selector robustness")
    print(f"PASS: 384 comparisons/36 families; Full changes {changed_full}, Drop changes {changed_drop}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args = p.parse_args()
    require(not args.output_dir.exists(), "Use a new evidence directory")
    summary = verify(args.config)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"verification.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
