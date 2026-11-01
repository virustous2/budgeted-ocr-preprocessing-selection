#!/usr/bin/env python3
"""Portable arithmetic replay of the sanitized supplementary A bundle only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_noise_feature_statistics import (
    ENGINES, KS, CONDITIONS, CONTRASTS, sha, require, coverage,
    check_row, independent_seed, audit_families, change_check,
)


def replay(bundle):
    meta = json.loads((bundle/"manifest.json").read_text())
    require(meta["kind"] == "supplementary_best_of_K_not_primary", "Wrong release kind")
    for path, digest in meta["files"].items():
        require(sha(bundle/path) == digest, f"Payload hash mismatch: {path}")
    paired = pd.read_csv(bundle/"paired_conditions.csv")
    keys = ["condition","block","engine","k","image_name"]
    require(not paired.duplicated(keys).any(), "Duplicate paired pages")
    require(set(paired.condition) == set(CONDITIONS) and set(paired.block) == set(ENGINES), "Wrong conditions/blocks")
    grouped = {}
    for condition in CONDITIONS:
        for block, engines in ENGINES.items():
            block_ids = None
            for engine in engines:
                for k in KS:
                    g = paired[(paired.condition == condition)&(paired.block == block)&
                               (paired.engine == engine)&(paired.k == k)].set_index("image_name").sort_index()
                    require(len(g) == (616 if block == "sroie" else 300), "Wrong cohort count")
                    if block_ids is None:
                        block_ids = g.index
                    require(g.index.equals(block_ids), "Engine/K cohort mismatch")
                    require(set(g.fold) == set(range(1,6)), "Missing fold")
                    require(np.isfinite(g[["Surrogate","NN","Random","FixedTopK"]]).all().all(), "Nonfinite CER")
                    grouped[condition,block,engine,k] = g
    require(len(paired) == 24256, "Unexpected paired row count")
    stats = {}
    for condition in CONDITIONS:
        table = pd.read_csv(bundle/condition/"statistics.csv")
        coverage(table,["block","engine","k","method_a","method_b"],
                 {(b,e,k,a,z) for b,es in ENGINES.items() for e in es for k in KS for a,z in CONTRASTS})
        ps = []
        for row in table.itertuples():
            g = grouped[condition,row.block,row.engine,row.k]
            ps.append(check_row(row,g[row.method_a].to_numpy(),g[row.method_b].to_numpy(),
                                independent_seed(row.engine,row.k,row.method_a,row.method_b)))
        table["replayed_p"] = ps
        require(audit_families(table,["block","method_a","method_b"]) == 15,"Wrong policy families")
        stats[condition] = table
    direct = pd.read_csv(bundle/"direct_drop_minus_full.csv")
    coverage(direct,["block","engine","k","policy"],
             {(b,e,k,p) for b,es in ENGINES.items() for e in es for k in KS for p in ("Surrogate","NN")})
    ps = []
    for row in direct.itertuples():
        full = grouped["full",row.block,row.engine,row.k]
        drop = grouped["drop_orig_noise",row.block,row.engine,row.k]
        require(full.index.equals(drop.index), "Unpaired intervention")
        np.testing.assert_array_equal(full.fold,drop.fold)
        np.testing.assert_allclose(full[["Random","FixedTopK"]],drop[["Random","FixedTopK"]],rtol=0,atol=1e-12)
        ps.append(check_row(row,drop[row.policy].to_numpy(),full[row.policy].to_numpy(),
            independent_seed(f"{row.block}:{row.engine}",row.k,row.policy,"drop_orig_noise_vs_full")))
    direct["replayed_p"] = ps
    require(audit_families(direct,["block","policy"]) == 6,"Wrong direct families")
    changed = change_check(stats["full"],stats["drop_orig_noise"],pd.read_csv(bundle/"holm_changes_drop_vs_full.csv"))
    primary = pd.read_csv(bundle/"primary_statistics.csv")
    original = change_check(primary,stats["full"],pd.read_csv(bundle/"holm_changes_full_vs_primary.csv"))
    for name in ("full_vs_primary_perpage_drift.csv","full_vs_primary_model_families.csv"):
        require((bundle/name).exists(),"Missing reproduction diagnostic")
    require(original == meta["full_vs_primary_changes"] and changed == meta["drop_vs_full_changes"],"Summary count mismatch")
    print(f"PASS: 384 comparisons, 36 Holm families; Full changes {original}, Drop changes {changed}; numeric payload hashes")
    return dict(comparisons=384,holm_families=36,full_vs_primary_changes=original,drop_vs_full_changes=changed)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    replay(p.parse_args().bundle)
