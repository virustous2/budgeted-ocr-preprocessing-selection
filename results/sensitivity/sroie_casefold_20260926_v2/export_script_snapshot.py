#!/usr/bin/env python3
"""Export SROIE metric sensitivity from saved per-page results; no OCR or fitting."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ENGINES = ("tesseract", "easyocr")
KS = (3, 5, 10, 15)
CONTRASTS = (("Surrogate", "Random"), ("Surrogate", "FixedTopK"),
             ("NN", "Random"), ("NN", "FixedTopK"), ("NN", "Surrogate"))
METRICS = ("primary", "casefold")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def seed_for(engine, k, a, b):
    # Same seed rule as bootstrap_shortlist_ci.py for Surrogate contrasts.
    comparator = {"Random": "random", "FixedTopK": "fixed_topk"}.get(b, b.lower())
    if a != "Surrogate":
        comparator = a.lower() + "_vs_" + comparator
    token = f"{engine}:{k}:{comparator}".encode()
    return (12345 + sum((i + 1) * v for i, v in enumerate(token))) % (2**32 - 1)


def paired_ci(diff, seed, n=10000):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def pvalue(a, b, zero="wilcox"):
    return 1.0 if np.all(np.asarray(a) == np.asarray(b)) else float(
        wilcoxon(a, b, alternative="two-sided", zero_method=zero).pvalue)


def aligned(frame, ids=None):
    if len(frame) != 616 or frame.image_name.nunique() != 616:
        raise ValueError("Expected 616 unique pages")
    frame = frame.set_index("image_name").sort_index()
    if ids is not None and list(frame.index) != list(ids):
        raise ValueError("Page-set mismatch")
    return frame


def calculate(inputs):
    records, baselines = [], []
    ids = None
    reference_folds = None
    for metric in METRICS:
        nn = pd.read_csv(inputs / f"{metric}_nn.csv")
        if len(nn) != 1232 or set(nn.engine) != set(ENGINES):
            raise ValueError("Unexpected NN cohort/engines")
        for engine in ENGINES:
            s = aligned(pd.read_csv(inputs / f"{metric}_{engine}.csv"), ids)
            ids = s.index
            n = aligned(nn[nn.engine == engine].copy(), ids)
            if reference_folds is None:
                reference_folds = n.fold
                if n.fold.value_counts().sort_index().tolist() != [124, 123, 123, 123, 123]:
                    raise ValueError("Unexpected fold sizes")
                if set(n.fold) != set(range(1, 6)):
                    raise ValueError("Unexpected fold labels")
            pd.testing.assert_series_equal(n.fold, reference_folds)
            for name in ("no_preprocessing", "oracle", "fixed_best"):
                values = s[name].to_numpy()
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid baseline CER")
                baselines.append(dict(metric=metric, engine=engine, policy=name, mean_cer_pct=100*values.mean()))
            for k in KS:
                for nn_col, s_col in [(f"surr_k{k}_cer", f"surr_k{k}"),
                                     (f"random_k{k}_cer", f"rand_k{k}"),
                                     (f"fixed_topk_k{k}_cer", f"fixed_topk_k{k}")]:
                    np.testing.assert_allclose(n[nn_col], s[s_col], atol=1e-12, rtol=0)
                vals = {"Surrogate": s[f"surr_k{k}"].to_numpy(),
                        "Random": s[f"rand_k{k}"].to_numpy(),
                        "FixedTopK": s[f"fixed_topk_k{k}"].to_numpy(),
                        "NN": n[f"nn_k{k}_cer"].to_numpy()}
                for policy, values in vals.items():
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError(f"Invalid CER: {policy}")
                for i, page in enumerate(ids):
                    records.append(dict(metric=metric, engine=engine, k=k, image_name=page,
                                        fold=int(n.loc[page, "fold"]), **{p: float(v[i]) for p, v in vals.items()}))
    paired = pd.DataFrame(records)
    rows, zeros = [], []
    for (metric, engine, k), g in paired.groupby(["metric", "engine", "k"], sort=True):
        for a, b in CONTRASTS:
            x, y = g[a].to_numpy(), g[b].to_numpy()
            d = x-y
            seed = seed_for(engine, k, a, b)
            lo, hi = paired_ci(d, seed)
            rows.append(dict(metric=metric, engine=engine, k=k, method_a=a, method_b=b,
                             n=616, mean_a_pct=100*x.mean(), mean_b_pct=100*y.mean(),
                             difference_pp=100*d.mean(), ci_low_pp=100*lo, ci_high_pp=100*hi,
                             p_two_sided=pvalue(x,y), n_zero=int((d==0).sum()),
                             bootstrap_seed=seed, bootstrap_replicates=10000,
                             ci_method="paired_page_percentile", ci_multiplicity_adjusted=False,
                             zero_method="wilcox"))
        if metric == "casefold":
            for method in ("wilcox", "pratt", "zsplit"):
                zeros.append(dict(engine=engine,k=k,zero_method=method,
                                  p_two_sided=pvalue(g.Surrogate,g.FixedTopK,method)))
    summary = pd.DataFrame(rows)
    for _, idx in summary.groupby(["metric", "method_a", "method_b"]).groups.items():
        if len(idx) != 8:
            raise ValueError("Incomplete Holm family")
        summary.loc[idx, "p_holm"] = holm(summary.loc[idx,"p_two_sided"])
    zero_frame = pd.DataFrame(zeros)
    for _, idx in zero_frame.groupby("zero_method").groups.items():
        zero_frame.loc[idx,"p_holm"] = holm(zero_frame.loc[idx,"p_two_sided"])
    # Prevent accidental departure from the primary publication's test convention.
    official = pd.read_csv(inputs / "primary_statistics.csv")
    for r in summary[summary.metric=="primary"].itertuples():
        matched = official[(official.dataset=="SROIE") & (official.engine==r.engine) &
                           (official.k==r.k) & (official.method_a==r.method_a) &
                           (official.method_b==r.method_b)]
        if len(matched) != 1:
            raise ValueError("Missing/ambiguous primary statistic")
        np.testing.assert_allclose([r.p_two_sided,r.p_holm],
                                   matched[["p_two_sided","p_holm"]].iloc[0],rtol=1e-10,atol=1e-15)
    return paired, summary, pd.DataFrame(baselines), zero_frame


def comparison(summary):
    key = ["method_a","method_b","engine","k"]
    cols = ["mean_a_pct","mean_b_pct","difference_pp","ci_low_pp","ci_high_pp","p_two_sided","p_holm"]
    frames = [summary[summary.metric==m][key+cols].rename(columns={c:f"{m}_{c}" for c in cols}) for m in METRICS]
    return frames[0].merge(frames[1],on=key,validate="one_to_one")


def tables(summary):
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{SROIE best-of-$K$: Holm-significant lower-CER counts. Each metric and policy contrast has its own eight-test family; A/B denotes which policy has lower mean CER.}",
             r"\label{tab:sroie_casefold_counts}", r"\small",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Contrast (A vs B) & Primary A/B & Casefold A/B \\", r"\midrule"]
    for a,b in CONTRASTS:
        counts=[]
        for metric in METRICS:
            g=summary[(summary.metric==metric)&(summary.method_a==a)&(summary.method_b==b)]
            sig=g.p_holm<.05
            counts.append(f"{int((sig & (g.difference_pp<0)).sum())}/{int((sig & (g.difference_pp>0)).sum())}")
        lines.append(f"{a.replace('NN','NN-K')} vs {b.replace('NN','NN-K')} & {counts[0]} & {counts[1]} " + r"\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}","",
              r"\begin{table}[H]",r"\centering",
              r"\caption{NN-K minus Surrogate on SROIE (CER percentage points). CS: primary case-sensitive metric; CF: casefold. CIs are paired percentile bootstrap 95\% intervals per comparison, not multiplicity-adjusted; $p_H$ is Holm-adjusted separately within each metric's eight-test family.}",
              r"\label{tab:sroie_casefold_nn}",r"\small",r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{lr r r r r r r}",r"\toprule",
              r"Engine & $K$ & CS $\Delta$ & CS 95\% CI & CS $p_H$ & CF $\Delta$ & CF 95\% CI & CF $p_H$ \\",r"\midrule"]
    wide=comparison(summary)
    g=wide[(wide.method_a=="NN")&(wide.method_b=="Surrogate")].set_index(["engine","k"])
    for engine in ENGINES:
        for k in KS:
            r=g.loc[(engine,k)]
            vals=[]
            for m in METRICS:
                vals += [f"{r[m+'_difference_pp']:+.2f}",
                         f"[{r[m+'_ci_low_pp']:+.3g}, {r[m+'_ci_high_pp']:+.3g}]",
                         f"{r[m+'_p_holm']:.3g}"]
            lines.append(f"{'Tesseract' if engine=='tesseract' else 'EasyOCR'} & {k} & "+" & ".join(vals)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}%",r"}",r"\end{table}",""]
    return "\n".join(lines)


def write_outputs(inputs, out):
    inputs, out=inputs.resolve(),out.resolve()
    if inputs == out or inputs in out.parents or out in inputs.parents:
        raise ValueError("Use separate sibling input/output directories")
    files=sorted(inputs.glob("*.csv"))
    if len(files)!=7:
        raise ValueError("Expected exactly seven input CSV files")
    frozen={p.name:sha(p) for p in files}
    paired, summary, baseline, zero=calculate(inputs)
    buffers={"paired_values.csv":paired.to_csv(index=False),
             "statistics.csv":summary.to_csv(index=False),
             "primary_casefold_comparison.csv":comparison(summary).to_csv(index=False),
             "baseline_means.csv":baseline.to_csv(index=False),
             "zero_method_sensitivity.csv":zero.to_csv(index=False),
             "sroie_casefold_tables.tex":tables(summary)}
    metadata={"change_id":"CHG-20260926-13","scope":"post_hoc_SROIE_best_of_K_only",
              "inputs_sha256":frozen,"script_sha256":sha(__file__),
              "versions":{p:version(p) for p in ("numpy","pandas","scipy")},
              "normalization":"NFC + whitespace collapse; CF additionally casefolds prediction and reference",
              "bootstrap":"10000 paired-page percentile draws; seed rule in seed_for; per-comparison 95%, not simultaneous",
              "holm":"8 tests per metric and policy contrast; reverse duplicates excluded",
              "limitations":"No test of metric-by-policy interaction; no selector sensitivity; saved OOF surrogate rankings not reconstructed",
              "counts":{"paired_rows":len(paired),"statistical_rows":len(summary),"comparisons":40},
              "outputs_sha256":{n:hashlib.sha256(v.encode()).hexdigest() for n,v in buffers.items()}}
    buffers["manifest.json"]=json.dumps(metadata,indent=2,sort_keys=True)+"\n"
    if frozen!={p.name:sha(p) for p in files}:
        raise ValueError("Inputs changed during calculation")
    out.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in out.iterdir()):
        raise ValueError("Symlink output refused")
    if any(p.name not in buffers for p in out.iterdir()):
        raise ValueError("Foreign output directory")
    for name,text in buffers.items():
        p=out/name
        if p.exists() and p.read_bytes()!=text.encode():
            raise ValueError("Refusing to overwrite differing artifact: "+str(p))
    for name,text in buffers.items():
        p=out/name
        if not p.exists():
            p.write_text(text,encoding="utf-8")
    print(json.dumps(metadata["counts"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    write_outputs(args.inputs,args.output_dir)

