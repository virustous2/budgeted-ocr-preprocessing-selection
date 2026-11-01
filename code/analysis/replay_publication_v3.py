#!/usr/bin/env python3
"""Independent arithmetic replay of published numeric results, no OCR/fitting.

Runs on the sanitized bundle alone. Uses NumPy/SciPy, not project inference
helpers. Assertions cover means, signed ranks, zero counts, CIs and Holm.
This does not prove nominal CI coverage or scientific/external validity.
"""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(p):
    p=np.asarray(p);order=np.argsort(p);a=np.empty(len(p))
    a[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return a


def ci(diff,rng):
    return 100*np.quantile(diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(axis=1),[.025,.975])


def selected(scores,engine,flag='selected_by_nested_model',column='actual_cer'):
    g=scores[scores.engine.eq(engine)]
    if flag=='rank': mask=g['rank'].eq(1)
    else: mask=g[flag].astype(str).str.lower().isin(['true','1'])
    g=g[mask].set_index('image_name',verify_integrity=True).sort_index()
    assert len(g)==len(scores[scores.engine.eq(engine)].image_name.unique())
    return g[column]


def replay(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    stats=pd.read_csv(root/'evidence/bestk/statistics.csv')
    assert len(stats)==160
    for r in stats.itertuples():
        s=pd.read_csv(root/'per_page_bestk'/r.block/f'per_page_results_{r.engine}.csv').set_index('image_name').sort_index()
        n=pd.read_csv(root/'nn_baseline'/r.block/'nn_per_image.csv');n=n[n.engine.eq(r.engine)].set_index('image_name').sort_index()
        assert s.index.equals(n.index)
        v={'Surrogate':s[f'surr_k{r.k}'],'Random':s[f'rand_k{r.k}'],'FixedTopK':s[f'fixed_topk_k{r.k}'],'NN':n[f'nn_k{r.k}_cer']}
        a,b=v[r.method_a].to_numpy(),v[r.method_b].to_numpy();diff=a-b
        p=1. if np.all(diff==0) else wilcoxon(a,b,alternative='two-sided',zero_method='wilcox').pvalue
        np.testing.assert_allclose([100*diff.mean(),p],[r.difference_pp,r.p_two_sided],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(ci(diff,np.random.default_rng(r.bootstrap_seed)),[r.ci_low_pp,r.ci_high_pp],rtol=0,atol=1e-12)
    for (block,_,_),g in stats.groupby(['block','method_a','method_b']):
        assert len(g)==(8 if block=='sroie' else 12)
        np.testing.assert_allclose(holm(g.p_two_sided),g.p_holm,rtol=1e-12,atol=1e-15)
    for family,name in [('policy','strict_selector_policy_paired_comparison.csv'),('static','strict_selector_static_paired_comparison.csv')]:
        table=pd.read_csv(root/'evidence'/family/name);assert len(table)==32
        rng=np.random.default_rng(42)
        for r in table.itertuples():
            block='sroie' if r.setting=='receipt' else r.setting
            variant='rank_only' if r.analysis_set=='rank_only_sensitivity' else 'full'
            s=pd.read_csv(root/'selectors'/block/variant/'scores.csv')
            a=selected(s,r.engine)
            if family=='policy':
                t=pd.read_csv(root/'selectors'/block/('nn_k' if r.method_b=='NN-K' else 'fixed_topk')/'scores.csv')
                b=selected(t,r.engine)
            else:
                flag,column={'Top1Pred':('rank','actual_cer'),'Agreement':('agreement_selected','actual_cer'),'FixedBest':('rank','fixedbest_actual_cer'),'NoPre':('rank','nopre_cer')}[r.method_b]
                b=selected(s,r.engine,flag,column)
            assert a.index.equals(b.index)
            diff=a.to_numpy()-b.to_numpy()
            p=1. if np.all(diff==0) else wilcoxon(a,b,alternative='two-sided',zero_method='wilcox').pvalue
            assert int(sum(diff==0))==r.n_zero_diff
            np.testing.assert_allclose([100*a.mean(),100*b.mean(),100*diff.mean()],[r.surrogate_mean_cer_pct,r.baseline_mean_cer_pct,r.mean_diff_surrogate_minus_baseline_pp],rtol=0,atol=1e-12)
            np.testing.assert_allclose(p,r.p_two_sided,rtol=1e-12,atol=1e-15)
            np.testing.assert_allclose(ci(diff,rng),[r.bootstrap_95ci_low_pp,r.bootstrap_95ci_high_pp],rtol=0,atol=1e-12)
        key='analysis_set' if family=='policy' else 'comparison_family'
        for _,g in table.groupby(key):
            assert len(g)==(16 if family=='policy' else 8)
            np.testing.assert_allclose(holm(g.p_two_sided),g.p_holm_manuscript,rtol=1e-12,atol=1e-15)
    print('PASS: 224 paired means/tests/CIs; 21 complete Holm families; sanitized-bundle hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    replay(p.parse_args().bundle)
