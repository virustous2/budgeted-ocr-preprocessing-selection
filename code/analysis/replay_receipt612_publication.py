"""Portable independent numeric replay: corrected primary, A and letter case.

Only numeric bundle, NumPy/pandas/SciPy and two independent replay helper modules
are needed. No OCR, model fitting, project source data or absolute paths.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from replay_publication_v3 import replay as replay_primary
from audit_noise_feature_statistics import (ENGINES, KS, CONTRASTS, sha, require,
    check_row, independent_seed, audit_families, coverage, change_check, independent_holm)

COUNTS={'raw':300,'contour':300,'sroie':612}


def original_values(bundle, condition, block, engine, k):
    source=bundle/'sensitivity/input_pages'/condition/block
    s=pd.read_csv(source/f'per_page_results_{engine}.csv').set_index('image_name').sort_index()
    nn=pd.read_csv(source/'nn_per_image.csv');nn=nn[nn.engine.eq(engine)].set_index('image_name').sort_index()
    folds=pd.read_csv(bundle/'response'/block/'fold_assignments.csv').set_index('image_name').sort_index()
    require(len(s)==COUNTS[block] and s.index.is_unique and nn.index.is_unique,'Wrong/duplicate cohort')
    require(s.index.equals(nn.index) and s.index.equals(folds.index),'Unpaired sensitivity source')
    np.testing.assert_array_equal(nn.fold,folds.fold)
    return {"Surrogate":s[f'surr_k{k}'].to_numpy(),"Random":s[f'rand_k{k}'].to_numpy(),
        "FixedTopK":s[f'fixed_topk_k{k}'].to_numpy(),"NN":nn[f'nn_k{k}_cer'].to_numpy()}


def replay(bundle):
    meta=json.loads((bundle/'manifest.json').read_text())
    require(meta['kind']=='corrected_receipt612_publication' and meta['counts']==COUNTS,'Wrong cohort bundle')
    replay_primary(bundle)
    for block,engines in ENGINES.items():
        folds=pd.read_csv(bundle/'response'/block/'fold_assignments.csv')
        require(len(folds)==COUNTS[block] and folds.image_name.nunique()==COUNTS[block],'Wrong fold coverage')
        response=pd.read_csv(bundle/'response'/block/'preprocessing_data.csv')
        require(len(response)==50*COUNTS[block] and response.groupby('image_name').size().eq(50).all(),'Wrong response coverage')
        ids=set(response.image_name.map(lambda x:Path(x).stem))
        require(ids==set(folds.image_name),'Response/fold identity mismatch')
        for e in engines:
            s=pd.read_csv(bundle/'per_page_bestk'/block/f'per_page_results_{e}.csv')
            require(len(s)==COUNTS[block] and s.image_name.nunique()==COUNTS[block] and set(s.image_name)==ids,'Wrong primary cohort')
        for variant in ['full','rank_only','nn_k','fixed_topk']:
            scores=pd.read_csv(bundle/'selectors'/block/variant/'scores.csv')
            require(len(scores)==3*COUNTS[block]*len(engines) and set(scores.engine)==set(engines),'Wrong selector coverage')
            require('ocr_text' not in scores and scores.groupby(['engine','image_name']).size().eq(3).all(),'Text/selector coverage')
            for e,g in scores.groupby('engine'):require(set(g.image_name)==ids,'Wrong selector identities')
    frames={};tables={}
    for condition,source in [('full','a_full'),('drop_orig_noise','drop_orig_noise')]:
        table=pd.read_csv(bundle/'sensitivity/A'/condition/'statistics.csv');ps=[]
        coverage(table,['block','engine','k','method_a','method_b'],
            {(b,e,k,a,c) for b,es in ENGINES.items() for e in es for k in KS for a,c in CONTRASTS})
        for b,es in ENGINES.items():
            for e in es:
                for k in KS:frames[condition,b,e,k]=original_values(bundle,source,b,e,k)
        for r in table.itertuples():
            v=frames[condition,r.block,r.engine,r.k]
            ps.append(check_row(r,v[r.method_a],v[r.method_b],independent_seed(r.engine,r.k,r.method_a,r.method_b)))
        table['replayed_p']=ps;require(audit_families(table,['block','method_a','method_b'])==15,'Wrong A families')
        tables[condition]=table
    direct=pd.read_csv(bundle/'sensitivity/A/direct_drop_minus_full.csv');ps=[]
    coverage(direct,['block','engine','k','policy'],{(b,e,k,p) for b,es in ENGINES.items() for e in es for k in KS for p in ['Surrogate','NN']})
    for r in direct.itertuples():
        x=frames['drop_orig_noise',r.block,r.engine,r.k];y=frames['full',r.block,r.engine,r.k]
        for p in ['Random','FixedTopK']:np.testing.assert_allclose(x[p],y[p],rtol=0,atol=1e-12)
        ps.append(check_row(r,x[r.policy],y[r.policy],independent_seed(f'{r.block}:{r.engine}',r.k,r.policy,'drop_orig_noise_vs_full')))
    direct['replayed_p']=ps;require(audit_families(direct,['block','policy'])==6,'Wrong direct families')
    changed=change_check(tables['full'],tables['drop_orig_noise'],pd.read_csv(bundle/'sensitivity/A/holm_changes_drop_vs_full.csv'))
    primary=pd.read_csv(bundle/'evidence/bestk/statistics.csv')
    unchanged=change_check(primary,tables['full'],pd.read_csv(bundle/'sensitivity/A/holm_changes_full_vs_primary.csv'))
    require(unchanged==0,'A Full differs from primary classifications')
    metric=pd.read_csv(bundle/'sensitivity/letter_case/statistics.csv')
    require(len(metric)==80,'Incomplete metric sensitivity')
    ps=[]
    for r in metric.itertuples():
        if r.metric=='case_sensitive':v=original_values(bundle,'case_sensitive','sroie',r.engine,r.k)
        elif r.metric=='casefold':v=frames['full','sroie',r.engine,r.k]
        else:raise ValueError('Unknown metric')
        ps.append(check_row(r,v[r.method_a],v[r.method_b],independent_seed(r.engine,r.k,r.method_a,r.method_b)))
    metric['replayed_p']=ps
    for _,g in metric.groupby('metric'):
        require(audit_families(g,['block','method_a','method_b'])==5,'Wrong metric families')
    # All headroom coefficients, not just plotted signs/ranges.
    for name,column,key in [('headroom_policy_summary_allK.csv','spearman_headroom_vs_advantage','policy'),
            ('headroom_nonshared_controls_allK.csv','spearman_headroom_vs_gain','comparison')]:
        table=pd.read_csv(bundle/'evidence/headroom'/name)
        require(len(table)==96,'Wrong headroom coverage')
        for r in table.itertuples():
            b='sroie' if r.dataset=='SROIE' else r.setting
            s=pd.read_csv(bundle/'per_page_bestk'/b/f'per_page_results_{r.engine}.csv').set_index('image_name').sort_index()
            nn=pd.read_csv(bundle/'nn_baseline'/b/'nn_per_image.csv');nn=nn[nn.engine.eq(r.engine)].set_index('image_name').loc[s.index]
            v={'Surrogate':s[f'surr_k{r.k}'],'Random':s[f'rand_k{r.k}'],'FixedTopK':s[f'fixed_topk_k{r.k}'],'NN-K':nn[f'nn_k{r.k}_cer']}
            h=v['FixedTopK']-s.oracle
            if key=='policy':gain=v['FixedTopK']-v[r.policy]
            else:gain={'Surrogate_vs_Random':v['Random']-v['Surrogate'],'NN-K_vs_Random':v['Random']-v['NN-K'],'Surrogate_vs_NN-K':v['NN-K']-v['Surrogate']}[r.comparison]
            rho,p=spearmanr(h,gain)
            np.testing.assert_allclose([rho,p],[getattr(r,column),r.spearman_p_value],rtol=1e-12,atol=1e-12)
        for _,g in table.groupby(key):
            require(len(g)==32,'Wrong headroom family')
            np.testing.assert_allclose(independent_holm(g.spearman_p_value),g.spearman_p_holm,rtol=1e-12,atol=1e-15)
    result=dict(primary_comparisons=224,primary_families=21,A_comparisons=384,A_families=36,
        metric_comparisons=80,metric_families=10,A_changed_classifications=changed,headroom_correlations=192)
    print('PASS corrected612 portable replay:',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    replay(p.parse_args().bundle)
