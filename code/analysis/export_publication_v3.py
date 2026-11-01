#!/usr/bin/env python3
"""Generate publication tables and an isolated public result bundle. No fitting/OCR.

Historical public roots are retained. This command refuses an existing output.
The publication audit replays table rendering from source CSVs, not hand values.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import csv
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ('raw', 'contour', 'sroie')
ENGINES = {'raw': ('vietocr','tesseract','easyocr'), 'contour': ('vietocr','tesseract','easyocr'), 'sroie': ('tesseract','easyocr')}
LABEL = {'raw':'Raw','contour':'Contour','sroie':'SROIE','vietocr':'VietOCR','tesseract':'Tesseract','easyocr':'EasyOCR'}
EVIDENCE = {
 'bestk': 'outputs/bestk_statistics_v3_casefold_20261010_v1',
 'static': 'outputs/strict_selector_static_comparison_v3_casefold_20261010_v1',
 'policy': 'outputs/strict_selector_policy_comparison_v3_casefold_20261010_v1',
 'headroom': 'outputs/headroom_v3_casefold_20261010_v1',
 'descriptives': 'outputs/final_descriptives_v3_casefold_20261010_v1',
 'zero': 'outputs/vn_zero_sensitivity_v3_20261010_v1',
}
RESPONSE = {
 'raw':'outputs/vn_v3_case_sensitive_raw_20260930_v1',
 'contour':'outputs_contour/vn_v3_case_sensitive_contour_20261001_v2',
 'sroie':'SROIRE_COMPARISON/output_sroire/casefold_training_inputs_20260926_v2',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fmt(value, decimals=2, signed=False):
    value=float(value)
    if round(value, decimals)==0: value=0.0
    return format(value, ('+' if signed and value else '')+f'.{decimals}f')

def ci_fmt(value):
    """Retain the direction of nonzero CI endpoints even below display precision."""
    value=float(value)
    if value==0: return '0.00'
    return format(value,'+.2f')

def pformat(value):
    s=format(float(value),'.3g')
    if 'e' in s:
        m,e=s.split('e'); return '$'+m+r'\times10^{'+str(int(e))+'}$'
    # Retain three significant digits rather than stripping trailing zeroes.
    x=float(value)
    digits=2 if x>=1 else max(0,2-int(np.floor(np.log10(x)))) if x>0 else 2
    return format(x,f'.{digits}f')

def row(cells):
    return ' & '.join(map(str,cells))+r' \\'+'\n'

def one(frame, **kwargs):
    for k,v in kwargs.items(): frame=frame[frame[k].eq(v)]
    if len(frame)!=1: raise ValueError(f'Expected one row: {kwargs}, found {len(frame)}')
    return frame.iloc[0]

def load():
    config=json.loads((ROOT/'analysis/bestk_sources_v3_casefold_20261010.json').read_text())
    selectors=json.loads((ROOT/'analysis/strict_selector_comparison_v3_casefold_20261010.json').read_text())
    read=lambda family,name: pd.read_csv(ROOT/EVIDENCE[family]/name)
    per={}; metrics={}; nn={}
    for b in BLOCKS:
        metrics[b]=json.loads((ROOT/config[b]['surrogate']/'metrics.json').read_text())
        nn[b]=pd.read_csv(ROOT/config[b]['nn']/'nn_per_image.csv')
        for e in ENGINES[b]:
            per[b,e]=pd.read_csv(ROOT/config[b]['surrogate']/f'per_page_results_{e}.csv')
            if len(per[b,e])!=(616 if b=='sroie' else 300): raise ValueError('Incomplete cohort')
    return dict(config=config,selectors=selectors,per=per,metrics=metrics,nn=nn,
      stats=read('bestk','statistics.csv'), static=read('static','strict_selector_static_paired_comparison.csv'),
      policy=read('policy','strict_selector_policy_paired_comparison.csv'),
      benefit=read('descriptives','preprocessing_benefit.csv'),quality=read('descriptives','model_quality.csv'),
      coupled=read('headroom','headroom_policy_summary_allK.csv'),nonshared=read('headroom','headroom_nonshared_controls_allK.csv'),
      zero=read('zero','zero_sensitivity.csv'))

def render(d, preserve_ci_sign=True, compact_sroie=False, preserve_effect_sign=False):
    out={}; per=d['per']; means=lambda b,e,c:100*per[b,e][c].mean()
    nnmean=lambda b,e,k:100*d['nn'][b].loc[d['nn'][b].engine.eq(e),f'nn_k{k}_cer'].mean()
    lines=[]
    for b in BLOCKS[:2]:
        for e in ENGINES[b]:
            lines.append(row([LABEL[b],LABEL[e]]+[fmt(means(b,e,c)) for c in ('no_preprocessing','fixed_best','oracle')]+[fmt(means(b,e,f'surr_k{k}'))+' / '+fmt(means(b,e,f'rand_k{k}')) for k in (3,5,10,15)]))
    out['vn_bestk.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS:
        for e in ENGINES[b]:
            cells=[LABEL[b],LABEL[e]]
            for k in (3,15):
                s,f=means(b,e,f'surr_k{k}'),means(b,e,f'fixed_topk_k{k}')
                cells += [fmt(s),fmt(f),fmt(s-f,signed=True)]
            lines.append(row(cells))
    out['fixedtopk.tex']=''.join(lines)
    lines=[]
    for e in ENGINES['sroie']:
        q=one(d['quality'],block='sroie',engine=e)
        cells=[LABEL[e]]
        if not compact_sroie:
            cells += [fmt(q[c],3) for c in ('r2_mean','spearman_global_mean','spearman_per_image_mean')]
        cells += [fmt(means('sroie',e,c)) for c in ('no_preprocessing','fixed_best','oracle')]
        for k in (3,15):
            policy=[fmt(means('sroie',e,f'surr_k{k}')),fmt(nnmean('sroie',e,k)),fmt(means('sroie',e,f'fixed_topk_k{k}')),fmt(means('sroie',e,f'rand_k{k}'))]
            if compact_sroie:
                lines.append(row([cells[0],str(k)]+cells[1:]+policy))
            else:
                cells += policy
        if not compact_sroie:lines.append(row(cells))
    out['sroie_results.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS:
        for e in ENGINES[b]:
            setting='receipt' if b=='sroie' else b
            fixed=one(d['static'],setting=setting,engine=e,method_b='FixedBest')
            values=[fixed.surrogate_mean_cer_pct]
            for method in ('Top1Pred','Agreement','FixedBest','NoPre'):
                values.append(one(d['static'],setting=setting,engine=e,method_b=method).baseline_mean_cer_pct)
            effect=ci_fmt(fixed.mean_diff_surrogate_minus_baseline_pp) if preserve_effect_sign else fmt(fixed.mean_diff_surrogate_minus_baseline_pp,signed=True)
            lines.append(row([LABEL[b],LABEL[e]]+[fmt(v) for v in values]+[effect,pformat(fixed.p_two_sided),pformat(fixed.p_holm_manuscript)]))
    out['selector.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS:
        for e in ENGINES[b]:
            r=one(d['benefit'],block=b,engine=e)
            lines.append(row([LABEL[b],LABEL[e],f'{int(r.n_benefit)}/{int(r.n_total)} ({fmt(r.pct_benefit,1)}'+r'\%)',fmt(r.mean_rel_imp,1)+r'\%',fmt(r.median_rel_imp,1)+r'\%']))
    out['oracle_improvement.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS:
        for e in ENGINES[b]:
            q=one(d['quality'],block=b,engine=e)
            model=d['metrics'][b][e]['best_model'].replace('RandomForest','Random Forest')
            lines.append(row([LABEL[b],LABEL[e],model,fmt(q.mae_mean,3),fmt(q.r2_mean,3)]+['$'+fmt(q[c+'_mean'],3)+r' \pm '+fmt(q[c+'_std'],3)+'$' for c in ('spearman_global','spearman_per_image')]))
    out['model_quality.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS:
        for a,c in [('Surrogate','Random'),('Surrogate','FixedTopK'),('NN','Random'),('NN','FixedTopK'),('NN','Surrogate')]:
            g=d['stats'].query('block == @b and method_a == @a and method_b == @c')
            counts=[]
            for sign in (-1,1):
                counts.append(str(sum((g.p_two_sided<.05)&(g.difference_pp*sign>0)))+'/'+str(sum((g.p_holm<.05)&(g.difference_pp*sign>0))))
            lines.append(row([LABEL[b]+' VN' if b!='sroie' else 'SROIE',a.replace('NN','NN-K')+' vs '+c]+counts))
    out['policy_counts.tex']=''.join(lines)
    lines=[]
    for typ,name,column,qcol,labels in [
      ('Coupled','policy','spearman_headroom_vs_advantage','q4_minus_q1_advantage_pp',{'Surrogate':'FixedTopK $-$ Surrogate','Random':'FixedTopK $-$ Random','NN-K':'FixedTopK $-$ NN-K'}),
      ('Non-shared','comparison','spearman_headroom_vs_gain','q4_minus_q1_gain_pp',{'Surrogate_vs_Random':'Random $-$ Surrogate','NN-K_vs_Random':'Random $-$ NN-K','Surrogate_vs_NN-K':'NN-K $-$ Surrogate'})]:
        frame=d['coupled'] if typ=='Coupled' else d['nonshared']
        for key,label in labels.items():
            g=frame[frame[name].eq(key)];r=g[column]
            lines.append(row([typ,label,'['+fmt(r.min(),3)+', '+fmt(r.max(),3)+']',str(sum((g.spearman_p_holm<.05)&(r>0)))+'/32',str(sum(g[qcol]>0))+'/32']))
    out['headroom.tex']=''.join(lines)
    lines=[]
    for k in (3,5,10,15):
        g=d['coupled'][d['coupled'].k.eq(k)].pivot(index=['dataset','setting','engine'],columns='policy',values='spearman_headroom_vs_advantage')
        counts=[sum((d['nonshared'].k.eq(k))&(d['nonshared'].comparison.eq(c))&(d['nonshared'].spearman_headroom_vs_gain>0)&(d['nonshared'].spearman_p_holm<.05)) for c in ('Surrogate_vs_Random','NN-K_vs_Random','Surrogate_vs_NN-K')]
        lines.append(row([k,'8/8',str(sum(g.Random>g.Surrogate))+'/8']+[str(x)+'/8' for x in counts]))
    out['headroom_budget.tex']=''.join(lines)
    for variant,analysis in [('full','primary_policy_comparison'),('rank_only','rank_only_sensitivity')]:
        ci_display=ci_fmt if preserve_ci_sign else lambda v:fmt(v,signed=True)
        lines=[]
        for method in ('NN-K','FixedTopK'):
            for b in BLOCKS:
                for e in ENGINES[b]:
                    r=one(d['policy'],setting='receipt' if b=='sroie' else b,engine=e,method_b=method,analysis_set=analysis)
                    lines.append(row([LABEL[b],LABEL[e],method,fmt(r.surrogate_mean_cer_pct),fmt(r.baseline_mean_cer_pct),fmt(r.mean_diff_surrogate_minus_baseline_pp,signed=True),'['+ci_display(r.bootstrap_95ci_low_pp)+', '+ci_display(r.bootstrap_95ci_high_pp)+']',pformat(r.p_holm_manuscript)]))
        out[variant+'_policy.tex']=''.join(lines)
    lines=[]
    for e in ENGINES['sroie']:
        for k in (5,10): lines.append(row([LABEL[e],k,fmt(means('sroie',e,f'surr_k{k}')),fmt(nnmean('sroie',e,k)),fmt(means('sroie',e,f'fixed_topk_k{k}')),fmt(means('sroie',e,f'rand_k{k}'))]))
    out['sroie_intermediate.tex']=''.join(lines)
    lines=[]
    for b in BLOCKS[:2]:
        for z in ('wilcox','pratt','zsplit'):
            r=one(d['zero'],block=b,engine='easyocr',k=15,zero_method=z)
            lines.append(row([LABEL[b],z,int(r.n_zero_diff),fmt(r.difference_pp,signed=True),pformat(r.p_holm_12)]))
    out['zero_sensitivity.tex']=''.join(lines)
    return out

def generate(output):
    if output.exists(): raise FileExistsError(output)
    d=load(); output.mkdir(parents=True);tables=output/'tables';tables.mkdir()
    for name,body in render(d).items(): (tables/name).write_text('% Generated from frozen page-level outputs; CHG-20261010-07.\n'+body)
    inputs={}
    def copy(source,dest):
        source=ROOT/source; dest=output/dest;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);inputs[str(source.relative_to(ROOT))]=sha(source)
    for family,path in EVIDENCE.items():
        for source in sorted((ROOT/path).glob('*')):
            if source.suffix in ('.csv','.json'): copy(source.relative_to(ROOT),Path('evidence')/family/source.name)
    for b in BLOCKS:
        for e in ENGINES[b]: copy(Path(d['config'][b]['surrogate'])/f'per_page_results_{e}.csv',Path('per_page_bestk')/b/f'per_page_results_{e}.csv')
        copy(Path(d['config'][b]['surrogate'])/'metrics.json',Path('per_page_bestk')/b/'metrics.json')
        copy(Path(d['config'][b]['nn'])/'nn_per_image.csv',Path('nn_baseline')/b/'nn_per_image.csv')
        copy(Path(RESPONSE[b])/'preprocessing_data.csv',Path('response')/b/'preprocessing_data.csv')
        copy(Path(RESPONSE[b])/('default_cer_sroie_all616_casefold.csv' if b=='sroie' else 'default_cer_vn_v3.csv'),Path('response')/b/'nopre.csv')
        copy(Path(d['config'][b]['folds']),Path('response')/b/'fold_assignments.csv')
        for variant,key in [('full','surrogate'),('rank_only','surrogate'),('nn_k','policy'),('fixed_topk','policy')]:
            src=ROOT/d['selectors'][b][key]/variant
            scores=pd.read_csv(src/'nested_selector_candidate_scores.csv',keep_default_na=False)
            if len(scores)!=(616 if b=='sroie' else 300)*len(ENGINES[b])*3: raise ValueError('Selector coverage')
            dst=output/'selectors'/b/variant;dst.mkdir(parents=True)
            # Strip only text; preserve every numeric field lexically. A pandas
            # CSV rewrite can perturb tied CERs at the last bit on rereading.
            with (src/'nested_selector_candidate_scores.csv').open(newline='') as fi, (dst/'scores.csv').open('w',newline='') as fo:
                reader=csv.DictReader(fi)
                fields=[x for x in reader.fieldnames if x!='ocr_text']
                writer=csv.DictWriter(fo,fieldnames=fields);writer.writeheader()
                for record in reader: writer.writerow({x:record[x] for x in fields})
            inputs[str((src/'nested_selector_candidate_scores.csv').relative_to(ROOT))]=sha(src/'nested_selector_candidate_scores.csv')
            for name in ('nested_selector_summary.csv','nested_selector_inner_validation.csv'):
                copy((src/name).relative_to(ROOT),Path('selectors')/b/variant/name)
    # Hash the final evaluated v3 references, not the original recovery references.
    manifest=pd.read_csv(ROOT/'ijdar_public_release/data/cohort_manifests/vietnamese_cohort_manifest.csv',dtype=str)
    for i,r in manifest.iterrows():
        path=ROOT/'datasets/vn_labels_corrected_20260927_v3/labels'/r.label_file
        manifest.loc[i,'label_sha256']=sha(path);inputs[str(path.relative_to(ROOT))]=sha(path)
    manifest.to_csv(output/'vietnamese_cohort_manifest.csv',index=False)
    files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps({'change':'CHG-20261010-07','inputs':inputs,'files':files,'metric':{'raw':'NFC/whitespace; case-sensitive; VN references v3','contour':'NFC/whitespace; case-sensitive; VN references v3','sroie':'NFC/whitespace; Unicode casefold both sides'},'contains_ocr_text':False},indent=2)+'\n')
    print('Generated',len(render(d)),'tables and sanitized public bundle:',output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    generate(parser.parse_args().output_dir)
