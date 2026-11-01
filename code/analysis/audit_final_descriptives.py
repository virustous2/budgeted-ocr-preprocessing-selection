"""Replay headroom, benefit and zero sensitivity without production helpers."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon

ROOT=Path(__file__).resolve().parents[1]

def check(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

def adjusted(p):
 p=np.asarray(p);order=np.argsort(p);result=np.empty(len(p))
 result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return result

def main():
 cfg=json.loads((ROOT/'analysis/bestk_sources_v3_casefold_20261010.json').read_text())
 h=ROOT/'outputs/headroom_v3_casefold_20261010_v1'
 summaries=pd.read_csv(h/'headroom_policy_summary_allK.csv');nonshared=pd.read_csv(h/'headroom_nonshared_controls_allK.csv');quart=pd.read_csv(h/'headroom_policy_quartiles_allK.csv')
 assert len(summaries)==96 and len(nonshared)==96 and len(quart)==384
 for block,paths in cfg.items():
  dataset='SROIE' if block=='sroie' else 'Vietnamese';setting='raw' if block=='sroie' else block
  nn=pd.read_csv(ROOT/paths['nn']/'nn_per_image.csv')
  for engine in nn.engine.unique():
   per=pd.read_csv(ROOT/paths['surrogate']/f'per_page_results_{engine}.csv').set_index('image_name').sort_index()
   n=nn[nn.engine==engine].set_index('image_name').sort_index()
   for k in [3,5,10,15]:
    fixed=per[f'fixed_topk_k{k}'].to_numpy();head=fixed-per.oracle.to_numpy()
    vals={'Surrogate':per[f'surr_k{k}'].to_numpy(),'Random':per[f'rand_k{k}'].to_numpy(),'NN-K':n[f'nn_k{k}_cer'].to_numpy()}
    idx=np.array_split(np.argsort(head,kind='mergesort'),4)
    def subset(d):return d[(d.dataset==dataset)&(d.setting==setting)&(d.engine==engine)&(d.k==k)]
    for policy,v in vals.items():
     row=subset(summaries);row=row[row.policy==policy].iloc[0];adv=fixed-v;rho,p=spearmanr(head,adv)
     check([rho,p,100*adv.mean(),100*head.mean()],[row.spearman_headroom_vs_advantage,row.spearman_p_value,row.mean_policy_advantage_pp,row.mean_headroom_pp])
     for q,j in enumerate(idx,1):
      z=subset(quart);z=z[(z.policy==policy)&(z.quartile==f'Q{q}')].iloc[0]
      assert z.n_pages==len(j);check(z.mean_policy_advantage_pp,100*adv[j].mean());check(z.mean_headroom_pp,100*head[j].mean())
    for name,a,b in [('Surrogate_vs_Random','Random','Surrogate'),('NN-K_vs_Random','Random','NN-K'),('Surrogate_vs_NN-K','NN-K','Surrogate')]:
     row=subset(nonshared);row=row[row.comparison==name].iloc[0];rho,p=spearmanr(head,vals[a]-vals[b])
     check([rho,p,100*(vals[a]-vals[b]).mean()],[row.spearman_headroom_vs_gain,row.spearman_p_value,row.mean_gain_pp])
 for frame,col in [(summaries,'policy'),(nonshared,'comparison')]:
  for _,g in frame.groupby(col):assert len(g)==32;check(adjusted(g.spearman_p_value),g.spearman_p_holm)
 desc=ROOT/'outputs/final_descriptives_v3_casefold_20261010_v1';benefit=pd.read_csv(desc/'preprocessing_benefit.csv')
 for block,data,nopre in [('raw','outputs/vn_v3_case_sensitive_raw_20260930_v1','default_cer_vn_v3.csv'),('contour','outputs_contour/vn_v3_case_sensitive_contour_20261001_v2','default_cer_vn_v3.csv'),('sroie','SROIRE_COMPARISON/output_sroire/casefold_training_inputs_20260926_v2','default_cer_sroie_all616_casefold.csv')]:
  pool=pd.read_csv(ROOT/data/'preprocessing_data.csv');default=pd.read_csv(ROOT/data/nopre)
  default['image_name']=default.file_name.map(lambda name:Path(name).stem)
  default=default.set_index('image_name')
  for row in benefit[benefit.block==block].itertuples():
   groups=pool.groupby('image_name')[f'cer_{row.engine}'];means=groups.mean();cv=groups.std(ddof=1)/means;oracle=groups.min();no=default.loc[oracle.index,f'cer_{row.engine}']
   check(row.mean_cv_pct,100*cv[means>0].mean());check(row.mean_rel_imp,100*((no-oracle)/no).mean());assert row.n_benefit==int((oracle<no).sum())
 z=pd.read_csv(ROOT/'outputs/vn_zero_sensitivity_v3_20261010_v1/zero_sensitivity.csv');assert len(z)==72
 for row in z.itertuples():
  d=pd.read_csv(ROOT/cfg[row.block]['surrogate']/f'per_page_results_{row.engine}.csv');a=d[f'surr_k{row.k}'].to_numpy();b=d[f'fixed_topk_k{row.k}'].to_numpy()
  assert row.n_zero_diff==int(((a-b)==0).sum());check(row.p_two_sided,1. if np.all(a==b) else wilcoxon(a,b,alternative='two-sided',zero_method=row.zero_method).pvalue)
 for _,g in z.groupby(['block','zero_method']):assert len(g)==12;check(adjusted(g.p_two_sided),g.p_holm_12)
 print('PASS: 192correlations/6Holm families,384quartiles,8benefit rows,72zero-sensitivity rows/6Holm families;no fits')

if __name__=='__main__':main()
