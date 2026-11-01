"""Versioned CV/oracle benefit and saved model-quality summaries; no refits."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from compute_preprocessing_benefit import load_oracle,load_nopre,within_image_cv,per_image_stats
from export_sroie_casefold_sensitivity import sha

ROOT=Path(__file__).resolve().parents[1]
DATA={
 'raw':('outputs/vn_v3_case_sensitive_raw_20260930_v1','default_cer_vn_v3.csv'),
 'contour':('outputs_contour/vn_v3_case_sensitive_contour_20261001_v2','default_cer_vn_v3.csv'),
 'sroie':('SROIRE_COMPARISON/output_sroire/casefold_training_inputs_20260926_v2','default_cer_sroie_all616_casefold.csv')}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--config',type=Path,required=True)
 parser.add_argument('--output-dir',type=Path,required=True)
 args=parser.parse_args()
 if args.output_dir.exists():parser.error('Choose a fresh directory')
 cfg=json.loads(args.config.read_text());benefit=[];quality=[];sources={};perpages=[]
 if set(cfg)!=set(DATA):raise ValueError('Wrong blocks')
 for block,(data,nopre_name) in DATA.items():
  response=ROOT/data/'preprocessing_data.csv';nopre=ROOT/data/nopre_name
  metrics_path=ROOT/cfg[block]['surrogate']/'metrics.json';metrics=json.loads(metrics_path.read_text())
  engines=['tesseract','easyocr'] if block=='sroie' else ['vietocr','tesseract','easyocr']
  pages=616 if block=='sroie' else 300
  for f in [response,nopre,metrics_path]:sources[str(f)]=sha(f)
  raw=pd.read_csv(response)
  if len(raw)!=pages*50 or raw.groupby('image_name').size().ne(50).any():raise ValueError('Wrong pool coverage')
  oracle=load_oracle(response,engines);default=load_nopre(nopre,engines)
  if default.image_name.duplicated().any() or set(default.image_name)!=set(oracle.image_name):raise ValueError('NoPre cohort mismatch')
  merged=oracle.merge(default,on='image_name',validate='one_to_one')
  if len(merged)!=pages:raise ValueError('Missing pages')
  cv=within_image_cv(response,engines);merged.insert(0,'block',block);perpages.append(merged)
  for engine in engines:
   columns=[f'oracle_{engine}',f'nopre_{engine}']
   if not np.isfinite(merged[columns]).all().all() or merged[f'nopre_{engine}'].le(0).any():raise ValueError('Nonfinite/zero baseline requires explicit handling')
   m=metrics[engine];perpath=ROOT/cfg[block]['surrogate']/f'per_page_results_{engine}.csv'
   sources[str(perpath)]=sha(perpath);per=pd.read_csv(perpath).set_index('image_name').sort_index()
   reference=merged.set_index('image_name').sort_index()
   for pc,mc in [('oracle',f'oracle_{engine}'),('no_preprocessing',f'nopre_{engine}')]:
    np.testing.assert_allclose(per[pc],reference[mc],rtol=0,atol=1e-12)
    np.testing.assert_allclose(per[pc].mean(),m['baselines'][pc],rtol=0,atol=1e-12)
   for k in [3,5,10,15]:
    for col,key in [(f'surr_k{k}','filter_results'),(f'rand_k{k}','random_results'),(f'fixed_topk_k{k}','fixed_topk_results')]:
     np.testing.assert_allclose(per[col].mean(),m[key][str(k)],rtol=0,atol=1e-12)
   benefit.append(dict(block=block,engine=engine,**cv[engine],**per_image_stats(merged,engine),
      nopre_mean_pct=100*reference[f'nopre_{engine}'].mean(),oracle_mean_pct=100*reference[f'oracle_{engine}'].mean()))
   quality.append(dict(block=block,engine=engine,**m['cv_metrics'],
      selected_models_per_fold=json.dumps(m['selected_models_per_fold']),
      verification_scope='copied saved fold aggregates; no independent OOF prediction replay'))
 args.output_dir.mkdir(parents=True)
 outputs={'preprocessing_benefit.csv':pd.DataFrame(benefit),'model_quality.csv':pd.DataFrame(quality),
          'oracle_nopre_per_page.csv':pd.concat(perpages,ignore_index=True)}
 for name,frame in outputs.items():frame.to_csv(args.output_dir/name,index=False)
 meta=dict(sources=sources,config_sha256=sha(args.config),script_sha256=sha(__file__),
   helper_sha256=sha(ROOT/'analysis/compute_preprocessing_benefit.py'),
   outputs={name:sha(args.output_dir/name) for name in outputs},
   definitions='sample within-page CV ddof=1; mean page-relative oracle improvement; model quality saved fold aggregates')
 (args.output_dir/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(pd.DataFrame(benefit).to_string(index=False));print('Saved model quality is extracted, not independently recalculated from predictions')

if __name__=='__main__':main()
