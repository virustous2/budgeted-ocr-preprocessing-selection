"""Both VN blocks, same 12-test Holm families per zero convention; no fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from export_sroie_casefold_sensitivity import holm,pvalue,sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a fresh versioned directory')
    config=json.loads(args.config.read_text());rows=[];sources={}
    for block in ['raw','contour']:
        for engine in ['vietocr','tesseract','easyocr']:
            path=ROOT/config[block]['surrogate']/f'per_page_results_{engine}.csv'
            sources[str(path)]=sha(path);d=pd.read_csv(path)
            if len(d)!=300 or d.image_name.nunique()!=300:raise ValueError('Wrong page coverage')
            for k in [3,5,10,15]:
                a=d[f'surr_k{k}'].to_numpy();b=d[f'fixed_topk_k{k}'].to_numpy();diff=a-b
                if not np.isfinite(diff).all():raise ValueError('Nonfinite CER')
                for zero in ['wilcox','pratt','zsplit']:
                    rows.append(dict(block=block,engine=engine,k=k,zero_method=zero,n_pages=300,
                        n_zero_diff=int((diff==0).sum()),difference_pp=100*diff.mean(),
                        p_two_sided=pvalue(a,b,zero)))
    result=pd.DataFrame(rows)
    for _,idx in result.groupby(['block','zero_method']).groups.items():
        if len(idx)!=12:raise ValueError('Incomplete family')
        result.loc[idx,'p_holm_12']=holm(result.loc[idx,'p_two_sided'])
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir/'zero_sensitivity.csv',index=False)
    metadata=dict(primary='wilcox unchanged; alternatives post hoc sensitivity',sources=sources,
        config_sha256=sha(args.config),script_sha256=sha(__file__),
        dependency_sha256=sha(ROOT/'analysis/export_sroie_casefold_sensitivity.py'),
        output_sha256=sha(args.output_dir/'zero_sensitivity.csv'))
    (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Wrote 72rows, six12-test families to',args.output_dir)

if __name__=='__main__':main()
