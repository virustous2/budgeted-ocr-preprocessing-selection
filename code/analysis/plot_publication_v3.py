#!/usr/bin/env python3
"""Plot checked best-of-K means and current headroom from an isolated bundle."""
import argparse
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plot_oracle_gap_diagnostic import draw_distribution, POLICY_ORDER, COMPARISON_ORDER, POLICY_LABELS, COMPARISON_LABELS


def plot(bundle,output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True);points=[]
    for block in ('raw','contour'):
        fig,axes=plt.subplots(1,3,figsize=(9,3),constrained_layout=True)
        for ax,engine in zip(axes,('vietocr','tesseract','easyocr')):
            df=pd.read_csv(bundle/'per_page_bestk'/block/f'per_page_results_{engine}.csv')
            nn=pd.read_csv(bundle/'nn_baseline'/block/'nn_per_image.csv');nn=nn[nn.engine.eq(engine)]
            for name,frame,cols in [('Surrogate',df,[f'surr_k{k}' for k in (3,5,10,15)]),('FixedTopK',df,[f'fixed_topk_k{k}' for k in (3,5,10,15)]),('Random',df,[f'rand_k{k}' for k in (3,5,10,15)]),('NN-K',nn,[f'nn_k{k}_cer' for k in (3,5,10,15)])]:
                values=[100*frame[c].mean() for c in cols];ax.plot((3,5,10,15),values,marker='o',label=name)
                points.extend(dict(block=block,engine=engine,policy=name,k=k,cer_pct=v) for k,v in zip((3,5,10,15),values))
            ax.set_title(engine);ax.set_xlabel('OCR budget K');ax.set_ylabel('Best-of-K CER (%)');ax.grid(alpha=.3)
        axes[-1].legend(fontsize=8);fig.savefig(output/f'{block}_pareto_multiengine.png',dpi=300);plt.close(fig)
    pd.DataFrame(points).to_csv(output/'curve_points.csv',index=False)
    coupled=pd.read_csv(bundle/'evidence/headroom/headroom_policy_summary_allK.csv')
    controls=pd.read_csv(bundle/'evidence/headroom/headroom_nonshared_controls_allK.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.25),constrained_layout=True)
    draw_distribution(axes[0],coupled,'policy',POLICY_ORDER,'spearman_headroom_vs_advantage',POLICY_LABELS,'(a) Coupled FixedTopK-based statistic')
    draw_distribution(axes[1],controls,'comparison',COMPARISON_ORDER,'spearman_headroom_vs_gain',COMPARISON_LABELS,'(b) Non-shared policy-gain controls')
    fig.savefig(output/'headroom_policy_controls.pdf',bbox_inches='tight');fig.savefig(output/'headroom_policy_controls.png',dpi=300,bbox_inches='tight');plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps({'bundle_manifest_sha256':sha(bundle/'manifest.json'),'script_sha256':sha(Path(__file__)),'plot_helper_sha256':sha(Path(__file__).with_name('plot_oracle_gap_diagnostic.py')),'files':{p.name:sha(p) for p in sorted(output.iterdir())}},indent=2)+'\n')
    print('Wrote two VN curves and headroom panel:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();plot(a.bundle,a.output_dir)
