"""Current 612-publication consistency audit; old audits remain historical.

Portable arithmetic replay plus cohort/source/display/figure/paper/public
checks. Not a scientific-validity, duplicate-recall or acceptance certificate.
"""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from export_receipt612_publication import ROOT, COUNTS, SOURCE, VERIFIED, ENGINES, load_bundle, render, a_tables, metric_tables, sha, validate_source
from replay_receipt612_publication import replay


def main():
    authority=json.loads((ROOT/'analysis/publication_authority_receipt612.json').read_text())
    bundle=ROOT/authority['bundle'];public=ROOT/authority['public_bundle']
    fig=ROOT/authority['figures'];public_fig=ROOT/authority['public_figures']
    good=[];bad=[];trace=[]
    def check(ok,label,source=''):
        (good if ok else bad).append(label)
        trace.append(dict(claim_group='receipt612_publication',claim=label,source_artifact=source,status='PASS' if ok else 'FAIL'))
    validate_source()
    meta=json.loads((bundle/'manifest.json').read_text())
    prepared=json.loads((SOURCE/'prepared.json').read_text())
    for name,h in prepared['sources'].items():check(sha(ROOT/name)==h,'Frozen upstream '+name)
    for name,h in prepared['code'].items():check(sha(ROOT/name)==h,'Frozen statistical code '+name)
    check(meta['source_manifest_sha256']==sha(SOURCE/'complete.json'),'Verified analysis manifest binding')
    check(meta['source_verification_sha256']==sha(VERIFIED),'Independent private verification binding')
    check(meta['exporter_sha256']==sha(ROOT/'analysis/export_receipt612_publication.py'),'Exporter code binding')
    for name,h in meta['inputs'].items():check(sha(ROOT/name)==h,'Publication source '+name)
    for name,h in meta['files'].items():
        check(sha(bundle/name)==h,'Bundle payload '+name)
        check(sha(public/name)==h,'Public payload '+name)
    check((bundle/'manifest.json').read_bytes()==(public/'manifest.json').read_bytes(),'Public bundle manifest')
    check('results/'+bundle.name in (bundle/'README.md').read_text(),'Bundle README points to its actual public directory')
    static=pd.read_csv(bundle/'evidence/static/strict_selector_static_paired_comparison.csv')
    for method,lower,significant in [('FixedBest',5,3),('Top1Pred',8,4),('NoPre',7,6),('Agreement',5,1)]:
        g=static[static.method_b.eq(method)]
        check(len(g)==8 and sum(g.mean_diff_surrogate_minus_baseline_pp<0)==lower and
            sum((g.mean_diff_surrogate_minus_baseline_pp<0)&(g.p_holm_manuscript<.05))==significant,
            'Numeric static headline '+method)
    direct=pd.read_csv(bundle/'sensitivity/A/direct_drop_minus_full.csv')
    check(len(direct)==64 and sum(direct.p_holm<.05)==0,'No Holm-significant direct A comparison')
    changes=pd.read_csv(bundle/'sensitivity/A/holm_changes_drop_vs_full.csv')
    changed=changes[changes.changed]
    check(len(changed)==7 and np.all(np.sign(changed.difference_pp_a)==np.sign(changed.difference_pp_b)),
          'Seven changed A classifications, no sign reversals in those rows')
    expected=render(load_bundle(bundle),compact_sroie=True,preserve_effect_sign=True)
    cols={'vn_bestk.tex':9,'fixedtopk.tex':8,'sroie_results.tex':9,'selector.tex':10,'oracle_improvement.tex':5,'model_quality.tex':7,'policy_counts.tex':4,'headroom.tex':5,'headroom_budget.tex':6,'full_policy.tex':8,'rank_only_policy.tex':8,'sroie_intermediate.tex':6,'zero_sensitivity.tex':5}
    for name,body in expected.items():
        for folder in [bundle/'tables',ROOT/'IJDAR_overleaf/primary_tables',ROOT/'ijdar_public_release/paper/primary_tables']:
            actual='\n'.join(x for x in (folder/name).read_text().splitlines() if not x.startswith('%'))+'\n'
            check(actual==body,'Table source rendering '+str(folder/name))
        for line in body.splitlines():check(len(line.split('&'))==cols[name],'Column count '+name)
    a=a_tables(pd.read_csv(bundle/'sensitivity/A/direct_drop_minus_full.csv'),pd.read_csv(bundle/'sensitivity/A/holm_changes_drop_vs_full.csv'))
    m=metric_tables(pd.read_csv(bundle/'sensitivity/letter_case/statistics.csv'))
    for name,body in [('noise_feature_tables.tex',a),('sroie_casefold_tables.tex',m)]:
        for folder in [bundle,ROOT/'IJDAR_overleaf',ROOT/'ijdar_public_release/paper']:
            check((folder/name).read_text()==body,'Sensitivity source rendering '+str(folder/name))
    # Exact lexical stripping, not lossy numeric reserialization.
    config=json.loads((SOURCE/'configuration.json').read_text())
    for b in COUNTS:
        for variant,key in [('full','surrogate'),('rank_only','surrogate'),('nn_k','policy'),('fixed_topk','policy')]:
            src=ROOT/config['selectors'][b][key]/variant/'nested_selector_candidate_scores.csv'
            dst=bundle/'selectors'/b/variant/'scores.csv'
            with src.open(newline='') as fi,dst.open(newline='') as fo:
                old=list(csv.DictReader(fi));new=list(csv.DictReader(fo))
            check(new==[{k:v for k,v in r.items() if k!='ocr_text'} for r in old],'Selector exact retained fields '+b+'/'+variant)
    result=replay(bundle)
    check(result['A_changed_classifications']==7,'Seven A classification changes, all retained')
    fm=json.loads((fig/'manifest.json').read_text())
    check(fm['bundle_manifest_sha256']==sha(bundle/'manifest.json'),'Corrected figure source')
    check(fm['exporter_sha256']==sha(ROOT/'analysis/export_receipt612_publication.py'),'Figure exporter binding')
    check(fm['plot_helper_sha256']==sha(ROOT/'analysis/plot_oracle_gap_diagnostic.py'),'Figure helper binding')
    prior=ROOT/'outputs/publication_figure_clarity_20261016_v2'
    for name in ['curve_points.csv','raw_pareto_multiengine.png','contour_pareto_multiengine.png','pipeline_flowchart_2_5.pdf']:
        check((fig/name).read_bytes()==(prior/name).read_bytes(),'Unchanged VN/pipeline asset '+name)
    for name,h in fm['files'].items():
        check(sha(fig/name)==h and sha(public_fig/name)==h,'Figure/source public hash '+name)
        if name in fm['paper_assets']:
            for folder in ['IJDAR_overleaf/figures','ijdar_public_release/paper/figures']:
                check(sha(ROOT/folder/name)==h,'Figure paper copy '+folder+'/'+name)
    check((fig/'manifest.json').read_bytes()==(public_fig/'manifest.json').read_bytes(),'Public figure manifest')
    points=pd.read_csv(fig/'curve_points.csv');d=load_bundle(bundle)
    check(len(points)==96,'96 VN curve points')
    for r in points.itertuples():
        values=d['nn'][r.block].query('engine==@r.engine')[f'nn_k{r.k}_cer'] if r.policy=='NN-K' else d['per'][r.block,r.engine][{'Surrogate':'surr','FixedTopK':'fixed_topk','Random':'rand'}[r.policy]+f'_k{r.k}']
        check(abs(100*values.mean()-r.cer_pct)<1e-12,'Plotted mean '+str((r.block,r.engine,r.k,r.policy)))
    for name in ['headroom_policy_summary_allK.csv','headroom_nonshared_controls_allK.csv']:
        check((fig/name).read_bytes()==(bundle/'evidence/headroom'/name).read_bytes(),'Plotted headroom source '+name)
    main=(ROOT/'IJDAR_overleaf/main.tex').read_text();esm=(ROOT/'IJDAR_overleaf/ESM_1.tex').read_text()
    for snippet in ['and 612 SROIE receipts','in five of eight settings, with three',
            'with four differences significant after Holm correction','25.15\\% versus 28.12\\%',
            'Surrogate is significantly better at $K=3,5,10$','NN-K is significantly better at $K=3,5$',
            'changed from case-sensitive to casefolded CER after initial results',
            'can arise from the shared term and oracle constraint',
            'The effect of deskewing on OCR was not evaluated.']:
        check(snippet in main,'Current main narrative '+snippet)
    for snippet in ['same corrected 612 receipts','Seven of the 160 policy significance classifications changed',
            'None of the 64 direct Drop-minus-Full tests survives Holm correction',
            'original OCR strings, not casefolded strings','not a formal test of the difference between contrasts',
            'do not include refit, seed or split variability','593 retained pages change numerical fold label']:
        check(snippet in esm,'Current ESM narrative '+snippet)
    check('184/200' not in esm,'Historical seed diagnostic not attributed to612 CIs')
    check('no SROIE EasyOCR NN-K--Surrogate difference is significant' not in main,'Obsolete NN conclusion removed')
    referenced=set(re.findall(r'\\tableinput\{primary_tables/([^}]+)\}',main+esm))
    check(referenced==set(expected),'All13 row assets referenced')
    bib=(ROOT/'IJDAR_overleaf/reference.bib').read_text();keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
    cites={k.strip() for g in re.findall(r'\\cite\w*\{([^}]+)\}',main+esm) for k in g.split(',')}
    check(cites<=keys,'Citation coverage')
    check('not an independent noise estimate' in main and 'not an independent noise estimate' in esm,'Laplacian definition retained')
    check('NoPre bypasses grayscale conversion' in main and 'neither an identity nor a grayscale-only candidate' in main,'Color NoPre versus grayscale pool retained')
    for name in ['main.tex','main.pdf','ESM_1.tex','ESM_1.pdf','reference.bib']:
        check((ROOT/'IJDAR_overleaf'/name).read_bytes()==(ROOT/'ijdar_public_release/paper'/name).read_bytes(),'Paper public equality '+name)
    pages={}
    for name in ['main','ESM_1']:
        folder=ROOT/'IJDAR_overleaf';log=(folder/(name+'.log')).read_text()
        check('undefined' not in log.lower() and 'Fatal error' not in log,'Build undefined/fatal '+name)
        check('Overfull' not in log,'Build overfull '+name)
        check((folder/(name+'.pdf')).stat().st_mtime_ns>=(folder/(name+'.tex')).stat().st_mtime_ns,'PDF freshness '+name)
        info=subprocess.check_output(['pdfinfo',str(folder/(name+'.pdf'))],text=True)
        pages[name]=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    # Scan every new numeric payload; diagrams are in a separate allowed bundle.
    for p in public.rglob('*'):
        if not p.is_file():continue
        check(p.suffix not in ['.pkl','.docx','.png','.jpg','.jpeg'],'No data/model binary '+str(p.relative_to(public)))
        if p.suffix in ['.json','.csv','.md','.tex']:
            check('/home/aibqp/' not in p.read_text(),'No personal path '+str(p.relative_to(public)))
        if p.suffix=='.csv':
            with p.open(newline='') as f:fields=next(csv.reader(f))
            check(not (set(fields)&{'ocr_text','reference_text','prediction','text','label_text'}),'No full-text field '+str(p.relative_to(public)))
    check(pd.read_csv(bundle/'sroie_additional_exclusions.csv').columns.tolist()==['left','right','decision','review_method'],'Exclusions metadata whitelist')
    for name in ['export_receipt612_publication.py','replay_receipt612_publication.py','audit_receipt612_publication.py','test_receipt612_publication.py','SROIE_RECEIPT_PUBLICATION_PROTOCOL_20261019.md']:
        check(sha(ROOT/'analysis'/name)==sha(ROOT/'ijdar_public_release/code/analysis'/name),'Public code '+name)
    report=dict(audit='Corrected612 artifact consistency/provenance/arithmetic;not scientific validity,receipt recall or acceptance',
        authority=authority,checks_passed=len(good),checks_failed=len(bad),failures=bad,pdf_pages=pages,replay=result)
    (ROOT/'submission_audit_results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'claim_traceability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['claim_group','claim','source_artifact','status']);w.writeheader();w.writerows(trace)
    print(json.dumps(report,indent=2));return bool(bad)


if __name__=='__main__':sys.exit(main())
