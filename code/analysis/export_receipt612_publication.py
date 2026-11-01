"""Portable, source-bound publication export of verified SROIE612 statistics.

No OCR/fitting. Never modifies old experiment/publication bundles. Existing
row renderer is reused without editing hash-locked production scripts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from export_publication_v3 import BLOCKS, ENGINES, LABEL, render, row, fmt, ci_fmt, pformat
from export_noise_feature_supplement import tables as historical_a_tables

ROOT = Path(__file__).resolve().parents[1]
CHANGE = 'CHG-20261019-06'
COUNTS = dict(raw=300, contour=300, sroie=612)
SOURCE = ROOT/'outputs/sroie_receipt_statistics_20261019_v1'
VERIFIED = ROOT/'analysis/outputs/sroie_receipt_statistics_verification_20261019_v1/verification.json'
PRIOR = ROOT/'outputs/publication_v3_casefold_20261010_v2'
FIGURES = ROOT/'outputs/publication_figure_clarity_20261016_v2'
SNAPSHOT = ROOT/'SROIRE_COMPARISON/output_sroire/receipt_dedup_20261016_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def archive(out):
    """Keep pre-edit sources/PDFs/authorities, including unrelated user edits."""
    out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT/'IJDAR_overleaf', ROOT/'PAPER_SOURCE_MAP.md',
             ROOT/'submission_audit_results.json', ROOT/'claim_traceability.csv',
             ROOT/'analysis/audit_submission_artifacts.py']
    paths += list((ROOT/'analysis').glob('*authority*.json'))
    paths += [ROOT/'ijdar_public_release'/n for n in ('README.md','MANIFEST.md')]
    for src in paths:
        dst = out/src.relative_to(ROOT)
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    write_json(out/'archive_manifest.json', {str(p.relative_to(out)):sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()})


def validate_source(source=SOURCE, verified=VERIFIED):
    report = json.loads(verified.read_text())
    if report['checks_failed'] or report['source_manifest_sha256'] != sha(source/'complete.json'):
        raise ValueError('Unverified/changed statistics')
    meta = json.loads((source/'complete.json').read_text())
    for name, digest in meta['files'].items():
        if sha(source/name) != digest: raise ValueError('Changed statistics: '+name)
    prepared=json.loads((source/'prepared.json').read_text())
    for bucket in ['sources','internal','code']:
        for name,digest in prepared[bucket].items():
            if sha(ROOT/name)!=digest:raise ValueError('Changed frozen dependency: '+name)
    return report


def load_bundle(bundle):
    read = lambda name: pd.read_csv(bundle/name)
    per, metrics, nn = {}, {}, {}
    for b in BLOCKS:
        metrics[b] = json.loads((bundle/'per_page_bestk'/b/'metrics.json').read_text())
        nn[b] = read(f'nn_baseline/{b}/nn_per_image.csv')
        for e in ENGINES[b]:
            per[b,e] = read(f'per_page_bestk/{b}/per_page_results_{e}.csv')
            if len(per[b,e]) != COUNTS[b] or per[b,e].image_name.nunique() != COUNTS[b]:
                raise ValueError('Wrong publication cohort')
    return dict(per=per, metrics=metrics, nn=nn,
        stats=read('evidence/bestk/statistics.csv'),
        static=read('evidence/static/strict_selector_static_paired_comparison.csv'),
        policy=read('evidence/policy/strict_selector_policy_paired_comparison.csv'),
        benefit=read('evidence/descriptives/preprocessing_benefit.csv'),
        quality=read('evidence/descriptives/model_quality.csv'),
        coupled=read('evidence/headroom/headroom_policy_summary_allK.csv'),
        nonshared=read('evidence/headroom/headroom_nonshared_controls_allK.csv'),
        zero=read('evidence/zero/zero_sensitivity.csv').rename(columns={'p_holm':'p_holm_12'}))


def a_tables(direct, changes):
    body = historical_a_tables(direct, changes)
    n = int(changes.changed.sum())
    return body.replace('All ten policy comparisons', f'All {n} policy comparisons')


def metric_tables(stats):
    lines = [r'\begin{table}[H]',r'\centering\small',
        r'\caption{Matched 612-receipt best-of-$K$ sensitivity: Holm-significant lower-CER counts. Each metric/contrast has a separate eight-test family; A/B identifies the lower-CER policy.}',
        r'\label{tab:sroie_casefold_counts}',r'\begin{tabular}{lcc}\toprule',
        r'Contrast (A vs B) & Case-sensitive A/B & Casefold A/B \\\midrule']
    contrasts = [('Surrogate','Random'),('Surrogate','FixedTopK'),('NN','Random'),('NN','FixedTopK'),('NN','Surrogate')]
    for a,b in contrasts:
        cells = [a.replace('NN','NN-K')+' vs '+b]
        for metric in ['case_sensitive','casefold']:
            g = stats.query('metric==@metric and method_a==@a and method_b==@b')
            if len(g) != 8: raise ValueError('Incomplete metric family')
            cells.append(f'{sum((g.p_holm<.05)&(g.difference_pp<0))}/{sum((g.p_holm<.05)&(g.difference_pp>0))}')
        lines.append(row(cells).rstrip())
    lines += [r'\bottomrule\end{tabular}\end{table}', '',r'\begin{table}[H]',r'\centering\small',
        r'\caption{NN-K minus Surrogate on the same 612 receipts and folds (pp). CS: case-sensitive sensitivity; CF: primary casefold. CIs are per-comparison 95\% paired percentile intervals, not multiplicity-adjusted; $p_H$ uses each metric\textquotesingle s eight-test family. Nonzero rounded-zero endpoints retain their signs. Significance follows Wilcoxon--Holm, not mean-CI zero exclusion.}',
        r'\label{tab:sroie_casefold_nn}',r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lr r r r r r r}\toprule',
        r'Engine & $K$ & CS $\Delta$ & CS 95\% CI & CS $p_H$ & CF $\Delta$ & CF 95\% CI & CF $p_H$ \\\midrule']
    for e in ENGINES['sroie']:
        for k in [3,5,10,15]:
            cells = [LABEL[e],k]
            for metric in ['case_sensitive','casefold']:
                g = stats.query('metric==@metric and engine==@e and k==@k and method_a=="NN" and method_b=="Surrogate"')
                if len(g) != 1: raise ValueError('Ambiguous metric row')
                r = g.iloc[0]
                cells += [fmt(r.difference_pp,signed=True),f'[{ci_fmt(r.ci_low_pp)}, {ci_fmt(r.ci_high_pp)}]',pformat(r.p_holm)]
            lines.append(row(cells).rstrip())
    return '\n'.join(lines+[r'\bottomrule\end{tabular}}\end{table}',''])


def generate(out):
    if out.exists(): raise FileExistsError(out)
    report = validate_source(); config = json.loads((SOURCE/'configuration.json').read_text())
    out.mkdir(parents=True); sources = {}
    def copy(src, name):
        src = Path(src); dst=out/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst); sources[str(src.relative_to(ROOT))]=sha(src)
    # Numeric snapshots preserve every retained source field string.
    for b in BLOCKS:
        spec = config['bestk']['primary'][b]
        for e in ENGINES[b]:
            copy(ROOT/spec['surrogate']/f'per_page_results_{e}.csv',f'per_page_bestk/{b}/per_page_results_{e}.csv')
        copy(ROOT/spec['surrogate']/'metrics.json',f'per_page_bestk/{b}/metrics.json')
        copy(ROOT/spec['nn']/'nn_per_image.csv',f'nn_baseline/{b}/nn_per_image.csv')
        copy(ROOT/spec['folds'],f'response/{b}/fold_assignments.csv')
        copy(ROOT/config['data'][b]['data'],f'response/{b}/preprocessing_data.csv')
        copy(ROOT/config['data'][b]['nopre'],f'response/{b}/nopre.csv')
        for variant,key in [('full','surrogate'),('rank_only','surrogate'),('nn_k','policy'),('fixed_topk','policy')]:
            src=ROOT/config['selectors'][b][key]/variant/'nested_selector_candidate_scores.csv'
            dst=out/'selectors'/b/variant/'scores.csv'; dst.parent.mkdir(parents=True,exist_ok=True)
            with src.open(newline='') as fi,dst.open('w',newline='') as fo:
                reader=csv.DictReader(fi); fields=[c for c in reader.fieldnames if c!='ocr_text']
                writer=csv.DictWriter(fo,fieldnames=fields); writer.writeheader()
                for r in reader: writer.writerow({c:r[c] for c in fields})
            sources[str(src.relative_to(ROOT))]=sha(src)
    mapping = {
        'bestk/primary/statistics.csv':'evidence/bestk/statistics.csv',
        'bestk/primary/paired_values.csv':'evidence/bestk/paired_values.csv',
        'selectors/policy.csv':'evidence/policy/strict_selector_policy_paired_comparison.csv',
        'selectors/static.csv':'evidence/static/strict_selector_static_paired_comparison.csv',
        'zero_sensitivity/statistics.csv':'evidence/zero/zero_sensitivity.csv'}
    for family in ['headroom','descriptives']:
        for p in (SOURCE/family).glob('*'):
            if p.suffix in ['.csv','.json']: mapping[str(p.relative_to(SOURCE))]='evidence/'+str(p.relative_to(SOURCE))
    for src,dst in mapping.items(): copy(SOURCE/src,dst)
    for name in ['paired_conditions.csv','direct_drop_minus_full.csv','holm_changes_drop_vs_full.csv','holm_changes_full_vs_primary.csv']:
        copy(SOURCE/'A'/name,'sensitivity/A/'+name)
    for c in ['a_full','drop_orig_noise']:
        copy(SOURCE/'bestk'/c/'statistics.csv',f'sensitivity/A/{"full" if c=="a_full" else c}/statistics.csv')
    for name in ['statistics.csv','paired.csv']:
        copy(SOURCE/'metric_sensitivity'/name,'sensitivity/letter_case/'+name)
    copy(PRIOR/'vietnamese_cohort_manifest.csv','vietnamese_cohort_manifest.csv')
    # Only identity/decision metadata is public, not visual receipt-content notes.
    exclusion=SNAPSHOT/'excluded_pairs.csv'
    with exclusion.open(newline='') as fi,(out/'sroie_additional_exclusions.csv').open('w',newline='') as fo:
        fields=['left','right','decision','review_method']
        writer=csv.DictWriter(fo,fieldnames=fields);writer.writeheader()
        for r in csv.DictReader(fi):writer.writerow({c:r[c] for c in fields})
    sources[str(exclusion.relative_to(ROOT))]=sha(exclusion)
    # Retain exact original numeric field strings for independent A/metric replay.
    # Parsed/re-serialized paired convenience CSVs alone need not retain all bits.
    for condition in ['a_full','drop_orig_noise','case_sensitive']:
        for b,spec in config['bestk'][condition].items():
            for e in ENGINES[b]:
                copy(ROOT/spec['surrogate']/f'per_page_results_{e}.csv',f'sensitivity/input_pages/{condition}/{b}/per_page_results_{e}.csv')
            copy(ROOT/spec['nn']/'nn_per_image.csv',f'sensitivity/input_pages/{condition}/{b}/nn_per_image.csv')
    # File hashes only, never benchmark transcriptions or image bytes.
    folds=pd.read_csv(out/'response/sroie/fold_assignments.csv')
    cohort=[]
    for r in folds.itertuples():
        image=next((SNAPSHOT/'images').glob(r.image_name+'.*'))
        cohort.append(dict(image_name=r.image_name,fold=int(r.fold),image_sha256=sha(image),
            reference_sha256=sha(SNAPSHOT/'ground_truth'/(r.image_name+'.txt'))))
    pd.DataFrame(cohort).to_csv(out/'sroie_cohort_manifest.csv',index=False)
    copy(ROOT/'analysis/SROIE_RECEIPT_PUBLICATION_PROTOCOL_20261019.md','PUBLICATION_PROTOCOL.md')
    copy(ROOT/'analysis/NOISE_FEATURE_SENSITIVITY_PROTOCOL.md','sensitivity/A/HISTORICAL_DESIGN_PROTOCOL.md')
    (out/'tables').mkdir()
    for name,body in render(load_bundle(out),compact_sroie=True,preserve_effect_sign=True).items():
        (out/'tables'/name).write_text('% '+CHANGE+'; source-bound SROIE612 numeric export.\n'+body)
    (out/'noise_feature_tables.tex').write_text(a_tables(pd.read_csv(out/'sensitivity/A/direct_drop_minus_full.csv'),pd.read_csv(out/'sensitivity/A/holm_changes_drop_vs_full.csv')))
    (out/'sroie_casefold_tables.tex').write_text(metric_tables(pd.read_csv(out/'sensitivity/letter_case/statistics.csv')))
    write_json(out/'verification_summary.json',dict(report,verification_scope='Private independent source replay; not scientific validity or screening-recall certificate'))
    readme='''# Corrected SROIE612 publication evidence

VN300 raw/contour use frozen v3 case-sensitive references; SROIE612 casefolds
both sides. Both members of two additional same-receipt pairs are excluded.
All affected SROIE training layers were refit on new GroupKFold partitions;
old/new differences are not a duplicate-only causal effect. No OCR was rerun.
The 2027-candidate review is not a guarantee of exhaustive duplicate recall.

Primary:224 paired comparisons/21 Holm families. A:384/36. Matched metric
sensitivity:80/10. Some comparisons are deliberately repeated across analyses;
these are not 688 unique hypotheses. All means/CI endpoints/p-values remain
full precision in CSVs. CIs:10000 paired-page percentile draws,unadjusted and
conditional on these fitted folds,not refit/seed/split uncertainty.

Replay from the public repository root (NumPy,pandas,SciPy;no OCR/training):

    python code/analysis/replay_receipt612_publication.py --bundle results/BUNDLE_NAME

All seven A classification changes and adverse primary outcomes are released.
No equivalence or selector feature-removal robustness claim. Historical616
BCa/seed diagnostics do not validate new612 CIs. Saved model-quality summaries
are not independent reconstructions of unavailable raw OOF predictions.

No images,crops,full transcriptions,OCR strings,caches or model pickle files
are included. Provenance paths are project-relative identifiers;upstream private
inputs are not all included. Old616 releases are historical,not current results.
'''
    (out/'README.md').write_text(readme.replace('BUNDLE_NAME',out.name))
    # Fail closed for accidental text/path exposure in this new release.
    for p in out.rglob('*'):
        if not p.is_file(): continue
        if str(ROOT) in p.read_text(): raise ValueError('Personal absolute path: '+str(p))
        if p.suffix=='.csv':
            with p.open(newline='') as f: fields=next(csv.reader(f))
            if set(fields)&{'ocr_text','reference_text','prediction','text','label_text'}:
                raise ValueError('Forbidden full-text column: '+str(p))
    write_json(out/'manifest.json',dict(change=CHANGE,kind='corrected_receipt612_publication',counts=COUNTS,
        source_manifest_sha256=sha(SOURCE/'complete.json'),source_verification_sha256=sha(VERIFIED),
        inputs=sources,exporter_sha256=sha(__file__),
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print('EXPORTED:',out)


def figures(bundle,out):
    """VN and pipeline unchanged; headroom regenerated using corrected CSVs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_oracle_gap_diagnostic import draw_distribution, POLICY_ORDER, COMPARISON_ORDER, POLICY_LABELS
    from export_figure_clarity import GAIN_LABELS
    out.mkdir(parents=True,exist_ok=False)
    names=['curve_points.csv','raw_pareto_multiengine.png','contour_pareto_multiengine.png',
        'pipeline_flowchart_2_5.pdf','pipeline_base.pdf','pipeline_flowchart_2_5.tex','pipeline_flowchart.drawio']
    for name in names: shutil.copyfile(FIGURES/name,out/name)
    coupled=pd.read_csv(bundle/'evidence/headroom/headroom_policy_summary_allK.csv')
    controls=pd.read_csv(bundle/'evidence/headroom/headroom_nonshared_controls_allK.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.25),constrained_layout=True)
    draw_distribution(axes[0],coupled,'policy',POLICY_ORDER,'spearman_headroom_vs_advantage',POLICY_LABELS,'(a) Coupled FixedTopK-based statistic')
    draw_distribution(axes[1],controls,'comparison',COMPARISON_ORDER,'spearman_headroom_vs_gain',GAIN_LABELS,'(b) Non-shared policy-gain controls')
    for suffix in ['pdf','png']: fig.savefig(out/('headroom_policy_controls.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)
    for name in ['headroom_policy_summary_allK.csv','headroom_nonshared_controls_allK.csv']:
        shutil.copyfile(bundle/'evidence/headroom'/name,out/name)
    write_json(out/'manifest.json',dict(change=CHANGE,bundle_manifest_sha256=sha(bundle/'manifest.json'),
        previous_figure_manifest_sha256=sha(FIGURES/'manifest.json'),exporter_sha256=sha(__file__),
        plot_helper_sha256=sha(ROOT/'analysis/plot_oracle_gap_diagnostic.py'),
        paper_assets=['raw_pareto_multiengine.png','contour_pareto_multiengine.png','pipeline_flowchart_2_5.pdf','headroom_policy_controls.pdf','headroom_policy_controls.png'],
        files={p.name:sha(p) for p in sorted(out.iterdir())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['archive','export','figures'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    if args.stage=='archive': archive(args.output_dir)
    elif args.stage=='export': generate(args.output_dir)
    else: figures(args.bundle,args.output_dir)
