"""Versioned display-only table overlay. Never rewrites numeric experiment data."""
import argparse
import json
from pathlib import Path
from export_publication_v3 import ROOT,load,render,sha,EVIDENCE

def export(output, compact_sroie=False, preserve_effect_sign=False):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    change='CHG-20261011-05' if preserve_effect_sign else 'CHG-20261011-01' if compact_sroie else 'CHG-20261010-08'
    for name,body in render(load(),compact_sroie=compact_sroie,preserve_effect_sign=preserve_effect_sign).items():
        (output/name).write_text('% Display-only '+change+'; numeric evidence unchanged.\n'+body)
    meta={'change':change,'compact_sroie':compact_sroie,'preserve_effect_sign':preserve_effect_sign,'numeric_bundle':'outputs/publication_v3_casefold_20261010_v2',
          'numeric_bundle_manifest_sha256':sha(ROOT/'outputs/publication_v3_casefold_20261010_v2/manifest.json'),
          'selector_statistics_sha256':sha(ROOT/EVIDENCE['policy']/'strict_selector_policy_paired_comparison.csv'),
          'renderer_sha256':sha(Path(__file__).with_name('export_publication_v3.py')),
          'files':{p.name:sha(p) for p in sorted(output.iterdir())}}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Generated display-only overlay:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--compact-sroie',action='store_true')
    p.add_argument('--preserve-effect-sign',action='store_true')
    args=p.parse_args();export(args.output_dir,args.compact_sroie,args.preserve_effect_sign)
