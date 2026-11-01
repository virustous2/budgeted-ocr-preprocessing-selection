"""Copy only manifest-bound numeric/figure assets and current paper to public.

Requires the private pre-edit archive; preserves all previous versioned bundles.
No git mutation, OCR or fitting. Rebuilding PDFs is a separate explicit command.
"""
import json
from pathlib import Path
import shutil
from export_receipt612_publication import ROOT, sha


def copy_bundle(source, dest):
    manifest=json.loads((source/'manifest.json').read_text())
    if dest.exists() and (dest/'manifest.json').read_bytes()!=(source/'manifest.json').read_bytes():
        raise ValueError('Refuse incompatible public bundle overwrite: '+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    for name,digest in manifest['files'].items():
        src=source/name
        if sha(src)!=digest:raise ValueError('Changed source: '+str(src))
        dst=dest/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copy2(source/'manifest.json',dest/'manifest.json')


def main():
    archive=ROOT/'analysis/outputs/receipt612_publication_before_20261019_v1/archive_manifest.json'
    if not archive.exists():raise ValueError('No pre-edit rollback archive')
    cfg=json.loads((ROOT/'analysis/publication_authority_receipt612.json').read_text())
    bundle=ROOT/cfg['bundle'];fig=ROOT/cfg['figures']
    copy_bundle(bundle,ROOT/cfg['public_bundle']);copy_bundle(fig,ROOT/cfg['public_figures'])
    for src in (bundle/'tables').glob('*.tex'):
        shutil.copy2(src,ROOT/'IJDAR_overleaf/primary_tables'/src.name)
    for name in ['sroie_casefold_tables.tex','noise_feature_tables.tex']:
        shutil.copy2(bundle/name,ROOT/'IJDAR_overleaf'/name)
    fm=json.loads((fig/'manifest.json').read_text())
    for name in fm['paper_assets']:shutil.copy2(fig/name,ROOT/'IJDAR_overleaf/figures'/name)
    target=ROOT/'ijdar_public_release/paper'
    for name in ['main.tex','main.pdf','ESM_1.tex','ESM_1.pdf','reference.bib','sroie_casefold_tables.tex','noise_feature_tables.tex']:
        shutil.copy2(ROOT/'IJDAR_overleaf'/name,target/name)
    for src in (ROOT/'IJDAR_overleaf/primary_tables').glob('*.tex'):
        shutil.copy2(src,target/'primary_tables'/src.name)
    for name in fm['paper_assets']:shutil.copy2(fig/name,target/'figures'/name)
    names=['export_receipt612_publication.py','sync_receipt612_publication.py','replay_receipt612_publication.py',
        'audit_receipt612_publication.py','test_receipt612_publication.py','SROIE_RECEIPT_PUBLICATION_PROTOCOL_20261019.md',
        'publication_authority_receipt612.json']
    for name in names:shutil.copy2(ROOT/'analysis'/name,ROOT/'ijdar_public_release/code/analysis'/name)
    print('Synchronized local numeric/figure/paper copies; rebuild both PDFs and run again before audit.')


if __name__=='__main__':main()
