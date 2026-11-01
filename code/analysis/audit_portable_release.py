"""Independent byte-level verification of metadata-only path sanitization."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_METADATA = {
    'evidence/bestk/run_metadata.json', 'evidence/descriptives/run_metadata.json',
    'evidence/headroom/input_provenance.json', 'evidence/policy/run_metadata.json',
    'evidence/static/run_metadata.json', 'evidence/zero/run_metadata.json',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(source, release, project_root, public=None, contour_source=None, contour_public=None):
    source, release, project_root = map(Path, (source, release, project_root))
    old_bytes = (source/'manifest.json').read_bytes()
    old = json.loads(old_bytes)
    new = json.loads((release/'manifest.json').read_text())
    prefix = (str(project_root.resolve())+'/').encode()
    assert new['release']['source_manifest_sha256'] == sha(old_bytes)
    assert {k:v for k,v in new.items() if k not in ('files', 'release')} == {
        k:v for k,v in old.items() if k != 'files'}
    assert set(new['files']) == set(old['files']) | {'README.md'}
    changed, csv_count = set(), 0
    for name, original_hash in old['files'].items():
        before = (source/name).read_bytes()
        after = (release/name).read_bytes()
        assert sha(before) == original_hash, name
        assert after == (before.replace(prefix, b'') if name in EXPECTED_METADATA else before), name
        if before != after:
            changed.add(name)
            assert json.loads(after) is not None
            record = new['release']['sanitized_files'][name]
            assert record == dict(source_sha256=sha(before), released_sha256=sha(after),
                                  replacement_count=before.count(prefix)), name
        if name.endswith('.csv'):
            assert before == after, name
            csv_count += 1
    assert changed == EXPECTED_METADATA
    assert set(new['release']['sanitized_files']) == changed
    assert {p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()} == set(new['files']) | {'manifest.json'}
    for name, recorded in new['files'].items():
        data = (release/name).read_bytes()
        assert sha(data) == recorded, name
        assert b'/home/' not in data, name
        if public is not None:
            assert data == (Path(public)/name).read_bytes(), name
    assert b'/home/' not in (release/'manifest.json').read_bytes()
    if public is not None:
        assert (release/'manifest.json').read_bytes() == (Path(public)/'manifest.json').read_bytes()
    if contour_source is not None:
        assert contour_public is not None
        assert Path(contour_public).read_bytes() == Path(contour_source).read_bytes().replace(prefix, b'')
        assert b'/home/' not in Path(contour_public).read_bytes()
    return dict(changed_metadata_files=len(changed), unchanged_csv_files=csv_count,
                source_manifest_sha256=sha(old_bytes), release_manifest_sha256=sha((release/'manifest.json').read_bytes()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--public', type=Path)
    parser.add_argument('--contour-source', type=Path)
    parser.add_argument('--contour-public', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = verify(args.source, args.release, args.project_root, args.public,
                    args.contour_source, args.contour_public)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
