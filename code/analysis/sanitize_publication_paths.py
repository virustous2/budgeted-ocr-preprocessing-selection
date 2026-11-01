"""Create a fresh metadata-only portable release; no OCR, fits or CSV rewrites."""
import argparse
import hashlib
import json
from pathlib import Path

METADATA = frozenset({
    'evidence/bestk/run_metadata.json', 'evidence/descriptives/run_metadata.json',
    'evidence/headroom/input_provenance.json', 'evidence/policy/run_metadata.json',
    'evidence/static/run_metadata.json', 'evidence/zero/run_metadata.json',
})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate(source, output, project_root):
    source, output, project_root = map(lambda p: Path(p).resolve(), (source, output, project_root))
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and destination must be separate')
    original_manifest = (source/'manifest.json').read_bytes()
    manifest = json.loads(original_manifest)
    prefix = (str(project_root)+'/').encode()
    listed = set(manifest['files']) | {'README.md', 'manifest.json'}
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    if actual != listed:
        raise ValueError('Unexpected or missing source files')
    payload, changes = {}, {}
    for name, expected_hash in manifest['files'].items():
        path = source/name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('Unsafe source path: '+name)
        data = path.read_bytes()
        if digest(data) != expected_hash:
            raise ValueError('Source hash mismatch: '+name)
        if prefix in data:
            if name not in METADATA:
                raise ValueError('Private prefix in unexpected payload: '+name)
            clean = data.replace(prefix, b'')  # Preserve all other bytes, including numbers.
            json.loads(clean)  # Metadata must remain valid JSON.
            changes[name] = dict(source_sha256=expected_hash, released_sha256=digest(clean),
                                 replacement_count=data.count(prefix))
            data = clean
        if b'/home/' in data:
            raise ValueError('Unmapped home path: '+name)
        payload[name] = data
    if set(changes) != METADATA:
        raise ValueError('Expected exactly the six historical metadata files')
    readme = (source/'README.md').read_text().replace(
        'results/primary_v3_casefold_20261010_v2', 'results/'+output.name)
    readme += ('\n## Portable metadata release — CHG-20261014-07\n\n'
               'Only six provenance JSON files have their original project-root prefix removed.\n'
               'These relative paths identify upstream project artifacts; they are not promises\n'
               'that excluded private inputs exist in this public repository. All numeric CSVs,\n'
               'tables, folds and model-output records are byte-identical to the source v2 bundle.\n'
               'The release manifest retains the source manifest hash, original input hashes,\n'
               'and before/after hashes for each sanitized metadata file. Figure/display\n'
               'provenance still refers to that original numeric-source manifest.\n')
    payload['README.md'] = readme.encode()
    manifest['files'] = {name: digest(data) for name, data in sorted(payload.items())}
    manifest['release'] = dict(change='CHG-20261014-07',
                               source_manifest_sha256=digest(original_manifest),
                               sanitized_files=changes,
                               path_basis='Relative to original project root')
    manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
    if prefix in manifest_bytes or b'/home/' in manifest_bytes or b'/home/' in payload['README.md']:
        raise ValueError('Private path in manifest or README')
    output.mkdir(parents=True)
    for name, data in payload.items():
        dest = output/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (output/'manifest.json').write_bytes(manifest_bytes)
    print(f'Portable release: {output}; six metadata files changed; numeric payload unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output_dir, args.project_root)
