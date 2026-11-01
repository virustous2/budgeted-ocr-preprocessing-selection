import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sanitize_publication_paths import generate, METADATA
from audit_portable_release import verify


class PortableReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'original'
        self.source.mkdir()
        files = {'table.csv': b'id,mean\na,0.12345678901234567\n'}
        for name in METADATA:
            files[name] = ('{"path": "'+str(self.root)+'/outputs/input.csv", "x": 0.12345678901234567}\n').encode()
        for name, data in files.items():
            p = self.source/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self.manifest = dict(inputs={'outputs/input.csv':'historical-hash'},
                             files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                             metric={'raw':'case-sensitive'})
        self.write_manifest()
        (self.source/'README.md').write_text('Historical result bundle\n')
        self.output = self.root/'portable'

    def write_manifest(self):
        (self.source/'manifest.json').write_text(json.dumps(self.manifest, indent=2)+'\n')

    def test_only_expected_paths_change_and_source_stays_frozen(self):
        original = (self.source/'manifest.json').read_bytes()
        generate(self.source, self.output, self.root)
        result = verify(self.source, self.output, self.root)
        self.assertEqual(result['changed_metadata_files'], 6)
        self.assertEqual(result['unchanged_csv_files'], 1)
        self.assertEqual((self.source/'manifest.json').read_bytes(), original)

    def test_no_overwrite(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            generate(self.source, self.output, self.root)

    def test_changed_source_hash_is_rejected_before_output(self):
        (self.source/'table.csv').write_bytes(b'corrupt\n')
        with self.assertRaises(ValueError):
            generate(self.source, self.output, self.root)
        self.assertFalse(self.output.exists())

    def test_private_path_in_numeric_csv_cannot_be_silently_changed(self):
        data = (str(self.root)+'/private\n').encode()
        (self.source/'table.csv').write_bytes(data)
        self.manifest['files']['table.csv'] = hashlib.sha256(data).hexdigest()
        self.write_manifest()
        with self.assertRaises(ValueError):
            generate(self.source, self.output, self.root)
        self.assertFalse(self.output.exists())

    def test_release_tampering_is_detected(self):
        generate(self.source, self.output, self.root)
        (self.output/'table.csv').write_bytes(b'changed\n')
        with self.assertRaises(AssertionError):
            verify(self.source, self.output, self.root)


if __name__ == '__main__':
    unittest.main()
