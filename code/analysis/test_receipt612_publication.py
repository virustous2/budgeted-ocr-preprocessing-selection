"""Cheap export/cohort/display regression tests; no OCR or fitting."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from export_receipt612_publication import generate, load_bundle, metric_tables, a_tables, validate_source
from replay_receipt612_publication import COUNTS, replay


class ReceiptPublicationTests(unittest.TestCase):
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):generate(Path(d))

    def test_reject_unverified_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'verification.json').write_text(json.dumps({'checks_failed':1}))
            with self.assertRaises(ValueError):validate_source(p,p/'verification.json')

    def test_reject_old_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps(dict(kind='corrected_receipt612_publication',counts=dict(raw=300,contour=300,sroie=616))))
            with self.assertRaises(ValueError):replay(p)

    def test_incomplete_metric_families_rejected(self):
        frame=pd.DataFrame([dict(metric='casefold',method_a='Surrogate',method_b='Random',p_holm=.01,difference_pp=-1)])
        with self.assertRaises(ValueError):metric_tables(frame)

    def test_dynamic_a_caption(self):
        direct=pd.DataFrame([dict(block='sroie',policy='Surrogate',difference_pp=.1,p_holm=.2)])
        changes=pd.DataFrame([dict(block='sroie',engine='tesseract',k=15,method_a='NN',method_b='Surrogate',
            changed=True,difference_pp_a=.1,difference_pp_b=.15,p_holm_a=.125,p_holm_b=.0306)])
        body=a_tables(direct,changes)
        self.assertIn('All 1 policy comparisons',body);self.assertNotIn('All ten',body)
        self.assertIn('0.125 & 0.0306',body)

    def test_counts_explicit(self):
        self.assertEqual(COUNTS,{'raw':300,'contour':300,'sroie':612})


if __name__=='__main__':unittest.main()
