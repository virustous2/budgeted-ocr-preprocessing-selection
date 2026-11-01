"""Cheap formatting/export/replay-guard tests; no OCR or model fitting."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from export_noise_feature_supplement import signed, tables, generate
from replay_noise_feature_sensitivity import replay


class SupplementTests(unittest.TestCase):
    def test_signed_rounded_zero(self):
        self.assertEqual(signed(.0004),'+0.00')
        self.assertEqual(signed(-.0004),'-0.00')
        self.assertEqual(signed(0),'0.00')
        self.assertEqual(signed(-.176548),'-0.18')

    def test_tables_retain_all_changed_rows_and_p_precision(self):
        # Synthetic formatting fixture: works in public without private outputs.
        direct=pd.DataFrame([dict(block=b,policy=p,difference_pp=-.1,p_holm=.01)
            for b in ('raw','contour','sroie') for p in ('Surrogate','NN')])
        pairs=[('Surrogate','FixedTopK')]*2+[('NN','FixedTopK')]*3+[('NN','Surrogate')]*5
        changes=pd.DataFrame([dict(block='raw',engine='vietocr',k=3,method_a=a,method_b=b,
            difference_pp_a=-.1,difference_pp_b=.1,p_holm_a=.3,p_holm_b=.25,changed=True)
            for a,b in pairs])
        body=tables(direct,changes)
        self.assertEqual(body.count('Surr.--FixedTopK'),2)
        self.assertEqual(body.count('NN-K--FixedTopK'),3)
        self.assertEqual(body.count('NN-K--Surr.'),5)
        self.assertIn('0.300 & 0.250',body)
        self.assertIn('1/0',body)
        self.assertEqual(body.count(r'\begin{table}'),2)

    def test_export_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                generate(Path('unused'),Path('unused'),Path(d))

    def test_replay_refuses_wrong_kind_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'manifest.json').write_text(json.dumps({'kind':'primary'}))
            with self.assertRaises(ValueError):replay(root)
            (root/'x.csv').write_text('wrong')
            (root/'manifest.json').write_text(json.dumps({'kind':'supplementary_best_of_K_not_primary','files':{'x.csv':'wronghash'}}))
            with self.assertRaises(ValueError):replay(root)


if __name__=='__main__':unittest.main()
