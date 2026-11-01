import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from export_publication_v3 import fmt,ci_fmt,pformat,row,one,generate,load,render
from replay_publication_v3 import holm,selected
import pandas as pd

class PublicationTests(unittest.TestCase):
    def test_selector_effect_signed_zero_only(self):
        data=load()
        old=render(data,compact_sroie=True)
        new=render(data,compact_sroie=True,preserve_effect_sign=True)
        for name in old:
            if name!='selector.tex':self.assertEqual(old[name],new[name])
        self.assertEqual(old['selector.tex'].replace(' & 0.00 & 0.483 & 0.966', ' & +0.00 & 0.483 & 0.966'),new['selector.tex'])
        self.assertEqual(sum(a!=b for a,b in zip(old['selector.tex'].splitlines(),new['selector.tex'].splitlines())),1)
    def test_compact_sroie_preserves_cer(self):
        data=load();old=render(data);new=render(data,compact_sroie=True)
        for name in old:
            if name!='sroie_results.tex':self.assertEqual(old[name],new[name])
        before=old['sroie_results.tex'].splitlines();after=new['sroie_results.tex'].splitlines()
        self.assertEqual(len(after),4)
        for i,line in enumerate(before):
            cells=line.removesuffix(' \\\\').split(' & ')
            for j,k in enumerate((3,15)):
                self.assertEqual(' & '.join([cells[0],str(k)]+cells[4:7]+cells[7+4*j:11+4*j])+' \\\\',after[2*i+j])
    def test_rounding(self):
        self.assertEqual(fmt(-.000485),'0.00')
        self.assertEqual(fmt(-.000485,signed=True),'0.00')
        self.assertEqual(pformat(.3),'0.300')
        self.assertEqual(pformat(.25),'0.250')
        self.assertTrue(row(['a','b']).endswith('\\\\\n'))
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):generate(Path(d))
    def test_ci_signed_zero(self):
        self.assertEqual(ci_fmt(.004694),'+0.00')
        self.assertEqual(ci_fmt(-.000485),'-0.00')
        self.assertEqual(ci_fmt(0),'0.00')
        self.assertEqual(ci_fmt(-1.09804),'-1.10')
    def test_holm(self):
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])
    def test_no_duplicate_lookup(self):
        with self.assertRaises(ValueError):one(pd.DataFrame({'x':[1,1]}),x=1)
    def test_selection_unique(self):
        df=pd.DataFrame({'engine':['easyocr']*2,'image_name':['a']*2,'actual_cer':[.1,.2],'selected_by_nested_model':[True,True]})
        with self.assertRaises(ValueError):selected(df,'easyocr')

if __name__=='__main__':unittest.main()
