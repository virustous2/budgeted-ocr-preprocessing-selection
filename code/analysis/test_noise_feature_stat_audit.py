"""Synthetic checks of independent A arithmetic audit; no OCR/fitting."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from audit_noise_feature_statistics import check_row, independent_holm, coverage, change_check


class ArithmeticAuditTests(unittest.TestCase):
    def make_row(self, x, y, seed=123):
        d = x-y
        p = 1. if (d == 0).all() else wilcoxon(x,y,alternative="two-sided",zero_method="wilcox").pvalue
        rng = np.random.default_rng(seed)
        ci = 100*np.quantile(d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1),[.025,.975])
        return SimpleNamespace(n=len(d),n_zero=int((d == 0).sum()),bootstrap_seed=seed,
            bootstrap_replicates=10000,alternative="two-sided",zero_method="wilcox",
            ci_method="paired_page_percentile",ci_multiplicity_adjusted=False,
            p_two_sided=p,difference_pp=100*d.mean(),ci_low_pp=ci[0],ci_high_pp=ci[1])

    def test_known_holm_and_invalid_p_rejected(self):
        np.testing.assert_allclose(independent_holm([.04,.01,.03]),[.06,.03,.06])
        with self.assertRaises(ValueError):
            independent_holm([np.nan,.1])

    def test_exact_zero_and_seed_guard(self):
        x = np.ones(6)
        row = self.make_row(x,x)
        self.assertEqual(check_row(row,x,x,123),1.)
        with self.assertRaises(ValueError):
            check_row(row,x,x,124)

    def test_small_nonzero_is_not_silently_zeroed(self):
        x = np.full(6,1e-8); y = np.zeros(6)
        row = self.make_row(x,y)
        self.assertEqual(check_row(row,x,y,123),.03125)

    def test_duplicate_and_missing_keys_rejected(self):
        valid = pd.DataFrame({"key":["a","b"]})
        coverage(valid,["key"],{("a",),("b",)})
        for frame in (valid.iloc[:1],pd.concat([valid,valid.iloc[:1]])):
            with self.assertRaises(ValueError):
                coverage(frame,["key"],{("a",),("b",)})

    def test_significant_direction_flip_counted(self):
        keys = dict(block="raw",engine="vietocr",k=3,method_a="NN",method_b="Surrogate")
        before = pd.DataFrame([{**keys,"difference_pp":-.1,"p_holm":.01}])
        after = pd.DataFrame([{**keys,"difference_pp":.1,"p_holm":.01}])
        saved = pd.DataFrame([{**keys,"difference_pp_a":-.1,"difference_pp_b":.1,
            "p_holm_a":.01,"p_holm_b":.01,"state_a":-1,"state_b":1,"changed":True}])
        self.assertEqual(change_check(before,after,saved),1)
        saved.loc[0,"changed"] = False
        with self.assertRaises(AssertionError):
            change_check(before,after,saved)


if __name__ == "__main__":
    unittest.main()
