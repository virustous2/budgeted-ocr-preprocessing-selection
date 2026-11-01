"""NN helper edge cases; no OCR, model fitting or experiment-output writes."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_nn_shortlist_baseline import safe_wilcoxon


class ZeroHandlingTests(unittest.TestCase):
    def test_near_equal_nonzero_differences_are_not_discarded(self):
        x = np.arange(1, 7, dtype=float) / 10
        y = x + 1e-8
        self.assertTrue(np.allclose(x, y))
        self.assertFalse(np.array_equal(x, y))
        got = safe_wilcoxon(x, y)
        expected = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        self.assertEqual(got["p"], expected.pvalue)
        self.assertEqual(got["p"], 0.03125)
        self.assertEqual(got["stat"], expected.statistic)

    def test_all_exact_zeros_use_documented_degenerate_convention(self):
        x = [0.1, 0.2, 0.3]
        self.assertEqual(safe_wilcoxon(x, x),
                         {"stat": 0.0, "p": 1.0, "n": 3, "alternative": "two-sided"})

    def test_empty_pairs_remain_undefined(self):
        for x, y in [([], []), ([np.nan, np.inf], [0.1, 0.2])]:
            got = safe_wilcoxon(x, y)
            self.assertEqual(got["n"], 0)
            self.assertTrue(math.isnan(got["stat"]))
            self.assertTrue(math.isnan(got["p"]))

    def test_finite_mask_and_mixed_zero_pairs_match_scipy(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, np.nan, 0.6, 0.7])
        y = np.array([0.1, 0.19, 0.32, 0.4, 0.5, np.inf, 0.69])
        mask = np.isfinite(x) & np.isfinite(y)
        for alternative in ["two-sided", "less", "greater"]:
            got = safe_wilcoxon(x, y, alternative=alternative)
            expected = wilcoxon(x[mask], y[mask], alternative=alternative, zero_method="wilcox")
            self.assertEqual(got["p"], expected.pvalue)
            self.assertEqual(got["stat"], expected.statistic)
            self.assertEqual(got["n"], int(mask.sum()))
            self.assertEqual(got["alternative"], alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
