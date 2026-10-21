import unittest

import numpy as np
import pandas as pd

from taleb_kurtosis.digital_tail_smile import interpolate, mean_curve, smile_figure, matched_zones


class SmileTests(unittest.TestCase):
    def test_no_extrapolation_or_wide_gap(self):
        actual = interpolate([0, .2, 1], [.2, .4, .8], [-.1, 0, .1, .5, 1, 1.1])
        np.testing.assert_allclose(actual, [np.nan, .2, .3, np.nan, .8, np.nan], equal_nan=True)

    def test_unsorted_points(self):
        np.testing.assert_allclose(interpolate([.2, 0], [.4, .2], [.1]), [.3])

    def test_equal_day_mean_and_coverage(self):
        curves = pd.DataFrame(dict(date=["a", "b", "a", "b"], sigma=[-3, -3, -4, -4],
                                   iv=[.2, .4, .5, np.nan], relative_iv=[1, 2, 2.5, np.nan]))
        means = mean_curve(curves).set_index("sigma")
        self.assertAlmostEqual(means.loc[-3, "iv"], .3)
        self.assertEqual(means.loc[-4, "days"], 1)
        self.assertEqual(len(smile_figure(curves).data), 3)

    def test_zones_use_identical_days(self):
        curves = pd.DataFrame(dict(date=["a"] * 3 + ["b"] * 3, sigma=[-3.5, -3, -2.5] * 2,
                                   iv=[.4, .3, .2, np.nan, .8, .6], relative_iv=[2, 1.5, 1, np.nan, 4, 3]))
        matched = matched_zones(curves)
        self.assertTrue(matched.days.eq(1).all())
        np.testing.assert_allclose(matched.iv, [.4, .3, .2])


if __name__ == "__main__":
    unittest.main()
