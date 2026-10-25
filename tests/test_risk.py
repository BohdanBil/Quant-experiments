import unittest

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_risk import risk
from portfolio_risk.data import FIELDS, clean_prices, range_variance


class RiskMeasureTest(unittest.TestCase):
    def test_var_and_es_are_positive_losses_matching_the_normal_formula(self):
        pnl = np.random.default_rng(0).normal(0, 100, 2_000_000)
        var, es = risk.var_es(pnl, 0.01)
        expected_var, expected_es = risk.normal_var_es(100, 0.01)
        self.assertAlmostEqual(var, expected_var, delta=1)
        self.assertAlmostEqual(es, expected_es, delta=1)
        self.assertGreater(es, var)

    def test_student_t_closed_form_matches_sampling(self):
        nu, sigma = 4.0, 0.02
        x = stats.t.rvs(nu, size=2_000_000, random_state=1) * sigma * np.sqrt((nu - 2) / nu)
        for alpha in (0.05, 0.01):
            for got, want in zip(risk.var_es(x, alpha), risk.student_t_var_es(sigma, nu, alpha)):
                self.assertAlmostEqual(got / want, 1, delta=0.02)

    def test_component_es_adds_up_to_es(self):
        cov = [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1]]
        holdings = np.random.default_rng(2).multivariate_normal([0, 0, 0], cov, 100_000)
        _, es = risk.var_es(holdings.sum(axis=1), 0.025)
        self.assertAlmostEqual(risk.component_es(holdings, 0.025).sum(), es, places=8)

    def test_max_drawdown(self):
        values = np.array([[100, 120, 90, 130, 117]])
        self.assertAlmostEqual(risk.max_drawdown(values)[0], 0.25)

    def test_min_variance_weights_are_long_only_and_sum_to_one(self):
        cov = np.diag([0.04, 0.01, 0.09])
        w = risk.min_variance_weights(cov)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w >= -1e-9).all())
        np.testing.assert_allclose(w, np.array([1 / 0.04, 1 / 0.01, 1 / 0.09]) / 136.11, atol=1e-3)


class DataTest(unittest.TestCase):
    def test_clean_prices_drops_gaps_and_stale_rows(self):
        idx = pd.date_range("2024-01-01", periods=5)
        prices = pd.DataFrame({"A": [1.0, 1.1, 1.1, np.nan, 1.2],
                               "B": [2.0, 2.1, 2.1, 2.2, 2.3]}, index=idx)
        self.assertEqual(list(clean_prices(prices).index), [idx[0], idx[1], idx[4]])

    def test_range_variance_uses_the_days_range_and_skips_bad_opens(self):
        # day 2 opens at a bad print of 5.0 although the stock closed at 10.2 on both days
        ohlc = pd.DataFrame([[10, 10, 10, 10], [10.0, 10.5, 9.8, 10.2], [5.0, 10.4, 5.0, 10.2]],
                            index=pd.date_range("2024-01-01", periods=3),
                            columns=pd.MultiIndex.from_product([FIELDS, ["A"]]), dtype=float)
        rv = range_variance(ohlc)["A"]
        garman_klass = 0.5 * np.log(10.5 / 9.8) ** 2 - (2 * np.log(2) - 1) * np.log(10.2 / 10) ** 2
        self.assertAlmostEqual(rv.iloc[0], garman_klass)
        self.assertLess(rv.iloc[1], 1e-3)       # not the ~0.27 the bad print would imply


if __name__ == "__main__":
    unittest.main()
