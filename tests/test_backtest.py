import unittest

import numpy as np
import pandas as pd

from portfolio_risk import backtest


class BacktestTest(unittest.TestCase):
    def test_kupiec_accepts_the_right_rate_and_rejects_double(self):
        hits = np.zeros(10_000, dtype=bool)
        hits[:100] = True                                   # exactly 1%
        self.assertAlmostEqual(backtest.kupiec_pof(hits, 0.01), 1.0)
        hits[:200] = True                                   # 2%: far too many breaches
        self.assertLess(backtest.kupiec_pof(hits, 0.01), 1e-6)

    def test_christoffersen_detects_clustering(self):
        rng = np.random.default_rng(8)
        independent = rng.random(10_000) < 0.02
        clustered = np.zeros(10_000, dtype=bool)
        for start in rng.choice(10_000 - 5, 40, replace=False):
            clustered[start:start + 5] = True
        self.assertGreater(backtest.christoffersen_independence(independent), 0.01)
        self.assertLess(backtest.christoffersen_independence(clustered), 1e-6)

    def test_traffic_light_zones(self):
        zones = [backtest.traffic_light(n) for n in (0, 4, 5, 9, 10)]
        self.assertEqual(zones, ["green", "green", "yellow", "yellow", "red"])

    def test_normal_model_is_calibrated_on_normal_data(self):
        rng = np.random.default_rng(9)
        returns = pd.DataFrame(rng.normal(0, 0.01, (3000, 3)), columns=list("ABC"),
                               index=pd.bdate_range("2000-01-03", periods=3000))
        bt = backtest.run_backtest(returns, pd.Series(1 / 3, index=list("ABC")),
                                   window=250, fhs_window=500, original_window=100)
        rate = (bt.losses > bt.var("Normal (250d)", 0.01)).mean()
        self.assertTrue(0.004 < rate < 0.018, rate)


if __name__ == "__main__":
    unittest.main()
