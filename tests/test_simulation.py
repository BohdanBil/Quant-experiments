import unittest

import numpy as np
import pandas as pd

from portfolio_risk import risk
from portfolio_risk.models import (EWMA, FilteredHistoricalModel, GJRGarch, NormalModel,
                                   StudentTModel, fit_gjr_garch, fit_t_dof, gjr_variance)
from portfolio_risk.simulation import simulate


class ConstantModel:
    """Every path earns the same fixed daily log return per stock."""

    name = "constant"

    def __init__(self, tickers, daily):
        self.tickers, self.daily = tickers, np.asarray(daily)

    def sample(self, rng, n_paths, horizon):
        return np.broadcast_to(self.daily, (n_paths, horizon, len(self.tickers))).copy()


class SimulationTest(unittest.TestCase):
    def test_weights_follow_tickers_not_position(self):
        model = ConstantModel(["A", "B"], [0.01, 0.0])
        weights = pd.Series({"B": 1.0, "A": 0.0})        # deliberately in the other order
        sim = simulate(model, weights, 10_000, 5, 3, np.random.default_rng(0), [5])
        np.testing.assert_allclose(sim.values, 10_000)

    def test_buy_and_hold_compounds_each_holding(self):
        model = ConstantModel(["A", "B"], [np.log(1.1), 0.0])
        sim = simulate(model, pd.Series({"A": 0.5, "B": 0.5}), 100, 2, 1,
                       np.random.default_rng(0), [2])
        self.assertAlmostEqual(sim.values[0, -1], 50 * 1.1 ** 2 + 50)

    def test_monte_carlo_matches_closed_form_under_normality(self):
        returns = pd.DataFrame({"A": np.random.default_rng(3).normal(0, 0.012, 5000)})
        model = NormalModel(returns)
        sim = simulate(model, pd.Series({"A": 1.0}), 10_000, 10, 400_000,
                       np.random.default_rng(4), [10])
        sigma = np.sqrt(model.cov[0, 0] * 10)
        for alpha in (0.05, 0.01):
            for got, want in zip(risk.var_es(sim.pnl(10), alpha), risk.lognormal_var_es(sigma, alpha)):
                self.assertAlmostEqual(got / (10_000 * want), 1, delta=0.015)


class ModelTest(unittest.TestCase):
    def test_t_fit_recovers_degrees_of_freedom(self):
        rng = np.random.default_rng(5)
        cov = np.array([[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]]) * 1e-4
        z = rng.multivariate_normal(np.zeros(3), cov, 20_000)
        x = z * np.sqrt(3 / rng.chisquare(5, (20_000, 1)))       # multivariate t, nu = 5
        self.assertAlmostEqual(fit_t_dof(x, np.cov(x, rowvar=False)), 5, delta=0.7)

    def test_student_t_model_keeps_the_sample_variance(self):
        rng = np.random.default_rng(6)
        returns = pd.DataFrame(rng.standard_t(6, (2000, 2)) * 0.01, columns=["A", "B"])
        model = StudentTModel(returns)
        day1 = model.sample(rng, 200_000, 1)[:, 0]
        np.testing.assert_allclose(day1.var(axis=0), np.diag(model.cov), rtol=0.05)

    def test_filtered_hs_has_zero_mean_and_current_volatility(self):
        rng = np.random.default_rng(7)
        calm, storm = rng.normal(0, 0.01, (1500, 2)), rng.normal(0, 0.03, (100, 2))
        model = FilteredHistoricalModel(pd.DataFrame(np.vstack([calm, storm])), window=1000)
        np.testing.assert_allclose(model.shocks.mean(axis=0), 0, atol=1e-12)
        self.assertTrue((model.sigma > 0.02).all())       # the recent storm sets today's volatility
        day1 = model.sample(rng, 200_000, 1)[:, 0]
        np.testing.assert_allclose(day1.std(axis=0), model.sigma * model.shocks.std(axis=0), rtol=0.02)

    def test_models_have_zero_expected_simple_return(self):
        rng = np.random.default_rng(10)
        returns = pd.DataFrame(rng.standard_t(6, (2000, 2)) * 0.015, columns=["A", "B"])
        for model in (NormalModel(returns), StudentTModel(returns), FilteredHistoricalModel(returns)):
            growth = np.exp(model.sample(rng, 400_000, 1)[:, 0])
            np.testing.assert_allclose(growth.mean(axis=0), 1, atol=1e-4, err_msg=model.name)


def simulate_gjr(n, omega, alpha, gamma, beta, rng):
    r = np.empty(n)
    var = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * var
    return r


class GarchTest(unittest.TestCase):
    PARAMS = (3e-6, 0.04, 0.10, 0.88)        # omega, alpha, gamma, beta: long-run variance 1e-4

    def test_variance_filter_matches_the_recursion(self):
        r = np.random.default_rng(11).normal(0, 0.01, 500)
        slow = [1e-4]
        for x in r:
            slow.append(2e-6 + (0.05 + 0.1 * (x < 0)) * x ** 2 + 0.85 * slow[-1])
        np.testing.assert_allclose(gjr_variance(r, 2e-6, 0.05, 0.1, 0.85, 1e-4), slow, rtol=1e-10)

    def test_fit_recovers_parameters(self):
        r = simulate_gjr(20_000, *self.PARAMS, np.random.default_rng(12))
        _, alpha, gamma, beta = fit_gjr_garch(r)
        self.assertAlmostEqual(alpha, 0.04, delta=0.02)
        self.assertAlmostEqual(gamma, 0.10, delta=0.04)
        self.assertAlmostEqual(beta, 0.88, delta=0.03)

    def test_garch_volatility_reverts_but_ewma_does_not(self):
        garch = GJRGarch()
        garch.omega, garch.alpha, garch.gamma, garch.beta = (np.array([p]) for p in self.PARAMS)
        rng = np.random.default_rng(13)
        start = np.full((50_000, 1), 4e-4)            # 4x the long-run variance
        for vol, target in ((garch, 1e-4), (EWMA(), 4e-4)):
            var = start
            for _ in range(300):
                var = vol.update(var, rng.standard_normal(var.shape) * np.sqrt(var))
            self.assertAlmostEqual(var.mean() / target, 1, delta=0.1, msg=vol.name)


if __name__ == "__main__":
    unittest.main()
