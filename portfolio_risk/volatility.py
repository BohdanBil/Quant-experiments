"""Volatility models used by filtered historical simulation.

Every model has:
    fit(returns, rv)            estimate parameters
    variance(returns, rv)       row t = variance forecast for day t from data up to t-1;
                                one extra row = forecast for the day after the sample
    initial_state / variance_of / step     roll the volatility forward inside a simulation

`returns` and `rv` are arrays of shape (days, stocks). `rv` is the range-based daily variance
from `data.range_variance`; only HAR uses it.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


class _RecursiveVolatility:
    """Simulation plumbing for models whose whole state is the current variance."""

    def paired_shocks(self, returns, rv):
        return None

    def initial_state(self, returns, rv, n_paths):
        return np.tile(self.variance(returns, rv)[-1], (n_paths, 1))

    def variance_of(self, state):
        return state

    def step(self, state, eps, paired):
        return self.update(state, eps)


class EWMA(_RecursiveVolatility):
    """RiskMetrics volatility: an exponentially weighted average of squared returns.
    Nothing to fit, and no pull back towards a long-run level."""

    name = "EWMA"

    def __init__(self, lam=0.94):
        self.lam = lam

    def fit(self, returns, rv=None):
        return self

    def variance(self, returns, rv=None):
        return ewma_variance(returns, self.lam)

    def update(self, var, eps):
        return self.lam * var + (1 - self.lam) * eps ** 2


class GJRGarch(_RecursiveVolatility):
    """GJR-GARCH(1,1) volatility for each stock:

        var[t] = omega + (alpha + gamma * [r[t-1] < 0]) * r[t-1]^2 + beta * var[t-1]

    Like EWMA, a big move raises tomorrow's volatility, but a fall raises it more than a rise
    of the same size (gamma, the leverage effect), and volatility drifts back to a long-run level.
    """

    name = "GJR-GARCH"

    def fit(self, returns, rv=None):
        params = np.array([fit_gjr_garch(column) for column in returns.T])
        self.omega, self.alpha, self.gamma, self.beta = params.T
        return self

    @property
    def persistence(self):
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def long_run_variance(self):
        return self.omega / (1 - self.persistence)

    def variance(self, returns, rv=None):
        return np.column_stack([
            gjr_variance(returns[:, i], self.omega[i], self.alpha[i], self.gamma[i],
                         self.beta[i], self.long_run_variance[i])
            for i in range(returns.shape[1])])

    def update(self, var, eps):
        return self.omega + (self.alpha + self.gamma * (eps < 0)) * eps ** 2 + self.beta * var


class HAR:
    """HAR volatility (Corsi, 2009) with a leverage term, on range-based daily variance:

        log rv[t] = b0 + b1 * log rv[t-1] + b2 * (5-day mean) + b3 * (22-day mean)
                    + b4 * fall[t-1] + b5 * (5-day mean of falls)

    where fall = min(return / sqrt(rv), 0): the day's fall measured against its own volatility.
    (In raw returns, a fall in a turbulent simulated state would raise volatility, causing a
    bigger fall, and so on without limit.)

    Three time scales (a day, a week, a month) let it follow short spikes and slower regime
    shifts at once, where GARCH has a single decay rate. Fitted by least squares per stock; a
    scale factor then turns exp(forecast) into the expected squared return.

    In a simulation each day's log rv = forecast + a historical surprise drawn from the same
    day as the return shock, so a big shock comes with the volatility jump it really had.
    """

    name = "HAR"
    N_COEF = 6

    def fit(self, returns, rv):
        X, y = self._design(returns, rv)
        X = X[:-1]
        self.coef = np.empty((returns.shape[1], self.N_COEF))
        self.scale = np.empty(returns.shape[1])
        for i in range(returns.shape[1]):
            ok = np.isfinite(X[:, i]).all(axis=1) & np.isfinite(y[:, i])
            self.coef[i] = np.linalg.lstsq(X[ok, i], y[ok, i], rcond=None)[0]
            self.scale[i] = np.mean(returns[ok, i] ** 2 / np.exp(X[ok, i] @ self.coef[i]))
        return self

    def variance(self, returns, rv):
        X, _ = self._design(returns, rv)
        return self.scale * np.exp(np.einsum("tik,ik->ti", X, self.coef))

    def paired_shocks(self, returns, rv):
        """Each day's surprise in log rv (actual minus forecast)."""
        X, y = self._design(returns, rv)
        return y - np.einsum("tik,ik->ti", X[:-1], self.coef)

    def initial_state(self, returns, rv, n_paths):
        log_rv = np.tile(np.log(rv[-22:]), (n_paths, 1, 1))
        falls = np.tile(np.minimum(returns[-5:] / np.sqrt(rv[-5:]), 0), (n_paths, 1, 1))
        return log_rv, falls, self._forecast(log_rv, falls)

    def variance_of(self, state):
        return self.scale * np.exp(state[2])

    def step(self, state, eps, paired):
        log_rv, falls, forecast = state
        today = forecast + paired
        log_rv = np.concatenate([log_rv[:, 1:], today[:, None]], axis=1)
        falls = np.concatenate([falls[:, 1:], np.minimum(eps / np.exp(today / 2), 0)[:, None]], axis=1)
        return log_rv, falls, self._forecast(log_rv, falls)

    def _forecast(self, log_rv, falls):
        """Forecast of tomorrow's log rv from the last 22 days of log rv and 5 days of falls."""
        X = np.stack([np.ones(log_rv.shape[::2]), log_rv[:, -1], log_rv[:, -5:].mean(axis=1),
                      log_rv.mean(axis=1), falls[:, -1], falls.mean(axis=1)], axis=-1)
        return np.einsum("nik,ik->ni", X, self.coef)

    @staticmethod
    def _design(returns, rv):
        """Regressors X (days + 1, stocks, 6): row t uses data up to day t-1, and the extra last
        row is for the day after the sample. Also returns y = log rv (days, stocks)."""
        y = np.log(rv)
        blank = np.full((1, y.shape[1]), np.nan)
        lagged = pd.DataFrame(np.vstack([blank, y]))
        falls = pd.DataFrame(np.vstack([blank, np.minimum(returns / np.sqrt(rv), 0)]))
        X = np.stack([np.ones(lagged.shape), lagged, lagged.rolling(5).mean(),
                      lagged.rolling(22).mean(), falls, falls.rolling(5).mean()], axis=-1)
        return X, y


def ewma_variance(returns, lam=0.94, burn_in=30):
    """RiskMetrics EWMA variance of each column (same row convention as `variance` above)."""
    var = np.empty((len(returns) + 1, returns.shape[1]))
    var[0] = returns[:burn_in].var(axis=0)
    for t, r in enumerate(returns):
        var[t + 1] = lam * var[t] + (1 - lam) * r ** 2
    return var


def gjr_variance(r, omega, alpha, gamma, beta, var0):
    """GJR-GARCH variance of one series, same row convention as `ewma_variance`.

    The recursion is a linear filter in the variance, so scipy runs it in compiled code.
    """
    x = omega + (alpha + gamma * (r < 0)) * r ** 2
    var = np.empty(len(r) + 1)
    var[0] = var0
    var[1:] = lfilter([1.0], [1.0, -beta], x, zi=[beta * var0])[0]
    return var


def fit_gjr_garch(r):
    """Quasi-maximum-likelihood GJR-GARCH(1,1) for one return series.

    Variance targeting fixes the long-run variance at the sample variance, leaving alpha,
    gamma and beta to estimate. Returns (omega, alpha, gamma, beta).
    """
    r = r - r.mean()
    v = r.var()

    def neg_log_lik(p):
        alpha, gamma, beta = p
        var = gjr_variance(r, v * (1 - alpha - gamma / 2 - beta), alpha, gamma, beta, v)[:-1]
        if (var <= 0).any():
            return 1e10
        return 0.5 * np.mean(np.log(var) + r ** 2 / var)

    result = minimize(neg_log_lik, x0=[0.03, 0.08, 0.9], method="SLSQP",
                      bounds=[(0, 0.5), (0, 0.5), (0, 0.999)], options={"ftol": 1e-10},
                      constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1] / 2 - p[2]})
    alpha, gamma, beta = result.x
    return v * (1 - alpha - gamma / 2 - beta), alpha, gamma, beta
