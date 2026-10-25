"""Risk measures.

Convention: VaR and ES are reported as positive numbers meaning *losses*. `alpha` is the
tail probability, so alpha=0.01 is the 99% level.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def var_es(pnl, alpha):
    """VaR and Expected Shortfall of a P&L (or return) sample."""
    losses = -np.asarray(pnl)
    var = np.quantile(losses, 1 - alpha)
    return var, losses[losses >= var].mean()


def mc_standard_error(pnl, alpha, n_batches=20):
    """Monte Carlo standard error of (VaR, ES), from independent batches of paths."""
    estimates = np.array([var_es(batch, alpha) for batch in np.array_split(pnl, n_batches)])
    return estimates.std(axis=0, ddof=1) / np.sqrt(n_batches)


def sampling_spread(pnl, alpha, n_paths, q=(5, 50, 95)):
    """How much (VaR, ES) would vary if only `n_paths` paths were simulated.

    Splits the sample into independent groups of `n_paths` and returns percentiles `q` of
    the estimates across groups, shape (2, len(q)).
    """
    pnl = np.asarray(pnl)
    groups = pnl[: len(pnl) // n_paths * n_paths].reshape(-1, n_paths)
    estimates = np.array([var_es(g, alpha) for g in groups])
    return np.percentile(estimates, q, axis=0).T


def normal_var_es(sigma, alpha, mean=0.0):
    """Closed-form VaR and ES (fractions of value) for normally distributed returns."""
    z = stats.norm.ppf(1 - alpha)
    return z * sigma - mean, sigma * stats.norm.pdf(z) / alpha - mean


def lognormal_var_es(sigma, alpha):
    """Closed-form VaR and ES (fractions of value) for a lognormal value with zero expected
    return, i.e. log return ~ N(-sigma^2 / 2, sigma^2)."""
    z = stats.norm.ppf(1 - alpha)
    return 1 - np.exp(-sigma ** 2 / 2 - z * sigma), 1 - stats.norm.cdf(-z - sigma) / alpha


def student_t_var_es(sigma, nu, alpha):
    """Closed-form VaR and ES (fractions of value) for Student-t returns with std `sigma`."""
    scale = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(1 - alpha, nu)
    return scale * q, scale * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / alpha


def component_es(holdings_pnl, alpha):
    """Euler allocation of ES: each holding's average loss in the scenarios beyond VaR.

    The components add up exactly to the portfolio ES.
    """
    total = holdings_pnl.sum(axis=1)
    var, _ = var_es(total, alpha)
    return -holdings_pnl[-total >= var].mean(axis=0)


def bootstrap_normal_var(log_returns, weights, alpha, rng, n_boot=1000):
    """1-day closed-form VaR re-estimated on resampled copies of the data (parameter risk)."""
    port = log_returns.to_numpy() @ weights
    sigmas = rng.choice(port, size=(n_boot, len(port))).std(axis=1, ddof=1)
    return lognormal_var_es(sigmas, alpha)[0]


def max_drawdown(values):
    """Largest peak-to-trough fall of each path, as a fraction of the peak."""
    peaks = np.maximum.accumulate(values, axis=1)
    return (1 - values / peaks).max(axis=1)


def min_variance_weights(cov):
    """Long-only minimum-variance weights."""
    n = len(cov)
    scaled = cov / np.mean(np.diag(cov))    # daily variances are tiny; rescale so tolerances bite
    result = minimize(lambda w: w @ scaled @ w, np.full(n, 1 / n), jac=lambda w: 2 * scaled @ w,
                      method="SLSQP", bounds=[(0, 1)] * n, options={"ftol": 1e-12},
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return result.x


def stress_loss(log_returns, weights, start, end):
    """Loss (fraction) of today's buy-and-hold portfolio over a historical period."""
    idx = log_returns.index
    window = log_returns[(idx > pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    return -((np.exp(window.sum()) - 1) * weights).sum()


def worst_loss(log_returns, weights, days):
    """Worst buy-and-hold loss (fraction) over any `days`-day window, and its last day."""
    growth = np.exp(log_returns.rolling(days).sum()).dropna()
    loss = -((growth - 1) * weights).sum(axis=1)
    return loss.max(), loss.idxmax()
