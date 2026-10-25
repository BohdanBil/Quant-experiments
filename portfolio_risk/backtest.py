"""Out-of-sample backtest of 1-day VaR and ES forecasts.

Every day each model is re-estimated using only the days *before* it, and its forecast is
compared with the loss that actually happened. The portfolio keeps constant weights.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from .models import ewma_variance, fit_t_dof
from .risk import normal_var_es, student_t_var_es, var_es

LEVELS = (0.05, 0.025, 0.01)          # tail probabilities: 95%, 97.5% and 99%


@dataclass
class Backtest:
    dates: pd.DatetimeIndex
    losses: np.ndarray                # realised daily loss, fraction of portfolio value
    forecasts: dict                   # model -> array (days, len(LEVELS), 2) of [VaR, ES]

    def var(self, model, alpha):
        return self.forecasts[model][:, LEVELS.index(alpha), 0]

    def es(self, model, alpha):
        return self.forecasts[model][:, LEVELS.index(alpha), 1]


def run_backtest(returns, weights, window=500, fhs_window=1000, original_window=205,
                 lam=0.94, refit_every=20):
    """1-day forecasts from five models, for every day once all of them have enough history.

    `returns` are daily simple returns; the Student-t degrees of freedom are refitted
    every `refit_every` days (the volatility is updated daily).
    """
    r = returns.to_numpy()
    w = weights.loc[returns.columns].to_numpy(dtype=float)
    port = r @ w
    sd = np.sqrt(ewma_variance(r, lam))       # sd[t]: volatility forecast for day t
    shocks = r / sd[:-1]
    start = fhs_window + 60                   # skip the EWMA warm-up period
    names = {
        "original": f"Original main.py (normal + drift, {original_window}d)",
        "normal": f"Normal ({window}d)",
        "t": f"Student-t ({window}d)",
        "hs": f"Historical ({window}d)",
        "fhs": f"Filtered HS (EWMA, {fhs_window}d)",
    }
    out = {name: np.empty((len(r) - start, len(LEVELS), 2)) for name in names.values()}
    for i, t in enumerate(range(start, len(r))):
        past = port[t - window:t]
        short = port[t - original_window:t]
        sigma = past.std(ddof=1)
        if i % refit_every == 0:
            x = r[t - window:t] - r[t - window:t].mean(axis=0)
            nu = fit_t_dof(x, np.cov(x, rowvar=False))
        fhs = shocks[t - fhs_window:t] @ (w * sd[t])    # past shocks at today's volatility
        fhs -= fhs.mean()
        for j, alpha in enumerate(LEVELS):
            out[names["original"]][i, j] = normal_var_es(short.std(ddof=1), alpha, short.mean())
            out[names["normal"]][i, j] = normal_var_es(sigma, alpha)
            out[names["t"]][i, j] = student_t_var_es(sigma, nu, alpha)
            out[names["hs"]][i, j] = var_es(past, alpha)
            out[names["fhs"]][i, j] = var_es(fhs, alpha)
    return Backtest(returns.index[start:], -port[start:], out)


def kupiec_pof(hits, alpha):
    """Kupiec proportion-of-failures test: p-value for 'the breach rate equals alpha'."""
    n, x = len(hits), int(np.sum(hits))
    lr = -2 * (xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
               - xlogy(n - x, 1 - x / n) - xlogy(x, x / n))
    return stats.chi2.sf(lr, 1)


def christoffersen_independence(hits):
    """Christoffersen test: p-value for 'breaches do not cluster' (today's breach does not
    make tomorrow's more likely)."""
    hits = np.asarray(hits, dtype=bool)
    if not hits.any():
        return np.nan
    prev, curr = hits[:-1], hits[1:]
    n00, n01 = np.sum(~prev & ~curr), np.sum(~prev & curr)
    n10, n11 = np.sum(prev & ~curr), np.sum(prev & curr)
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0     # no breach before the last day
    p = (n01 + n11) / len(curr)
    ll_independent = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll_markov = (xlogy(n00, 1 - p01) + xlogy(n01, p01)
                 + xlogy(n10, 1 - p11) + xlogy(n11, p11))
    return stats.chi2.sf(-2 * (ll_independent - ll_markov), 1)


def traffic_light(breaches):
    """Basel zone for the number of 99% VaR breaches in 250 trading days."""
    return "green" if breaches <= 4 else "yellow" if breaches <= 9 else "red"


def es_ratio(losses, var, es):
    """Average realised loss / forecast ES on days the VaR was breached.

    About 1 means ES is right on average; above 1 means it understated the tail.
    """
    hit = losses > var
    return np.mean(losses[hit] / es[hit]) if hit.any() else np.nan


def summary(bt):
    rows = {}
    for model in bt.forecasts:
        hit95 = bt.losses > bt.var(model, 0.05)
        hit99 = bt.losses > bt.var(model, 0.01)
        rolling99 = pd.Series(hit99).rolling(250).sum()
        rows[model] = {
            "95% breaches": hit95.mean(),
            "95% Kupiec p": kupiec_pof(hit95, 0.05),
            "99% breaches": hit99.mean(),
            "99% Kupiec p": kupiec_pof(hit99, 0.01),
            "99% clustering p": christoffersen_independence(hit99),
            "worst 250d": int(rolling99.max()),
            "last 250d": int(rolling99.iloc[-1]),
            "ES 97.5% ratio": es_ratio(bt.losses, bt.var(model, 0.025), bt.es(model, 0.025)),
        }
    return pd.DataFrame(rows).T
