"""Monte Carlo engine: buy-and-hold portfolio paths from any return model."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Simulation:
    """Simulated paths for one model."""

    model: str
    weights: np.ndarray
    value: float
    values: np.ndarray        # (n_paths, horizon + 1) portfolio value; column 0 is today
    asset_returns: dict       # day -> (n_paths, n_stocks) cumulative return of each stock

    def pnl(self, day):
        return self.values[:, day] - self.value

    def holdings_pnl(self, day, weights=None):
        """P&L of each holding on `day`. Other weights revalue the same scenarios."""
        w = self.weights if weights is None else np.asarray(weights)
        return self.value * w * self.asset_returns[day]


def simulate(model, weights, value, horizon, n_paths, rng, report_days, chunk=10_000):
    """Buy the target weights today and hold them (no rebalancing) for `horizon` days.

    `weights` is a pandas Series indexed by ticker, so it can never be applied to the
    wrong stock. Paths are generated in chunks to keep memory use small.
    """
    w = weights.loc[model.tickers].to_numpy(dtype=float)
    values = np.empty((n_paths, horizon + 1))
    values[:, 0] = value
    asset_returns = {d: np.empty((n_paths, len(w))) for d in report_days}
    for start in range(0, n_paths, chunk):
        stop = min(start + chunk, n_paths)
        growth = np.exp(np.cumsum(model.sample(rng, stop - start, horizon), axis=1))
        values[start:stop, 1:] = value * growth @ w
        for d in report_days:
            asset_returns[d][start:stop] = growth[:, d - 1] - 1
    return Simulation(model.name, w, value, values, asset_returns)
