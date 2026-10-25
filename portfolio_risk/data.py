"""Download, cache and clean daily prices."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(tickers, start, end, suffix=".AX", cache_dir="data", refresh=False):
    """Dividend-adjusted daily closes, one column per ticker, in the order given.

    The download is cached to CSV, so later runs use identical data (and give identical
    results) instead of whatever Yahoo returns that day.
    """
    cache = Path(cache_dir) / f"prices_{'-'.join(tickers)}_{start}_{end}.csv"
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = yf.download([t + suffix for t in tickers], start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"No prices downloaded for {tickers}; check tickers and connection")
        prices = raw["Close"].rename(columns=lambda c: c.removesuffix(suffix))
        cache.parent.mkdir(exist_ok=True)
        prices.to_csv(cache)
    # yfinance sorts columns alphabetically; restore the caller's order
    return clean_prices(prices[list(tickers)])


def clean_prices(prices):
    """Keep days on which every stock has a price, and drop stale rows: exchange holidays
    that Yahoo fills with the previous close for every stock."""
    prices = prices.dropna()
    stale = (prices.diff() == 0).all(axis=1)
    return prices[~stale]


def log_returns(prices):
    return np.log(prices).diff().dropna()


def simple_returns(prices):
    return prices.pct_change(fill_method=None).dropna()
