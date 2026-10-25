"""Download, cache and clean daily prices."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FIELDS = ["Open", "High", "Low", "Close"]


def load_ohlc(tickers, start, end, suffix=".AX", cache_dir="data", refresh=False):
    """Dividend-adjusted daily open, high, low and close; columns are (field, ticker).

    The download is cached to CSV, so later runs use identical data (and give identical
    results) instead of whatever Yahoo returns that day.
    """
    cache = Path(cache_dir) / f"ohlc_{'-'.join(tickers)}_{start}_{end}.csv"
    if cache.exists() and not refresh:
        ohlc = pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True)
    else:
        raw = yf.download([t + suffix for t in tickers], start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"No prices downloaded for {tickers}; check tickers and connection")
        ohlc = raw[FIELDS].rename(columns=lambda c: c.removesuffix(suffix), level=1)
        cache.parent.mkdir(exist_ok=True)
        ohlc.to_csv(cache)
    # yfinance sorts tickers alphabetically; restore the caller's order
    ohlc = ohlc.reindex(columns=pd.MultiIndex.from_product([FIELDS, list(tickers)]))
    return ohlc.loc[clean_prices(ohlc["Close"]).index]


def load_prices(tickers, start, end, **kwargs):
    """Cleaned dividend-adjusted closes, one column per ticker."""
    return load_ohlc(tickers, start, end, **kwargs)["Close"]


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


def suspicious_opens(ohlc):
    """Opening prices that are almost certainly bad prints: an overnight gap above 8% that is
    also more than three times the day's close-to-close move."""
    close = ohlc["Close"]
    gap = np.log(ohlc["Open"] / close.shift(1)).abs()
    return (gap > 0.08) & (gap > 3 * np.log(close).diff().abs())


def range_variance(ohlc):
    """Each day's variance measured from its prices: the overnight gap squared plus the
    Garman-Klass estimate from the open, high, low and close.

    Because it uses the whole day's range, it registers a turbulent day even when the close
    barely moved. Days with a bad opening price or missing prices fall back to the squared
    close-to-close return. Rows line up with `log_returns` (the first day is dropped).
    """
    o, h, l, c = (ohlc[f] for f in FIELDS)
    close_move = np.log(c).diff()
    overnight = np.log(o / c.shift(1))
    intraday = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    rv = overnight ** 2 + intraday
    rv = rv.where(~suspicious_opens(ohlc) & (rv > 0)).fillna(close_move ** 2)
    return rv.clip(lower=0.01 * rv.median(), axis=1).iloc[1:]
