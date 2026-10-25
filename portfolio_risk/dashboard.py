"""Turns analysis results into JSON for the web dashboard (see web/ and serve.py)."""
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import backtest as bt_tests
from . import risk

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
MEASURES = (("var95", 0.05, 0), ("var99", 0.01, 0), ("es975", 0.025, 1))
REGIMES = {"2004–09 (GFC)": ("2004", "2009"), "2010–19": ("2010", "2019"),
           "2020–26 (COVID)": ("2020", "2026")}


def clean(obj):
    """Make results JSON-friendly: plain types, floats to 6 significant digits, NaN to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.ndarray, pd.Index, pd.Series)):
        return clean(list(obj))
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(f"{obj:.6g}") if math.isfinite(obj) else None
    return obj


def measures(pnl):
    """VaR/ES (positive losses) with Monte Carlo standard errors."""
    out = {}
    for key, alpha, k in MEASURES:
        out[key] = risk.var_es(pnl, alpha)[k]
        out[key + "_se"] = risk.mc_standard_error(pnl, alpha)[k]
    return out


def histogram(pnls, bins=90):
    """Shared-bin histograms (share of paths per bin) for several P&L samples."""
    lo, hi = np.percentile(np.concatenate(list(pnls.values())), [0.01, 99.99])
    edges = np.linspace(lo, hi, bins + 1)
    return {"edges": edges,
            "share": {name: np.histogram(p, edges)[0] / len(p) for name, p in pnls.items()}}


def path_risk(sim, value, stop_loss):
    drawdown = risk.max_drawdown(sim.values)
    floor = value * (1 - stop_loss)
    counts, edges = np.histogram(drawdown, bins=50, range=(0, max(0.5, np.percentile(drawdown, 99.9))))
    return {"median_drawdown": np.median(drawdown), "p95_drawdown": np.percentile(drawdown, 95),
            "p_down_any": (sim.values.min(axis=1) <= floor).mean(),
            "p_down_end": (sim.values[:, -1] <= floor).mean(),
            "drawdown_hist": {"edges": edges, "share": counts / len(drawdown)}}


def stress_tests(log_returns, weights, value, periods, horizons):
    rows = []
    for name, (start, end) in periods.items():
        if log_returns.index[0] < pd.Timestamp(start):
            loss = risk.stress_loss(log_returns, weights, start, end)
            rows.append({"name": name, "start": start, "end": end, "loss": loss, "loss_aud": loss * value})
    for n in sorted(set(horizons)):
        loss, last_day = risk.worst_loss(log_returns, weights, n)
        rows.append({"name": f"Worst {n}-day period since {log_returns.index[0]:%Y}",
                     "end": last_day, "days": n, "loss": loss, "loss_aud": loss * value})
    return rows


def portfolio_summary(sims, main, weights, value, days, log_returns, stress_periods,
                      stop_loss=0.10, put_strike=0.95, n_sample_paths=40):
    """Everything about one portfolio's simulated risk; `main` is one of `sims`."""
    horizon = days[-1]
    contrib_day = 10 if 10 in days else horizon
    comp = risk.component_es(main.holdings_pnl(contrib_day), 0.025)
    payoff = np.maximum(put_strike * value - main.values[:, -1], 0)
    hedged = main.pnl(horizon) + payoff - payoff.mean()
    return {
        "weights": dict(weights.items()),
        "value": value,
        "horizon": horizon,
        "days": days,
        "main_model": main.model,
        "measures": {s.model: {d: measures(s.pnl(d)) for d in days} for s in sims},
        "fan": {s.model: {f"p{q}": row for q, row in
                          zip(PERCENTILES, np.percentile(s.values, PERCENTILES, axis=0))} for s in sims},
        "sample_paths": main.values[:n_sample_paths],
        "histograms": {d: histogram({s.model: s.pnl(d) for s in sims}) for d in days},
        "contributions": {"day": contrib_day, "tickers": list(weights.index), "weights": weights.values,
                          "es": comp, "share": comp / comp.sum(), "total": comp.sum()},
        "path_risk": {s.model: path_risk(s, value, stop_loss) for s in sims},
        "stop_loss": stop_loss,
        "put": {"strike": put_strike, "premium": payoff.mean(),
                "before": measures(main.pnl(horizon)), "after": measures(hedged)},
        "stress": stress_tests(log_returns, weights, value, stress_periods, (10, horizon)),
    }


def backtest_section(bt, main_model):
    summary = bt_tests.summary(bt)
    regimes = []
    for label, (a, b) in REGIMES.items():
        sel = (bt.dates >= f"{a}-01-01") & (bt.dates <= f"{b}-12-31")
        for model in bt.forecasts:
            hits = (bt.losses > bt.var(model, 0.01))[sel]
            regimes.append({"regime": label, "model": model, "days": int(sel.sum()),
                            "breaches": hits.mean(), "kupiec_p": bt_tests.kupiec_pof(hits, 0.01),
                            "clustering_p": bt_tests.christoffersen_independence(hits)})
    pairs = {}
    for model in bt.forecasts:
        hits = bt.losses > bt.var(model, 0.01)
        pairs[model] = bt.dates[1:][hits[1:] & hits[:-1]]
    return {
        "dates": bt.dates, "losses": bt.losses, "main_model": main_model,
        "var99": {m: bt.var(m, 0.01) for m in bt.forecasts},
        "var95": {main_model: bt.var(main_model, 0.05)},
        "summary": [{"model": m, **row} for m, row in summary.to_dict(orient="index").items()],
        "regimes": regimes,
        "back_to_back": pairs,
    }


def market_section(prices, simple_returns, log_returns, weights, corr_window=500):
    weekly = prices.resample("W-FRI").last().dropna()
    portfolio = (1 + simple_returns @ weights.loc[simple_returns.columns]).cumprod()
    port_vol = (simple_returns @ weights.loc[simple_returns.columns]).rolling(63).std() * np.sqrt(252)
    port_vol = port_vol.resample("W-FRI").last().dropna()
    return {
        "weekly_dates": weekly.index,
        "growth": {t: 100 * weekly[t] / weekly[t].iloc[0] for t in prices.columns},
        "portfolio_growth": 100 * portfolio.resample("W-FRI").last().reindex(weekly.index).ffill().fillna(1),
        "vol_dates": port_vol.index,
        "portfolio_vol": port_vol,
        "correlation": {"tickers": list(log_returns.columns), "days": corr_window,
                        "matrix": log_returns.iloc[-corr_window:].corr().values,
                        "matrix_full": log_returns.corr().values, "since": log_returns.index[0]},
    }


def convergence_section(sim, day, sizes=(100, 250, 500, 1000, 2500, 5000, 10_000)):
    pnl = sim.pnl(day)
    sizes = [n for n in sizes if len(pnl) // n >= 10]
    out = {"day": day, "sizes": sizes, "paths": len(pnl)}
    for key, alpha, k in (("var99", 0.01, 0), ("es975", 0.025, 1)):
        spread = np.array([risk.sampling_spread(pnl, alpha, n)[k] for n in sizes])
        out[key] = {"p5": spread[:, 0], "p50": spread[:, 1], "p95": spread[:, 2],
                    "all": risk.var_es(pnl, alpha)[k]}
    return out
