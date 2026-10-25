"""Monte Carlo VaR and Expected Shortfall for a portfolio of ASX stocks.

Downloads (and caches) dividend-adjusted prices, simulates the portfolio under several
return models, backtests the models on 20+ years of history, and writes a report and
charts to results/.

    python main.py                                     # defaults below
    python main.py --weights CBA=0.4,BHP=0.2,TLS=0.1,WBC=0.1,NAB=0.1,STO=0.1
    python main.py --end today --refresh               # latest data
"""
import argparse
import datetime as dt
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_risk import backtest, data, plots, risk
from portfolio_risk.models import FilteredHistoricalModel, GJRGarch, NormalModel, StudentTModel
from portfolio_risk.simulation import simulate

TICKERS = ["CBA", "BHP", "TLS", "WBC", "NAB", "STO"]
START, END = "2000-01-01", "2026-10-24"   # fixed dates -> reproducible results (end is exclusive)
VALUE = 10_000                            # portfolio value in AUD
HORIZON = 100                             # trading days
PATHS = 100_000
SEED = 42
WINDOW = 500                              # days used to fit the normal and Student-t models
FHS_WINDOW = 1_000                        # days of shocks the filtered HS model draws from
GARCH_WINDOW = 2_520                      # days used to fit the GARCH volatility model (~10 years)
ORIGINAL_WINDOW = 205                     # ~300 calendar days, as in the first version
STRESS_PERIODS = {
    "GFC, Lehman collapse to market low": ("2008-09-12", "2009-03-06"),
    "COVID crash": ("2020-02-20", "2020-03-23"),
}
STOP_LOSS = 0.10                          # "down 10%" level for the path-risk section
PUT_STRIKE = 0.95                         # protective put struck at 95% of today's value
MEASURES = [("VaR 95%", 0.05, 0), ("VaR 99%", 0.01, 0), ("ES 97.5%", 0.025, 1)]
RESULTS = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="ASX codes")
    parser.add_argument("--weights", help="e.g. CBA=0.4,BHP=0.2,... (default: equal weights)")
    parser.add_argument("--value", type=float, default=VALUE, help="portfolio value in AUD")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="trading days to simulate")
    parser.add_argument("--paths", type=int, default=PATHS, help="Monte Carlo paths per model")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END, help="exclusive end date, or 'today'")
    parser.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    parser.add_argument("--skip-backtest", action="store_true", help="faster run without the backtest")
    parser.add_argument("--show", action="store_true", help="also open the charts in windows")
    args = parser.parse_args()
    if args.paths < 1000 or args.horizon < 1:
        parser.error("--paths must be at least 1000 and --horizon at least 1")
    if args.end == "today":
        args.end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    return args


def parse_weights(text, tickers):
    """'CBA=0.4,BHP=0.2,...' -> Series indexed by ticker. Default: equal weights."""
    if text is None:
        return pd.Series(1 / len(tickers), index=tickers)
    weights = pd.Series({k.strip(): float(v) for k, v in (item.split("=") for item in text.split(","))})
    if set(weights.index) != set(tickers):
        raise SystemExit(f"--weights needs exactly one weight for each of {tickers}")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise SystemExit("--weights must be non-negative and add up to 1")
    return weights[tickers]


def aud(x):
    return f"A${x:,.0f}"


def plus_minus(se):
    return f"±{se:,.0f}" if se >= 10 else f"±{se:.1f}"


def heading(title):
    return ["", title, "-" * len(title)]


def wrap(text):
    return textwrap.wrap(text, width=100)


def risk_table(sims, sigma_1d, hist_pnl, value, days):
    """VaR and ES for each horizon and method, as a printable DataFrame."""
    rows = {}
    for d in days:
        horizon = f"{d} day{'s' if d > 1 else ''}"
        rows[(horizon, "Closed form (normal)")] = [
            aud(value * risk.lognormal_var_es(sigma_1d * np.sqrt(d), a)[k]) for _, a, k in MEASURES]
        if d == 1:
            rows[(horizon, f"Historical, last {WINDOW} days")] = [
                aud(risk.var_es(hist_pnl, a)[k]) for _, a, k in MEASURES]
        for sim in sims:
            pnl = sim.pnl(d)
            rows[(horizon, f"MC {sim.model}")] = [
                f"{aud(risk.var_es(pnl, a)[k])} {plus_minus(risk.mc_standard_error(pnl, a)[k])}"
                for _, a, k in MEASURES]
    return pd.DataFrame(rows, index=[name for name, _, _ in MEASURES]).T


def backtest_table(summary):
    def zone(n):
        return f"{int(n)} {backtest.traffic_light(n)}"

    return pd.DataFrame({
        "breaches 95%": summary["95% breaches"].map("{:.2%}".format),
        "p 95%": summary["95% Kupiec p"].map("{:.3f}".format),
        "breaches 99%": summary["99% breaches"].map("{:.2%}".format),
        "p 99%": summary["99% Kupiec p"].map("{:.3f}".format),
        "clustering p": summary["99% clustering p"].map("{:.3f}".format),
        "worst year": summary["worst 250d"].map(zone),
        "last year": summary["last 250d"].map(zone),
        "ES ratio": summary["ES 97.5% ratio"].map("{:.2f}".format),
    })


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")
    tickers, value, horizon = args.tickers, args.value, args.horizon
    weights = parse_weights(args.weights, tickers)
    w = weights.to_numpy()
    days = sorted({d for d in (1, 10, horizon) if d <= horizon})

    prices = data.load_prices(tickers, args.start, args.end, refresh=args.refresh)
    log_r, simple_r = data.log_returns(prices), data.simple_returns(prices)
    needed = FHS_WINDOW + 60 + (0 if args.skip_backtest else 250)
    if len(log_r) < needed:
        raise SystemExit(f"Only {len(log_r)} trading days of data; need {needed}. Use an earlier --start.")
    recent = log_r.iloc[-WINDOW:]
    sigma_1d = (recent.to_numpy() @ w).std(ddof=1)

    models = [
        NormalModel(log_r.iloc[-ORIGINAL_WINDOW:], drift=True, name="Original (normal + drift)"),
        NormalModel(recent),
        StudentTModel(recent),
        FilteredHistoricalModel(log_r, window=FHS_WINDOW),
        FilteredHistoricalModel(log_r, window=FHS_WINDOW, vol=GJRGarch(), fit_window=GARCH_WINDOW),
    ]
    # every model gets the same seed, so each run of the script gives the same numbers
    sims = [simulate(m, weights, value, horizon, args.paths, np.random.default_rng(args.seed), days)
            for m in models]
    original, normal, student_t, fhs_ewma, fhs = sims    # fhs (GJR-GARCH) is the main model
    end_pnl = fhs.pnl(horizon)

    out = [f"Portfolio risk report: {', '.join(tickers)} (ASX)",
           f"Data: {len(log_r):,} trading days, {log_r.index[0]:%Y-%m-%d} to "
           f"{log_r.index[-1]:%Y-%m-%d}, dividend-adjusted closes",
           f"Portfolio: {aud(value)} bought today and held (no rebalancing): "
           + ", ".join(f"{t} {x:.1%}" for t, x in weights.items()),
           f"Models fitted on the last {WINDOW} days; filtered HS draws from {FHS_WINDOW:,} days of "
           f"shocks with EWMA or GJR-GARCH volatility (GARCH fitted on {GARCH_WINDOW:,} days).",
           f"Expected returns set to 0 (no drift). Main model: {fhs.model}, the best in the "
           "backtest (section 6).",
           f"Monte Carlo: {args.paths:,} paths per model, seed {args.seed}.",
           f"Daily volatility: {fhs.pnl(1).std() / value:.2%} now (GJR-GARCH) vs {sigma_1d:.2%} "
           f"on average over the last {WINDOW} days."]
    out += heading("Headline")
    out += wrap(f"Over the next {horizon} trading days there is a 1-in-100 chance of losing more than "
                f"{aud(risk.var_es(end_pnl, 0.01)[0])} (VaR 99%, {fhs.model}); in the worst 2.5% "
                f"of outcomes the average loss is {aud(risk.var_es(end_pnl, 0.025)[1])} (ES 97.5%). "
                f"The original script's assumptions give a VaR 99% of "
                f"{aud(risk.var_es(original.pnl(horizon), 0.01)[0])}.")

    garch = models[-1].vol
    out += heading("1. Volatility by stock (annualised)")
    out += pd.DataFrame({
        "now (GJR-GARCH)": [f"{v:.1%}" for v in models[-1].sigma * np.sqrt(252)],
        "now (EWMA)": [f"{v:.1%}" for v in models[-2].sigma * np.sqrt(252)],
        "long run": [f"{v:.1%}" for v in np.sqrt(garch.long_run_variance * 252)],
        "half-life (days)": [f"{h:.0f}" for h in np.log(0.5) / np.log(garch.persistence)],
        "alpha": [f"{x:.3f}" for x in garch.alpha],
        "gamma": [f"{x:.3f}" for x in garch.gamma],
        "beta": [f"{x:.3f}" for x in garch.beta],
    }, index=tickers).to_string().splitlines()
    out += [""] + wrap("GJR-GARCH: tomorrow's variance = omega + (alpha, plus gamma if the stock fell) "
                       "x today's squared return + beta x today's variance. Half-life: days for a "
                       "volatility shock to fade by half.")

    out += heading("2. Value at Risk and Expected Shortfall (loss in AUD; ± is Monte Carlo standard error)")
    hist_pnl = value * (simple_r.iloc[-WINDOW:].to_numpy() @ w)
    out += risk_table(sims, sigma_1d, hist_pnl, value, days).to_string().splitlines()
    lo, _, hi = risk.sampling_spread(end_pnl, 0.01, 100)[0]
    se = risk.mc_standard_error(end_pnl, 0.01)[0]
    out += [""] + wrap(f"Precision: with the original 100 paths, 90% of runs would put the {horizon}-day "
                       f"VaR 99% ({fhs.model}) anywhere from {aud(lo)} to {aud(hi)}. With "
                       f"{args.paths:,} paths it is {aud(risk.var_es(end_pnl, 0.01)[0])} ± {se:,.0f}.")
    boot = value * risk.bootstrap_normal_var(recent, w, 0.01, np.random.default_rng(args.seed))
    out += wrap(f"Parameter risk: re-fitting the normal model on resampled copies of its {WINDOW} days "
                f"moves its 1-day VaR 99% between {aud(np.percentile(boot, 5))} and "
                f"{aud(np.percentile(boot, 95))} (90% range).")

    d = 10 if 10 in days else horizon
    comp = risk.component_es(fhs.holdings_pnl(d), 0.025)
    es_total = comp.sum()
    out += heading(f"3. Where the risk comes from ({d}-day ES 97.5%, {fhs.model}: {aud(es_total)})")
    out += pd.DataFrame({"weight": weights.map("{:.1%}".format),
                         "ES contribution": [aud(c) for c in comp],
                         "share of ES": [f"{c / es_total:.1%}" for c in comp]},
                        index=tickers).to_string().splitlines()
    min_var = pd.Series(risk.min_variance_weights(models[1].cov), index=tickers)
    es_min_var = risk.var_es(fhs.holdings_pnl(d, min_var.to_numpy()).sum(axis=1), 0.025)[1]
    out += [""] + wrap("Long-only minimum-variance weights: "
                       + ", ".join(f"{t} {x:.0%}" for t, x in min_var.items())
                       + f". Their {d}-day ES 97.5% is {aud(es_min_var)} vs {aud(es_total)} for the "
                       "current weights (fitted on the same data, so somewhat optimistic).")

    out += heading(f"4. Path risk over {horizon} days (what a single end-date VaR misses)")
    floor = value * (1 - STOP_LOSS)
    path_risk = {}
    for sim in (normal, student_t, fhs_ewma, fhs):
        drawdown = risk.max_drawdown(sim.values)
        path_risk[sim.model] = {
            "median max drawdown": f"{np.median(drawdown):.1%}",
            "1-in-20 max drawdown": f"{np.percentile(drawdown, 95):.1%}",
            f"P(down {STOP_LOSS:.0%} at any point)": f"{(sim.values.min(axis=1) <= floor).mean():.1%}",
            f"P(down {STOP_LOSS:.0%} on day {horizon})": f"{(sim.values[:, -1] <= floor).mean():.1%}",
        }
    out += pd.DataFrame(path_risk).to_string().splitlines()
    payoff = np.maximum(PUT_STRIKE * value - fhs.values[:, -1], 0)
    premium = payoff.mean()
    hedged = end_pnl + payoff - premium
    out += [""] + wrap(
        f"Protective put (strike {PUT_STRIKE:.0%}, expires day {horizon}, fair premium under the "
        f"model {aud(premium)}): VaR 99% {aud(risk.var_es(end_pnl, 0.01)[0])} -> "
        f"{aud(risk.var_es(hedged, 0.01)[0])}, ES 97.5% {aud(risk.var_es(end_pnl, 0.025)[1])} -> "
        f"{aud(risk.var_es(hedged, 0.025)[1])}. The put's kinked payoff is easy to simulate but "
        "breaks closed-form (normal) VaR.")

    out += heading("5. Historical stress tests (today's weights, bought and held)")
    for name, (start, end) in STRESS_PERIODS.items():
        if log_r.index[0] < pd.Timestamp(start):
            loss = risk.stress_loss(log_r, weights, start, end)
            out.append(f"{name} ({start} to {end}): {aud(loss * value)} loss ({loss:.1%})")
    for n in sorted({10, horizon}):
        loss, last_day = risk.worst_loss(log_r, weights, n)
        out.append(f"Worst {n}-day loss since {log_r.index[0]:%Y}: {aud(loss * value)} ({loss:.1%}), "
                   f"ending {last_day:%Y-%m-%d}. Model {n}-day ES 97.5% today: "
                   f"{aud(risk.var_es(fhs.pnl(n), 0.025)[1])}")

    bt = None
    if not args.skip_backtest:
        bt = backtest.run_backtest(simple_r, weights, WINDOW, FHS_WINDOW, ORIGINAL_WINDOW, GARCH_WINDOW)
        out += heading(f"6. Backtest of 1-day VaR, re-fitted every day: {bt.dates[0]:%Y-%m-%d} to "
                       f"{bt.dates[-1]:%Y-%m-%d} ({len(bt.losses):,} days)")
        out += backtest_table(backtest.summary(bt)).to_string().splitlines()
        out += [""] + wrap("Targets: breaches 5% and 1% of days. p = Kupiec test of the breach rate; "
                           "clustering p = Christoffersen test (do breaches bunch up?); p below 0.05 "
                           "rejects the model. Worst/last year = 99% breaches in 250 days, Basel zones "
                           "0-4 green, 5-9 yellow, 10+ red. ES ratio = actual / forecast loss on "
                           "breach days, ideally 1.")

    RESULTS.mkdir(exist_ok=True)
    figures = {
        "fan_chart": plots.fan_chart(fhs, normal, value),
        "pnl_distribution": plots.pnl_distribution(fhs, normal, horizon),
        "convergence": plots.convergence(fhs, horizon),
        "risk_contributions": plots.risk_contributions(tickers, w, comp / es_total, d),
    }
    if bt is not None:
        figures["backtest"] = plots.backtest_chart(bt, [list(bt.forecasts)[0], list(bt.forecasts)[-1]])
    else:
        (RESULTS / "backtest.png").unlink(missing_ok=True)    # don't leave a chart from an older run
    for name, fig in figures.items():
        plots.save(fig, RESULTS / f"{name}.png")
    out += ["", f"Charts and this report saved to {RESULTS}/"]

    report = "\n".join(out)
    print(report)
    (RESULTS / "report.txt").write_text(report + "\n")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
