# Monte Carlo portfolio risk: VaR and Expected Shortfall for ASX stocks

How much could an A$10,000 equal-weight portfolio of six ASX stocks (CBA, BHP, TLS, WBC, NAB,
STO) lose over the next 1, 10 or 100 trading days? This project answers with Monte Carlo
simulation under four return models, then tests the models against 20+ years of real losses.

All numbers below come from `python main.py` on data up to 23 Oct 2026 (seed 42). The full
output is in [results/report.txt](results/report.txt).

## Key results

**100-day risk, A$ loss** (Monte Carlo standard error A$4–21)

| Model | VaR 95% | VaR 99% | ES 97.5% |
|---|---:|---:|---:|
| First version's assumptions (normal, drift from 205 days) | 623 | 1,133 | 1,138 |
| Normal (500-day covariance, no drift) | 1,323 | 1,795 | 1,807 |
| Student-t (ν = 4.8, fitted) | 1,325 | 1,804 | 1,815 |
| Filtered historical simulation + EWMA volatility | 1,966 | 2,958 | 2,988 |
| **Filtered historical simulation + GJR-GARCH** (main model) | **1,494** | **2,221** | **2,246** |

- **The first version understated the risk by about half** (A$1,133 vs A$2,221). Most of the gap
  is its drift: +14.7% a year, estimated from 205 days and statistically indistinguishable
  from zero. Removing it alone moves the 100-day VaR 99% from A$1,133 to A$1,777.
- **100 simulations are far too few.** With 100 paths, the 100-day VaR 99% lands anywhere from
  A$1,536 to A$2,626 (90% of runs). With 100,000 paths it is A$2,221 ± 14.
- **Only the filtered historical simulation models pass the backtest** (below). The GJR-GARCH
  version is the main model: it is the most accurate at the 95% level and has the fewest
  clustered breaches. Its ES matches realised tail losses on average (ratio 1.01).
- **How volatility evolves matters at long horizons.** EWMA lets volatility wander with no
  anchor, which inflates the 100-day VaR to A$2,958. GARCH pulls volatility back towards each
  stock's long-run level within days or weeks.

![Fan chart of simulated portfolio value](results/fan_chart.png)

![Distribution of 100-day profit and loss](results/pnl_distribution.png)

### Backtest: 1-day VaR, re-fitted daily on past data only, Feb 2004 – Oct 2026 (5,735 days)

| Model | 95% VaR breaches (target 5%) | 99% VaR breaches (target 1%) | Kupiec p (99%) | Worst 250 days (Basel zone) | ES ratio (target 1) |
|---|---:|---:|---:|---|---:|
| First version (normal + drift, 205 days) | 5.68% | 2.14% | 0.000 | 18 (red) | 1.17 |
| Normal, 500 days | 5.27% | 1.99% | 0.000 | 18 (red) | 1.23 |
| Student-t, 500 days | 5.70% | 1.57% | 0.000 | 15 (red) | 1.10 |
| Historical simulation, 500 days | 5.61% | 1.45% | 0.001 | 13 (red) | 1.07 |
| Filtered HS + EWMA | 4.50% | 0.99% | 0.963 | 9 (yellow) | 1.00 |
| **Filtered HS + GJR-GARCH** | **4.71%** | **1.01%** | **0.931** | **9 (yellow)** | **1.01** |

A p-value below 0.05 rejects the model. Two details are worth noting:

- At 95% the normal model looks fine (5.27% breaches, p = 0.36), which is why its tail problem
  is easy to miss.
- Breaches on consecutive days fall from 10 for the first version to 6 with EWMA and 4 with
  GJR-GARCH; a well-calibrated model would have about 0.6 by chance. The four that remain are all two-day crash
  starts: January 2008, the late-2014 oil collapse, the August 2024 yen carry-trade unwind and
  the April 2025 tariff shock.

![VaR backtest](results/backtest.png)

### Other findings

- **Volatility by stock:** GJR-GARCH finds a leverage effect for five of the six stocks: a fall
  raises volatility more than a rise of the same size. Telstra is the exception. Over the last
  10 years the banks' volatility shocks faded fast, with half-lives of 5–11 days, versus
  17–22 days for BHP and Santos (see Limitations: these estimates shift between periods).
- **Where the risk comes from:** the three banks hold 50% of the money but cause 70% of the
  10-day Expected Shortfall; Telstra holds 17% and causes 4%. Long-only minimum-variance
  weights (58% Telstra) would cut 10-day ES from A$872 to A$657.
- **Path risk:** there is a 23.5% chance of being down 10% at some point within 100 days, but
  only 13.0% of being down 10% on day 100. A single end-date VaR hides this.
- **Options:** a 100-day put struck at 95% (model fair premium A$164) cuts the 100-day VaR 99%
  from A$2,221 to A$664. Simulation handles the put's kinked payoff; a closed-form normal VaR
  cannot.
- **Stress tests (today's weights):** Lehman collapse to the March 2009 low: −26.0%. COVID
  crash, 20 Feb – 23 Mar 2020: −40.0%. That is far beyond the model's 100-day ES (22%), a
  reminder that ES is an average of bad outcomes, not the worst case.

![Risk contributions](results/risk_contributions.png)

![Monte Carlo convergence](results/convergence.png)

## Method

**Data.** Daily dividend-adjusted closes from Yahoo Finance, January 2000 to October 2026. Days
on which any stock is missing are dropped, as are holiday rows where Yahoo repeats every
price. Data before 2000 is left out because Yahoo's 1997–98 ASX prices are mostly stale.
Prices are cached in `data/` and the end date is fixed, so results are reproducible.

**Portfolio.** Bought today and held, with no rebalancing. Weights are matched to stocks by
ticker, never by position.

**Models.** All of them set the expected return to zero: a two-year sample cannot estimate it,
and a positive drift shrinks the risk numbers. Each log return has mean −σ²/2, so the expected
*simple* return is exactly zero.

- *Normal:* multivariate normal log returns, sample covariance of the last 500 days.
- *Student-t:* multivariate t with the same covariance and degrees of freedom fitted by maximum
  likelihood. A shared shock each day makes the stocks crash together (tail dependence).
- *Filtered historical simulation* (Barone-Adesi et al., 1999): each stock's returns are divided
  by its volatility to get standardised shocks. Paths are built from randomly drawn historical
  *days* of these shocks (the last 1,000 days), which keeps the real joint moves and fat tails,
  rescaled to the current volatility. Volatility then updates along each path. Two volatility
  models are compared:
  - *EWMA* (RiskMetrics, λ = 0.94): nothing to estimate, no long-run level.
  - *GJR-GARCH(1,1)*: variance = ω + (α + γ·[fell yesterday])·r² + β·variance. Fitted per stock
    on the last 10 years by Gaussian quasi-maximum likelihood with variance targeting. The
    variance recursion runs as a linear filter (`scipy.signal.lfilter`), so a fit takes about
    2 ms per stock. Estimates are stable across optimizer starting points and close to a
    free-ω fit.

The main model was chosen by the backtest. The GJR-GARCH version is as accurate as EWMA at 99%,
more accurate at 95%, and has the fewest clustered breaches. Choosing a model by its backtest
slightly flatters it, so the EWMA version is reported alongside.

**Risk measures.** VaR is the loss exceeded with the given probability. Expected Shortfall
(ES) is the average loss beyond it; ES 97.5% is the Basel FRTB standard. Monte Carlo standard
errors come from 20 independent batches. A closed-form lognormal formula checks the normal
simulation (A$196 vs A$195 for the 1-day VaR 99%). Bootstrapping the estimation window
shows parameter risk: the 1-day normal VaR 99% ranges from A$178 to A$214 (90% range).

**Risk decomposition.** Euler allocation: each stock's contribution is its average loss in the
scenarios beyond VaR, and the contributions add up exactly to the portfolio ES.

**Backtest.** Every day from 2004, each model is re-fitted using only earlier data and its
1-day VaR and ES are compared with the realised loss. The GARCH and Student-t parameters are
re-estimated every 20 days; volatilities update daily. The tests are Kupiec (is the breach
rate right?), Christoffersen (do breaches cluster?), the Basel traffic light (99% breaches per
250 days), and the ratio of realised to forecast loss on breach days.

## Limitations

- **Some clustering remains.** Even the GJR-GARCH model fails the Christoffersen test
  (p = 0.003), because no model updated once a day can catch the second day of a sudden crash.
  Intraday data or implied volatility (for example, the ASX VIX) would be the next step.
- **Only the 1-day forecasts are backtested.** Since 2000 there are only about 68
  non-overlapping 100-day periods, too few to test a 1% tail.
- **GARCH parameters are unstable across periods.** Fitted on 2010–2019, the banks' volatility
  half-lives are 14–24 days. Fitted on 2021–2026, they are under 2 days, because shocks faded
  almost immediately. The 10-year window used mixes both regimes, and it also sets the
  long-run volatility each stock reverts to.
- No transaction costs, liquidity effects or currency risk; dividends are reinvested through
  adjusted prices.
- The minimum-variance weights are fitted on the same data used to evaluate them.
- The put premium is the model's own fair value at a zero interest rate. Market prices
  usually imply higher volatility.

## Usage

```bash
pip install -r requirements.txt
python main.py                                    # report + charts in results/ (~15-20 s)
python main.py --weights CBA=0.4,BHP=0.2,TLS=0.1,WBC=0.1,NAB=0.1,STO=0.1
python main.py --horizon 10 --paths 50000 --skip-backtest
python main.py --end today --refresh              # latest data instead of the cached snapshot
python main.py --help                             # all options
python -m unittest                                # 20 tests
```

Tested with Python 3.14 and the package versions pinned in `requirements.txt`.

## Project structure

```
main.py                  command-line entry point; builds the report
portfolio_risk/
    data.py              download, cache and clean prices
    models.py            normal, Student-t and filtered HS models; EWMA and GJR-GARCH volatility
    simulation.py        Monte Carlo engine (buy-and-hold paths)
    risk.py              VaR, ES, standard errors, Euler allocation, stress tests
    backtest.py          rolling 1-day backtest and its statistical tests
    plots.py             charts
tests/                   unit tests
results/                 report and charts from the last run
```
