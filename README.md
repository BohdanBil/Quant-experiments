# Monte Carlo portfolio risk: VaR and Expected Shortfall for ASX stocks

How much could an A$10,000 equal-weight portfolio of six ASX stocks (CBA, BHP, TLS, WBC, NAB,
STO) lose over the next 1, 10 or 100 trading days? This project answers with Monte Carlo
simulation under three return models, then tests the models against 20+ years of real losses.

All numbers below come from `python main.py` on data up to 23 Oct 2026 (seed 42). The full
output is in [results/report.txt](results/report.txt).

## Key results

**100-day risk, A$ loss** (Monte Carlo standard error A$4–22)

| Model | VaR 95% | VaR 99% | ES 97.5% |
|---|---:|---:|---:|
| First version's assumptions (normal, drift from 205 days) | 623 | 1,133 | 1,138 |
| Normal (500-day covariance, no drift) | 1,323 | 1,795 | 1,807 |
| Student-t (ν = 4.8, fitted) | 1,325 | 1,804 | 1,815 |
| **Filtered historical simulation** | **1,967** | **2,967** | **2,995** |

- **The first version understated the risk by more than half.** Its drift (+14.7% a year,
  estimated from 205 days and statistically indistinguishable from zero) alone cut the 100-day
  VaR 99% from A$1,777 to A$1,133.
  Volatility is also above its two-year average right now (1.07% vs 0.85% a day), which only
  the filtered model picks up.
- **100 simulations are far too few.** With 100 paths, the 100-day VaR 99% lands anywhere from
  A$1,976 to A$3,544 (90% of runs). With 100,000 paths it is A$2,967 ± 22.
- **Only filtered historical simulation passes the backtest** (below). It breaches its 99% VaR on
  0.99% of days (target 1%), and its ES matches realised tail losses on average (ratio 1.00).

![Fan chart of simulated portfolio value](results/fan_chart.png)

![Distribution of 100-day profit and loss](results/pnl_distribution.png)

### Backtest: 1-day VaR, re-fitted daily on past data only, Feb 2004 – Oct 2026 (5,735 days)

| Model | 99% VaR breaches (target 1%) | Kupiec p | Worst 250 days (Basel zone) | ES ratio (target 1) |
|---|---:|---:|---|---:|
| First version (normal + drift, 205 days) | 2.14% | 0.000 | 18 (red) | 1.17 |
| Normal, 500 days | 1.99% | 0.000 | 18 (red) | 1.23 |
| Student-t, 500 days | 1.57% | 0.000 | 15 (red) | 1.10 |
| Historical simulation, 500 days | 1.45% | 0.001 | 13 (red) | 1.07 |
| **Filtered historical simulation** | **0.99%** | **0.963** | **9 (yellow)** | **1.00** |

A p-value below 0.05 rejects the model. At 95% the normal model passes (5.27% breaches,
p = 0.36), which is why its tail problem is easy to miss.

![VaR backtest](results/backtest.png)

### Other findings

- **Where the risk comes from:** the three banks hold 50% of the money but cause 66% of the
  10-day Expected Shortfall; Telstra holds 17% and causes 6%. Long-only minimum-variance
  weights (58% Telstra) would cut 10-day ES from A$886 to A$758.
- **Path risk:** there is a 34% chance of being down 10% at some point within 100 days, but
  only 19% of being down 10% on day 100. A single end-date VaR hides this.
- **Options:** a 100-day put struck at 95% (model fair premium A$263) cuts the 100-day VaR 99%
  from A$2,967 to A$763. Simulation handles the put's kinked payoff; a closed-form normal VaR
  cannot.
- **Stress tests (today's weights):** Lehman collapse to the March 2009 low: −26.0%. COVID
  crash, 20 Feb – 23 Mar 2020: −40.0%. That is beyond the model's 100-day ES (30%), a reminder
  that ES is an average of bad outcomes, not the worst case.

![Risk contributions](results/risk_contributions.png)

![Monte Carlo convergence](results/convergence.png)

## Method

**Data.** Daily dividend-adjusted closes from Yahoo Finance, January 2000 to October 2026. Days
on which any stock is missing are dropped, as are holiday rows where Yahoo repeats every
price. Data before 2000 is left out because Yahoo's 1997–98 ASX prices are mostly stale.
Prices are cached in `data/` and the end date is fixed, so results are reproducible.

**Portfolio.** Bought today and held, with no rebalancing. Weights are matched to stocks by
ticker, never by position.

**Models.** All three set the expected return to zero: a two-year sample cannot estimate it,
and a positive drift shrinks the risk numbers. Each log return has mean −σ²/2, so the
expected *simple* return is exactly zero.

- *Normal:* multivariate normal log returns, sample covariance of the last 500 days.
- *Student-t:* multivariate t with the same covariance and degrees of freedom fitted by maximum
  likelihood. A shared shock each day makes the stocks crash together (tail dependence).
- *Filtered historical simulation* (Barone-Adesi et al., 1999): each stock's returns are divided
  by its EWMA volatility (λ = 0.94). Paths are built from randomly drawn historical *days* of
  these shocks (the last 1,000 days), which keeps the real joint moves and fat tails, rescaled
  to the current volatility. Volatility updates along each path, so turbulence persists.

**Risk measures.** VaR is the loss exceeded with the given probability. Expected Shortfall
(ES) is the average loss beyond it; ES 97.5% is the Basel FRTB standard. Monte Carlo standard
errors come from 20 independent batches. A closed-form lognormal formula checks the normal
simulation (A$196 vs A$195 for the 1-day VaR 99%). Bootstrapping the estimation window
shows parameter risk: the 1-day normal VaR 99% ranges from A$178 to A$214 (90% range).

**Risk decomposition.** Euler allocation: each stock's contribution is its average loss in the
scenarios beyond VaR, and the contributions add up exactly to the portfolio ES.

**Backtest.** Every day from 2004, each model is re-fitted using only earlier data and its
1-day VaR and ES are compared with the realised loss. The tests are Kupiec (is the breach rate
right?), Christoffersen (do breaches cluster?), the Basel traffic light (99% breaches per 250
days), and the ratio of realised to forecast loss on breach days.

## Limitations

- **Breaches still cluster for every model**, filtered HS included (Christoffersen p < 0.001).
  EWMA volatility reacts to a crash one day late. A GARCH model with an asymmetric response to
  losses is the natural next step.
- **EWMA volatility has no mean reversion.** Over 100 days, a few simulated paths reach extreme
  volatility. GARCH would fix this too.
- **Only the 1-day forecasts are backtested.** Since 2000 there are only about 68
  non-overlapping 100-day periods, too few to test a 1% tail.
- No transaction costs, liquidity effects or currency risk; dividends are reinvested through
  adjusted prices.
- The minimum-variance weights are fitted on the same data used to evaluate them.
- The put premium is the model's own fair value at a zero interest rate. Market prices
  usually imply higher volatility.

## Usage

```bash
pip install -r requirements.txt
python main.py                                    # report + charts in results/ (~10 s)
python main.py --weights CBA=0.4,BHP=0.2,TLS=0.1,WBC=0.1,NAB=0.1,STO=0.1
python main.py --horizon 10 --paths 50000 --skip-backtest
python main.py --end today --refresh              # latest data instead of the cached snapshot
python main.py --help                             # all options
python -m unittest                                # 17 tests
```

Tested with Python 3.14 and the package versions pinned in `requirements.txt`.

## Project structure

```
main.py                  command-line entry point; builds the report
portfolio_risk/
    data.py              download, cache and clean prices
    models.py            normal, Student-t and filtered historical simulation models
    simulation.py        Monte Carlo engine (buy-and-hold paths)
    risk.py              VaR, ES, standard errors, Euler allocation, stress tests
    backtest.py          rolling 1-day backtest and its statistical tests
    plots.py             charts
tests/                   unit tests
results/                 report and charts from the last run
```
