# Monte-Carlo-simulations

Monte Carlo simulation for estimating portfolio risk (Value at Risk / Conditional Value at Risk) on a basket of ASX-listed stocks.

## What it does

1. Downloads 300 days of historical closing prices for six stocks (`CBA`, `BHP`, `TLS`, `WBC`, `NAB`, `STO`) via `yfinance`.
2. Computes mean daily returns and the covariance matrix of returns.
3. Generates random portfolio weights.
4. Runs 100 Monte Carlo simulations of a $10,000 portfolio over a 100-day horizon, using Cholesky decomposition to preserve correlation between assets.
5. Plots all simulated portfolio value paths.
6. Computes **VaR** (Value at Risk) and **CVaR** (Conditional VaR / Expected Shortfall) at a 95% confidence level (alpha = 5%) on the simulated ending portfolio values:
   - **VaR** — the portfolio value below which the worst 5% of simulated outcomes fall.
   - **CVaR** — the average portfolio value across those worst 5% of outcomes (always more conservative than VaR).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Prints `VaR` and `CVaR` in dollars, and opens a plot of the 100 simulated portfolio paths.

## Project structure

- `main.py` — main script (data download + Monte Carlo simulation + VaR/CVaR)
