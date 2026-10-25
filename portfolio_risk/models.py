"""Return models for the Monte Carlo engine.

Each model has a `tickers` list, a `name`, and `sample(rng, n_paths, horizon)`, which returns
daily log returns with shape (n_paths, horizon, n_stocks).

Expected returns are set to zero unless asked otherwise: a couple of years of data cannot
pin down a stock's mean return, and an optimistic drift quietly shrinks the risk numbers.
"Zero" means zero expected *simple* return, so each log return gets a mean of -variance/2
(otherwise compounding alone would add a small positive drift).
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter
from scipy.special import gammaln


class NormalModel:
    """Multivariate normal log returns with the sample covariance (the textbook model).

    drift=True uses the sample mean as expected return, as the original main.py did.
    """

    def __init__(self, log_returns, drift=False, name="Normal"):
        self.name = name
        self.tickers = list(log_returns.columns)
        self.cov = log_returns.cov().to_numpy()
        self.mean = log_returns.mean().to_numpy() if drift else -0.5 * np.diag(self.cov)
        self._chol = np.linalg.cholesky(self.cov)

    def sample(self, rng, n_paths, horizon):
        z = rng.standard_normal((n_paths, horizon, len(self.tickers)))
        return self.mean + z @ self._chol.T


class StudentTModel:
    """Multivariate Student-t: fat tails, plus one shared shock per day that makes the
    stocks crash together (tail dependence). Covariance equals the sample covariance;
    the degrees of freedom nu are fitted by maximum likelihood (lower = fatter tails)."""

    def __init__(self, log_returns):
        x = (log_returns - log_returns.mean()).to_numpy()
        self.tickers = list(log_returns.columns)
        self.cov = np.cov(x, rowvar=False)
        self.mean = -0.5 * np.diag(self.cov)
        self.nu = fit_t_dof(x, self.cov)
        self.name = f"Student-t (nu={self.nu:.1f})"
        self._chol = np.linalg.cholesky(self.cov)

    def sample(self, rng, n_paths, horizon):
        z = rng.standard_normal((n_paths, horizon, len(self.tickers))) @ self._chol.T
        chi2 = rng.chisquare(self.nu, size=(n_paths, horizon, 1))
        return self.mean + z * np.sqrt((self.nu - 2) / chi2)


class FilteredHistoricalModel:
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    1. Divide each stock's returns by its volatility (EWMA or GARCH) to get standardised shocks.
    2. Build paths from randomly drawn historical *days* of shocks, scaled by the current
       volatility. Drawing whole days keeps how the stocks really moved together.
    3. Update each volatility with the simulated return, so a bad day raises the risk of
       the next one (volatility clustering).

    `vol` is fitted on the last `fit_window` days; shocks come from the last `window` days.
    """

    def __init__(self, log_returns, window=1000, vol=None, fit_window=2520):
        r = log_returns.to_numpy()
        self.vol = (vol or EWMA()).fit(r[-fit_window:])
        self.name = f"Filtered HS + {self.vol.name}"
        var = self.vol.variance(r)
        shocks = (r / np.sqrt(var[:-1]))[-window:]
        self.tickers = list(log_returns.columns)
        self.shocks = shocks - shocks.mean(axis=0)
        self.sigma = np.sqrt(var[-1])                 # volatility forecast for tomorrow
        self._half_shock_var = 0.5 * self.shocks.var(axis=0)

    def sample(self, rng, n_paths, horizon):
        var = np.tile(self.sigma ** 2, (n_paths, 1))
        out = np.empty((n_paths, horizon, len(self.tickers)))
        for t in range(horizon):
            eps = self.shocks[rng.integers(len(self.shocks), size=n_paths)] * np.sqrt(var)
            out[:, t] = eps - self._half_shock_var * var     # mean -variance/2
            var = self.vol.update(var, eps)
        return out


class EWMA:
    """RiskMetrics volatility: an exponentially weighted average of squared returns.
    Nothing to fit, and no pull back towards a long-run level."""

    name = "EWMA"

    def __init__(self, lam=0.94):
        self.lam = lam

    def fit(self, returns):
        return self

    def variance(self, returns):
        return ewma_variance(returns, self.lam)

    def update(self, var, eps):
        return self.lam * var + (1 - self.lam) * eps ** 2


class GJRGarch:
    """GJR-GARCH(1,1) volatility for each stock:

        var[t] = omega + (alpha + gamma * [r[t-1] < 0]) * r[t-1]^2 + beta * var[t-1]

    Like EWMA, a big move raises tomorrow's volatility, but a fall raises it more than a rise
    of the same size (gamma, the leverage effect), and volatility drifts back to a long-run level.
    """

    name = "GJR-GARCH"

    def fit(self, returns):
        params = np.array([fit_gjr_garch(column) for column in returns.T])
        self.omega, self.alpha, self.gamma, self.beta = params.T
        return self

    @property
    def persistence(self):
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def long_run_variance(self):
        return self.omega / (1 - self.persistence)

    def variance(self, returns):
        return np.column_stack([
            gjr_variance(returns[:, i], self.omega[i], self.alpha[i], self.gamma[i],
                         self.beta[i], self.long_run_variance[i])
            for i in range(returns.shape[1])])

    def update(self, var, eps):
        return self.omega + (self.alpha + self.gamma * (eps < 0)) * eps ** 2 + self.beta * var


def ewma_variance(returns, lam=0.94, burn_in=30):
    """RiskMetrics EWMA variance of each column.

    Row t is the forecast for day t using data up to day t-1; the extra last row is the
    forecast for the day after the sample.
    """
    var = np.empty((len(returns) + 1, returns.shape[1]))
    var[0] = returns[:burn_in].var(axis=0)
    for t, r in enumerate(returns):
        var[t + 1] = lam * var[t] + (1 - lam) * r ** 2
    return var


def gjr_variance(r, omega, alpha, gamma, beta, var0):
    """GJR-GARCH variance of one series, same row convention as `ewma_variance`.

    The recursion is a linear filter in the variance, so scipy runs it in compiled code.
    """
    x = omega + (alpha + gamma * (r < 0)) * r ** 2
    var = np.empty(len(r) + 1)
    var[0] = var0
    var[1:] = lfilter([1.0], [1.0, -beta], x, zi=[beta * var0])[0]
    return var


def fit_gjr_garch(r):
    """Quasi-maximum-likelihood GJR-GARCH(1,1) for one return series.

    Variance targeting fixes the long-run variance at the sample variance, leaving alpha,
    gamma and beta to estimate. Returns (omega, alpha, gamma, beta).
    """
    r = r - r.mean()
    v = r.var()

    def neg_log_lik(p):
        alpha, gamma, beta = p
        var = gjr_variance(r, v * (1 - alpha - gamma / 2 - beta), alpha, gamma, beta, v)[:-1]
        if (var <= 0).any():
            return 1e10
        return 0.5 * np.mean(np.log(var) + r ** 2 / var)

    result = minimize(neg_log_lik, x0=[0.03, 0.08, 0.9], method="SLSQP",
                      bounds=[(0, 0.5), (0, 0.5), (0, 0.999)], options={"ftol": 1e-10},
                      constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1] / 2 - p[2]})
    alpha, gamma, beta = result.x
    return v * (1 - alpha - gamma / 2 - beta), alpha, gamma, beta


def fit_t_dof(x, cov, bounds=(2.05, 100.0)):
    """Maximum-likelihood degrees of freedom of a zero-mean multivariate t with covariance `cov`."""
    n, d = x.shape
    maha = np.einsum("ij,jk,ik->i", x, np.linalg.inv(cov), x)
    logdet = np.linalg.slogdet(cov)[1]

    def neg_log_lik(nu):
        # scale matrix = cov * (nu - 2) / nu, so that the covariance stays equal to `cov`
        const = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log((nu - 2) * np.pi) - logdet / 2
        return -(n * const - (nu + d) / 2 * np.log1p(maha / (nu - 2)).sum())

    return minimize_scalar(neg_log_lik, bounds=bounds, method="bounded").x
