import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocklist = ['CBA', 'BHP', 'TLS', 'WBC', 'NAB', 'STO']
stocks = [stock + '.AX' for stock in stocklist]
endData = dt.datetime.now()
startData = endData - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startData, endData)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Method 
# number of simulations
mc_sims = 100 
T = 100 # timeframes in days 

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    # MC loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation: Portfolio Performance')
plt.show()

def mcVaR(returns, alpha=5):
    """ Input: pandas Series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected returns to be a pandas Series")


def mcCVaR(returns, alpha=5):
    """ Input: pandas Series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected returns to be a pandas Series")


portfolioReturns = pd.Series(portfolio_sims[-1, :])

VaR = mcVaR(portfolioReturns, alpha=5)
CVaR = mcCVaR(portfolioReturns, alpha=5)

print('VaR ${}'.format(round(VaR, 2)))
print('CVaR ${}'.format(round(CVaR, 2)))