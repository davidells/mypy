import os
import pandas as pd
import numpy as np

import portfolio

from nose.tools import assert_almost_equals

def setup_module():
    global EWA, EWC, prices
    path = os.path.dirname(os.path.abspath(__file__))
    EWA = pd.read_csv(path + "/EWA.csv")['Adj Close']
    EWC = pd.read_csv(path + "/EWC.csv")['Adj Close']
    prices = pd.concat([EWA, EWC], axis=1, keys=['EWA', 'EWC'])
    
def portfolio_prices_test():
    port = portfolio.portfolio_prices(prices, [1, -0.4])
    assert_almost_equals(port.iloc[0], 5.6539, places=4)
    assert_almost_equals(port.iloc[1], 5.6344, places=4)
    assert_almost_equals(port.iloc[-2], 9.1112, places=4)
    assert_almost_equals(port.iloc[-1], 9.0446, places=4)
    
def portfolio_returns_test():
    (ret, positions, pnl, grossMktVal) = portfolio.portfolio_returns(prices, [1, -0.4], 1)
    assert_almost_equals(ret.iloc[-1], -0.0022, places=4)
    assert_almost_equals(positions.iloc[-1]['EWA'], 19.2250, places=4)
    assert_almost_equals(pnl.iloc[-1], -0.0666, places=4)
    assert_almost_equals(grossMktVal.iloc[-1], 29.4055, places=4)
    
def portfolio_shares_test():
    weights = pd.DataFrame({'EWA': np.ones(len(EWA)), 'EWC': [-0.4] * len(EWC)}, index=prices.index)
    shares = portfolio.portfolio_shares(prices, weights, 1, 500)
    assert_almost_equals(shares.iloc[-1]['EWA'], 17.0036, places=4)
    assert_almost_equals(shares.iloc[-1]['EWC'], -6.8015, places=4)
    portfolioMarketVal = (shares * prices).abs().sum(axis=1)
    assert_almost_equals(portfolioMarketVal.iloc[-1], 500, places=4)
    assert_almost_equals(portfolioMarketVal.sum(), len(prices) * 500, places=4)