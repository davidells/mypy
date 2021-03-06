import numpy as np
import pandas as pd

def portfolio_price(prices, weights):
    return (prices * weights).sum(axis=1)

def portfolio_returns(prices, weights, units):
    positions = prices * weights * units
    pnl = (positions.shift(1) * prices.pct_change()).sum(axis=1)
    grossMktVal = positions.abs().sum(axis=1)
    ret = (pnl / grossMktVal.shift(1)).fillna(value=0)
    return {
        'returns': ret,
        'positions': positions,
        'pnl': pnl,
        'grossMktVal': grossMktVal
    }

def portfolio_shares(prices, weights, units, marketValue):
    returns = portfolio_returns(prices, weights, units)
    grossMktVal = returns['grossMktVal']
    gmvFrame = pd.DataFrame([grossMktVal] * prices.shape[1]).transpose()
    gmvFrame.columns = prices.columns
    gmvFrame.index = prices.index
    shares = units * weights * (marketValue / gmvFrame)
    return shares.replace([np.inf, -np.inf], np.nan).fillna(value=0)