import numpy as np
import pandas as pd

def apr(returns, periods=252):
    return pow((1 + returns.mean()), periods) - 1

def annual_sharpe(returns, risk_free_rate=0, periods=252):
    r = returns - risk_free_rate
    return np.sqrt(periods) * (r.mean() / r.std())

def cumret(returns):
    return (1 + returns).cumprod() - 1

def drawdown(returns):
    equity = cumret(returns)
    emax = pd.expanding_max(equity)
    return (equity - emax) * -1

def time_in_drawdown(returns):
    equity = cumret(returns)
    emax = pd.expanding_max(equity)
    block = (emax.shift(1) != emax).astype(int).cumsum()
    block_by_values = block.groupby(by=block.__getitem__)
    return block_by_values.transform(lambda x: range(1, len(x) + 1)) - 1