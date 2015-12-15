import numpy as np
import pandas as pd

def apr(returns, periods=252):
    return pow((1 + returns.mean()), periods) - 1

def annual_sharpe(returns, risk_free_rate=0, periods=252):
    r = returns - risk_free_rate
    return np.sqrt(periods) * (r.mean() / r.std())

def cumret(returns):
    return (1 + returns).cumprod() - 1

def total_return(returns):
    return cumret(returns).iloc[-1]

def drawdown(returns):
    equity = cumret(returns)
    emax = pd.expanding_max(equity)
    return equity - emax

def time_in_drawdown(returns):
    equity = cumret(returns)
    emax = pd.expanding_max(equity)
    block = (emax.shift(1) != emax).astype(int).cumsum()
    blocks_by_value = block.groupby(by=block.__getitem__)
    replace_block_with_range = lambda x: range(1, len(x) + 1)
    return blocks_by_value.transform(replace_block_with_range) - 1

def report(returns):
    return {
        'total_return': total_return(returns),
        'annual_sharpe': annual_sharpe(returns),
        'apr': apr(returns),
        'drawdown': drawdown(returns).min(),
        'time_in_drawdown': time_in_drawdown(returns).max()
    }

def report_str(returns, name):
    return """%s
        Total return: %.3f
        Sharpe ratio: %.2f
        APR: %.3f
        Drawdown: %.2f
        Max Time in Drawdown: %d
    """ % (
        name,
        totalret(returns),
        annual_sharpe(returns),
        apr(returns),
        drawdown(returns).min(),
        time_in_drawdown(returns).max()
    )