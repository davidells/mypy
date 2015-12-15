import numpy as np
import pandas as pd

import portfolio
import returns

def vs_sharpe_table(test_val_name, test_vals, strategy_fn):
    results = []
    for test_val in test_vals:
        # Get portfolio returns of this
        # strategy (defined by strategy_fn)
        rets = apply(
            portfolio.portfolio_returns, 
            strategy_fn(test_val))

        # Add the sharpe ratio, and then our test val
        res = { 'sharpe': returns.annual_sharpe(rets['returns']) }
        res[test_val_name] = test_val
        results.append(res)
        
    return pd.DataFrame(results)

def vs_sharpe(test_val_name, test_vals, strategy_fn):
    sharpe_table = vs_sharpe_table(test_val_name, test_vals, strategy_fn)
    return sharpe_table.fillna(0).sort('sharpe')[test_val_name].iloc[-1]

def plot_vs_sharpe(test_val_name, test_vals, strategy_fn):
    df = vs_sharpe_table(test_val_name, test_vals, strategy_fn)
    df.plot(x=test_val_name, y='sharpe', figsize=(16,3))