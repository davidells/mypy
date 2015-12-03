import numpy as np
import pandas as pd

def empty_series(proto):
    s = pd.Series(np.zeros(len(proto)), index=proto.index)
    s[:] = np.nan
    return s

def ma(price_series, n):
    return pd.rolling_mean(price_series, n)

def n_period_return(price_series, n):
    ''' Using the given price series, returns a new series 
        corresponding to the cumulative net returns of n periods ago
        for each period in the series'''
    gross_ret = 1 + price_series.pct_change()
    gross_np_ret = pd.rolling_apply(gross_ret, n, lambda x: x.prod())
    return gross_np_ret - 1

def _window_apply(df, ii, fn):
    window = df.iloc[map(int, ii),]
    return fn(window)

def rolling_apply(df, lookback, fn):
    index_series = pd.Series(range(len(df)))
    result = pd.rolling_apply(
        index_series, lookback, 
        lambda ii: _window_apply(df, ii, fn))
    
    result.index = df.index
    return result

def rolling_z_score(series, lookback=21):
    return (series - pd.rolling_mean(series, lookback)) / pd.rolling_std(series, lookback)




