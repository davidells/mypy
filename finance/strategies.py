import numpy as np
import pandas as pd

import hedge_ratio as hr
import util

def rolling_hedge_mean_revert_strategy(x, y, lookback):
    hedge_ratios = hr.rolling_hedge_ratio(x, y, lookback)
    weights = pd.DataFrame({
        'x': -hedge_ratios, 
        'y': np.ones(shape=len(y))
    })    
    
    prices = pd.DataFrame({'x': x, 'y': y})
    port = (prices * weights).sum(axis=1)
    
    # Keep units inverse to the rolling z score.
    # Remember, we're betting on mean reversion here.
    units = -util.rolling_z_score(port, lookback=lookback)

    return (prices, weights, units)

def rolling_ratio_mean_revert_strategy(x, y, lookback):
    hedge_ratios = pd.rolling_mean(x / y, lookback)
    weights = pd.DataFrame({
        'x': -hedge_ratios, 
        'y': np.ones(shape=len(y))
    })    
    
    prices = pd.DataFrame({'x': x, 'y': y})
    port = (prices * weights).sum(axis=1)
    
    # Keep units inverse to the rolling z score.
    # Remember, we're betting on mean reversion here.
    units = -util.rolling_z_score(port, lookback=lookback)

    return (prices, weights, units)

# TODO: Support scaling
def bollinger_band_units(zScore, entryZScore=1, exitZScore=0):
    longsEntry = zScore < -entryZScore
    longsExit = zScore >= exitZScore
    shortsEntry = zScore > entryZScore
    shortsExit = zScore <= exitZScore
    
    unitsLong = pd.Series(data=np.nan, index=zScore.index)
    unitsShort = pd.Series(data=np.nan, index=zScore.index)
    
    unitsLong[0] = 0
    unitsLong[longsEntry] = 1
    unitsLong[longsExit] = 0
    unitsLong = unitsLong.fillna(method='ffill')
    
    unitsShort[0] = 0
    unitsShort[shortsEntry] = -1
    unitsShort[shortsExit] = 0
    unitsShort = unitsShort.fillna(method='ffill')
    
    return unitsLong + unitsShort

def number_of_trades(units):
    units_change = (units - units.shift().fillna(0))
    return len(units_change[units_change != 0])

def trades_per_period(units):
    return number_of_trades(units) * 1.0 / len(units)

def time_in_market_percent(units):
    return len(units[units != 0]) * 1.0 / len(units)

def periods_in_market(units):
    return len(units[units != 0])

def report(units):
    return {
        'total_trades': number_of_trades(units),
        'trades_per_period': trades_per_period(units),
        'periods_per_trade': 1 / trades_per_period(units),
        'time_in_market': time_in_market_percent(units),
        'periods_in_market': periods_in_market(units)
    }

def report_str(units, name):
    return """%s
        Total trades: %d
        Trades per day: %.2f
        Days per trade: %.2f
        Percent time in market: %.2f
        Bars in market: %d
    """ % (
        name,
        number_of_trades(units),
        trades_per_period(units),
        1 / trades_per_period(units),
        time_in_market_percent(units),
        periods_in_market(units)
    )
