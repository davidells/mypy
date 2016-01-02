import numpy as np
import pandas as pd

import statsmodels.tsa.stattools as sms

import hedge_ratio as hr
import mean_reversion as mr
import optimize
import portfolio
import returns
import util

def rolling_hedge_mean_revert_strategy(x, y, lookback, hedge_lookback=None):
    if hedge_lookback is None:
        hedge_lookback = lookback
        
    hedge_ratios = hr.rolling_hedge_ratio(x, y, hedge_lookback)
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

def bollinger_mean_revert_fit(x, y, model=None, index=None):
    model = model or {}
    
    def strategy(x, y, lookback, hedge_lookback, entry_z_score, exit_z_score):
        results = bollinger_mean_revert(x, y, {
            'lookback': lookback,
            'hedge_lookback': hedge_lookback,
            'entry_z_score': entry_z_score,
            'exit_z_score': exit_z_score
        }, index=index)
        return (results['prices'], results['weights'], results['units'])
    
    #strategy = rolling_hedge_mean_revert_strategy

    lookback = model.get('lookback')
    hedge_lookback = model.get('hedge_lookback')
    
    entry_z_score = model.get('entry_z_score') or 1.0
    optimize_entry_z_score = entry_z_score == 'optimize'
    if optimize_entry_z_score:
        entry_z_score = 1.0
    
    exit_z_score = model.get('exit_z_score') or 0.0
    optimize_exit_z_score = exit_z_score == 'optimize'
    if optimize_exit_z_score:
        exit_z_score = 0.0
    
    debug = model.get('debug') or False
    
    # default for lookback is to optimize
    lookback = lookback or optimize.vs_sharpe(
            'lookback', [10, 15, 20, 25, 50],
            lambda lb: strategy(x, y, lb, hedge_lookback, entry_z_score, exit_z_score),
            index=index,
            debug=debug)
    
    # default for hedge lookback is to optimize
    if hedge_lookback is None:
        hedge_lookback = optimize.vs_sharpe(
                'hedge_lookback', [10, 15, 20, 25, 50],
                lambda hlb: strategy(x, y, lookback, hlb, entry_z_score, exit_z_score),
                index=index,
                debug=debug)
    # Special keywork 'lookback' can be used
    elif hedge_lookback == 'lookback':
        hedge_lookback = lookback
        
    # Optimize entry Z score if set to 'optimize'
    if optimize_entry_z_score:
        entry_z_score = optimize.vs_sharpe(
                'entry_z_score', np.arange(0, 2.01, 0.5),
                lambda entry: strategy(x, y, lookback, hedge_lookback, entry, exit_z_score),
                index=index,
                debug=debug)
    
    # Optimize exit Z score if set to 'optimize'
    if optimize_exit_z_score:
        exit_z_score = optimize.vs_sharpe(
                'exit_z_score', np.arange(0, 1.01, 0.5),
                lambda exit: strategy(x, y, lookback, hedge_lookback, entry_z_score, exit),
                index=index,
                debug=debug)
        
        
    fit = { 'type': 'bollinger_mean_revert' }
    fit.update(model)
    fit.update({
        'lookback': lookback,
        'hedge_lookback': hedge_lookback,
        'entry_z_score': entry_z_score,
        'exit_z_score': exit_z_score
    })
    return fit

def bollinger_mean_revert(x, y, model=None, index=None):
    if model is None:
        model = bollinger_mean_revert_fit(x, y, index=index)
        
    if index is None:
        index = x.index

    lookback = model['lookback']
    hedge_lookback = model['hedge_lookback']
    entry_z_score = model['entry_z_score']
    exit_z_score = model['exit_z_score']
    static_hedges = model.get('static_hedges') or False
    
    # Setup mean reversion portfolio to use for tradable zScore
    strategy = rolling_hedge_mean_revert_strategy
    (prices, weights, units) = strategy(x, y, lookback, hedge_lookback)
    
    prices = prices.loc[index]
    weights = weights.loc[index]
    units = units[index]
    
    port = portfolio.portfolio_price(prices, weights)
    
    # Ok, now let's look at the half-life of this portfolio, this needs to be low (under 30)
    # TODO: Test this? Throw it out if it doesn't meet standard?
    halflife = mr.halflife(port)
    cadf = sms.adfuller(port, maxlag=1, regression='c')[0]
    hurst = mr.hurst_exponent(port.pct_change().fillna(0))[0]
    
    # Collect zScore and theoretical results in the form of a sharpe ratio
    zScore = util.rolling_z_score(port, lookback)
    rets = portfolio.portfolio_returns(prices, weights, -zScore)['returns']
    theoretical_sharpe = returns.annual_sharpe(rets)
    
    # Collect units of this portfolio according to bollinger band strategy (buying and selling
    # against the given zScore)    
    units = bollinger_band_units(zScore, entryZScore=entry_z_score, exitZScore=exit_z_score)
    
    # Random experiment - static weights. Basically, force hedge weights to be static for the duration
    # of any particular trade.
    if static_hedges == True:
        unit_changes = units != units.shift().fillna(0)
        weights[~unit_changes] = np.nan
        weights['x'][0] = 0
        weights['y'][0] = 0
        weights = weights.fillna(method='ffill')

    # Finally, collect the results        
    results = {
        'lookback': lookback,
        'hedge_lookback': hedge_lookback,
        'static_hedges': static_hedges,
        'halflife': halflife,
        'cadf': cadf,
        'hurst': hurst,
        'theoretical_sharpe': theoretical_sharpe,
        'prices': prices,
        'weights': weights,
        'units': units
    }
        
    port_rets = portfolio.portfolio_returns(prices, weights, units)
    port_rets['returns'] = port_rets['returns'][index]
    port_rets['pnl'] = port_rets['pnl'][index]
    port_rets['grossMktVal'] = port_rets['grossMktVal'][index]
        
    results.update(port_rets)
    
    rets = port_rets['returns']
    results.update(
        returns.report(rets))
    
    results.update(
        report(units))
    
    return results

def number_of_trades(units):
    units_change = (units - units.shift().fillna(0))
    return len(units_change[units_change != 0])

def trades_per_period(units):
    return number_of_trades(units) * 1.0 / len(units)

def periods_per_trade(units):
    tpp = trades_per_period(units)
    return tpp > 0 and 1 / tpp or 0

def time_in_market_percent(units):
    return len(units[units != 0]) * 1.0 / len(units)

def periods_in_market(units):
    return len(units[units != 0])

def report(units):
    return {
        'total_trades': number_of_trades(units),
        'trades_per_period': trades_per_period(units),
        'periods_per_trade': periods_per_trade(units),
        'time_in_market': time_in_market_percent(units),
        'periods_in_market': periods_in_market(units)
    }

def report_str(units, name):
    return """%s
        Total trades: %d
        Trades per period: %.2f
        Periods per trade: %.2f
        Percent time in market: %.2f
        Bars in market: %d
    """ % (
        name,
        number_of_trades(units),
        trades_per_period(units),
        periods_per_trade(units),
        time_in_market_percent(units),
        periods_in_market(units)
    )
