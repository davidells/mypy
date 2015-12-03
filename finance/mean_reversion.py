import numpy as np
import scipy as sp
import pandas as pd
import sklearn.linear_model as lm
import statsmodels.tsa.stattools as sms


import hedge_ratio as hr

def halflife(y):
    yDelta = y.diff()[2:]
    model = lm.LinearRegression()
    model = model.fit(y[2:,np.newaxis], yDelta)
    return np.log(2) / model.coef_

def cadf(x, y, method="ols", maxlag=1, regression="c"):
    beta = hr.hedge_ratio(x, y, method=method)
    return sms.adfuller(
        y - beta * x, 
        maxlag=maxlag, 
        regression=regression)

# LM_stat and vratiotest are direct ports of LM_stat and Lo.Mac from the R package 'vrtest'
def __LM_stat(y, k):
    n = len(y)
    m = y.mean()
    y1 = pow(y - m, 2)
    vr1 = sum(y1) / n
    
    flt = sp.signal.lfilter(
        np.repeat(1.0, k), [1.0], y)
    flt = flt[(k-1):]
    summ = (pow(flt - k * m, 2)).sum()
    
    vr2 = summ / (n * k)
    vr = vr2 / vr1
    
    tem1 = 2.0 * (2 * k - 1) * (k - 1)
    tem2 = 3.0 * k
    
    m1 = np.sqrt(n) * (vr - 1) / np.sqrt(tem1 / tem2)
    
    w = 4 * pow(1.0 - (np.array(range(1, k)) / (k * 1.0)), 2)
    dvec = np.zeros(k-1)
    for j in range(len(dvec)):
        ym = y1[(j+1):n].values * y1[0:(n-j-1)].values
        dvec[j] = (ym).sum() / pow(y1.sum(), 2)
    summ = w.dot(dvec)
    m2 = np.sqrt(n) * (vr - 1) * pow(n * summ, -0.5)
    
    return (m1, m2)

def vratiotest(y, kvec):
    n = len(y)
    mq = np.zeros((len(kvec), 2))
    for i in range(len(kvec)):
        mq[i,:] = __LM_stat(y, kvec[i])
    mq = pd.DataFrame(mq)
    mq.index = ["k=" + str(k) for k in kvec]
    mq.columns = ["M1", "M2"]
    return mq

# __rescaled_range adapted from HurstK in FGN package in R
def __rescaled_range(series):
    y = series - series.mean()
    s = y.cumsum()
    r = (s.max() - s.min()) / np.sqrt(pow(y,2).sum() / len(y))
    return r

# Implemented based on method discussed at
# http://www.bearcave.com/misl/misl_tech/wavelets/hurst/index.html
def hurst_exponent(series):
    length = len(series)
    upper_bound = int(np.floor(np.log2(length)))
    divisions = range(upper_bound)
    
    hurst_vals = np.zeros((len(divisions), 2))
    for i in divisions:
        n = int(np.floor(length / pow(2,i)))
        range_series = pd.rolling_apply(series, n, __rescaled_range)
        range_mean = range_series[-1::-n].mean()
        hurst_vals[i,:] = [np.log2(n), np.log2(range_mean)]
        
    hurst_lm = lm.LinearRegression()
    hurst_lm.fit(hurst_vals[:,0,None], hurst_vals[:,1])
    return hurst_lm.coef_

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