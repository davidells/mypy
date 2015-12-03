import numpy as np
import pandas as pd
import util

from nose.tools import assert_true, assert_equals, assert_almost_equals

def empty_series_test():
    s = pd.Series(range(10))
    s2 = util.empty_series(s)
    assert_equals(len(s2), 10)
    for e in s2:
        assert_true(np.isnan(e))
        
def ma_test():
    s = pd.Series(range(10))
    assert_true(util.ma(s, 3).equals(pd.rolling_mean(s, 3)))

def n_period_return_test():
    s = pd.Series(range(10))
    r = util.n_period_return(s, 3)
    assert_true(np.isnan(r[0]))
    assert_true(np.isnan(r[1]))
    assert_true(np.isnan(r[2]))
    assert_true(np.isnan(r[3]))
    assert_almost_equals(r[4], 3)
    assert_almost_equals(r[5], 1.5)
    assert_almost_equals(r[6], 1)
    assert_almost_equals(r[7], 0.75)
    assert_almost_equals(r[8], 0.6)
    assert_almost_equals(r[9], 0.5)
    
def rolling_apply_test():
    s = pd.Series(range(10))
    s2 = util.rolling_apply(s, 3, lambda x: min(x))
    
    assert_true(np.isnan(s2[0]))
    assert_true(np.isnan(s2[1]))
    for i in range(2, 10):
        assert_equals(s2[i], s[i - 2])
        
def rolling_z_score_test():
    s = pd.Series(range(10)).cumsum()
    z = util.rolling_z_score(s, lookback=3)
    assert_true(np.isnan(z[0]))
    assert_true(np.isnan(z[1]))
    assert_almost_equals(z[2], 1.091089, places=6)
    assert_almost_equals(z[3], 1.059626, places=6)
    assert_almost_equals(z[4], 1.044074, places=6)
    assert_almost_equals(z[5], 1.034910, places=6)
    assert_almost_equals(z[6], 1.028887, places=6)
    assert_almost_equals(z[7], 1.024631, places=6)
    assert_almost_equals(z[8], 1.021466, places=6)
    assert_almost_equals(z[9], 1.019020, places=6)
       
