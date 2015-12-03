import os
import pandas as pd
import mean_reversion as mr

from nose.tools import assert_almost_equals, assert_equals
from numpy.testing import assert_array_equal

def setup_module():
    global EWA, EWC
    path = os.path.dirname(os.path.abspath(__file__))
    EWA = pd.read_csv(path + "/EWA.csv")['Adj Close']
    EWC = pd.read_csv(path + "/EWC.csv")['Adj Close']

def halflife_test():
    assert_almost_equals(
        mr.halflife(EWA), 127.63, places=2)

def cadf_ols_test():
    result = mr.cadf(EWA, EWC, method="ols")
    assert_almost_equals(result[0], -3.6375, places=4)
    assert_almost_equals(result[1],  0.0051, places=4)
    
def cadf_tls_test():
    result = mr.cadf(EWA, EWC, method="tls")
    assert_almost_equals(result[0], -3.6667, places=4)
    assert_almost_equals(result[1],  0.0046, places=4)
    
def vratiotest_test():
    result = mr.vratiotest(EWA, (4,16))
    assert_almost_equals(result.values[0][0], 61.4775, places=4)
    assert_almost_equals(result.values[0][1], 36.6405, places=4)
    assert_almost_equals(result.values[1][0], 127.7714, places=4)
    assert_almost_equals(result.values[1][1], 76.8186, places=4)
    
def hurst_exponent_test():
    path = os.path.dirname(os.path.abspath(__file__))
    bwn = pd.read_csv(path + "/brown72.h", header=None)[[0]]
    result = mr.hurst_exponent(bwn)
    assert_almost_equals(result, 0.724765, places=6)
    
def bollinger_band_units_test():
    example = pd.Series([-0.5, -1.0, -1.1, -0.5, 0, 1.0, 1.1, 0.5, 0])
    
    units = mr.bollinger_band_units(example, entryZScore=1, exitZScore=0)
    assert_array_equal(units.values, [0, 0, 1, 1, 0, 0, -1, -1, 0]) 
    
    units2 = mr.bollinger_band_units(example, entryZScore=0.5, exitZScore=0)
    assert_array_equal(units2.values, [0, 1, 1, 1, 0, -1, -1, -1, 0]) 