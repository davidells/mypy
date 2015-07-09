import os
import pandas as pd
import pandas.io.data as web
import scipy.odr as odr

from datetime import datetime
from hedge_ratio import hedge_ratio
from sklearn.linear_model import LinearRegression

from nose.tools import assert_almost_equals, raises

def setup_module():
    global EWA, EWC
    path = os.path.dirname(os.path.abspath(__file__))
    EWA = pd.read_csv(path + "/EWA.csv")['Adj Close']
    EWC = pd.read_csv(path + "/EWC.csv")['Adj Close']
    
def test_hedge_ratio_ols():
    model = LinearRegression().fit(EWA[:,None], EWC)
    assert_almost_equals(
        model.coef_, 
        hedge_ratio(EWA, EWC, method="ols"),
        places=10)
    
def test_hedge_ratio_tls():
    model = odr.ODR(
        odr.Data(EWA, EWC),
        odr.Model(lambda B,x: B[0]*x + B[1]),
        beta0=[0,0]).run()

    assert_almost_equals(
        model.beta[0], 
        hedge_ratio(EWA, EWC, method="tls"),
        places=10)

@raises(ValueError)
def test_hedge_ratio_bad_arg():
    hedge_ratio(EWA, EWC, method="unknown")