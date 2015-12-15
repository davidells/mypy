import numpy as np
import pandas as pd
import scipy.odr as odr

from sklearn.linear_model import LinearRegression

import util

# Concepts adapted from http://quanttrader.info/public/betterHedgeRatios.pdf
def hedge_ratio(x, y, method="ols"):
    
    if method == "ols":
        model = LinearRegression().fit(x[:,None], y)
        return model.coef_[0]

    elif method == "tls":
        return odr.ODR(
            odr.Data(x, y), 
            odr.Model(lambda B,x: B[0]*x + B[1]),
            beta0=[0,0]).run().beta[0]
    else:
        raise ValueError("method must equal 'ols' (ordinary least squares) "
                         " or 'tls' (total least squares)")

def rolling_hedge_ratio(x, y, lookback):
    df = pd.DataFrame({'x': x, 'y': y})
    return util.rolling_apply(df, lookback, 
        lambda win: hedge_ratio(win['x'], win['y']))