import numpy as np
import sklearn.linear_model as lm

def halflife(y):
    yDelta = y.diff()[2:]
    model = lm.LinearRegression()
    model = model.fit(y[2:,np.newaxis], yDelta)
    return np.log(2) / model.coef_
