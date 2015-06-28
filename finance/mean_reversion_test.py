import pandas as pd
import mean_reversion as mr
from nose.tools import assert_almost_equals

def halflife_test():
    y = pd.Series([1,2,3,2,1,2,3,2,1,2,3,2,1])
    hl = mr.halflife(y)
    assert_almost_equals(hl, 0.693, places=3)
