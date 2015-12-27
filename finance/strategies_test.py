import pandas as pd

import strategies as strat

from nose.tools import assert_almost_equals, assert_equals
from numpy.testing import assert_array_equal

def bollinger_band_units_test():
    example = pd.Series([-0.5, -1.0, -1.1, -0.5, 0, 1.0, 1.1, 0.5, 0])
    
    units = strat.bollinger_band_units(example, entryZScore=1, exitZScore=0)
    assert_array_equal(units.values, [0, 0, 1, 1, 0, 0, -1, -1, 0]) 
    
    units2 = strat.bollinger_band_units(example, entryZScore=0.5, exitZScore=0)
    assert_array_equal(units2.values, [0, 1, 1, 1, 0, -1, -1, -1, 0]) 