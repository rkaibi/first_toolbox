import pandas as pd
from first_toolbox.lib import make_prediction

data = pd.read_csv('raw_data/data', sep=",", header=None)
data.columns = ['text']

def test_function():
    assert(make_prediction(data, 2) == None)
