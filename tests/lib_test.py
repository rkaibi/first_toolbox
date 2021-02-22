import pandas as pd
from first_toolbox.lib import make_pred

data = pd.read_csv('raw_data/data', sep=",", header=None)
data.columns = ['text']

def test_function():
    assert(round(make_pred(data, 2, ['the team is winning'])[0,0],1) == 0.2)
