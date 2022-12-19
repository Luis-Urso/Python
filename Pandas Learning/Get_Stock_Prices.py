import numpy as np
import pandas as pd
import plotly.graph_objs as go

from pandas_datareader import data,wb
from datetime import date

startdate = pd.to_datetime('2018-07-15')
enddate = pd.to_datetime(date.today())

data.DataReader("SPY",'yahoo',startdate,enddate)

print(data)