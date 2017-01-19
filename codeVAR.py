import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime

import patsy as pt
import matplotlib.pyplot as plt
import mpld3

a = DataReader('JPM',  'yahoo', datetime(2006,6,1), datetime(2016,6,1))
a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
a_returns.index = a.index.values[1:a.index.values.shape[0]]
a_returns.columns = ["Returns"]

a_ts = pd.DataFrame(np.log(a['Adj Close'].values))
a_ts.index = a.index.values
a_ts.columns = ["Index"]


plt.figure(figsize=(15, 5))
plt.ylabel("Returns")
plt.plot(a_returns)
plt.show()

plt.figure(figsize=(15, 5))
plt.ylabel("Log Value")
plt.plot(a_ts["Index"])
plt.show()

from statsmodels.tsa.api import VAR

model = VAR(a)
model.select_order() # uses information criteria to select the order of the model
reg = model.fit(2) # number of AR terms to include

reg.plot() # plots the fitted model

