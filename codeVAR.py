from __future__ import division, print_function

import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime
import patsy as pt
import matplotlib.pyplot as plt
import mpld3
import itertools

a = DataReader('JPM',  'yahoo', datetime(2006,6,1), datetime(2016,6,1))

# Differencing observations to obtain stationary data

a_diff = pd.DataFrame(np.diff(a.values, axis=0), 
	index=a.index.values[1:], # re-applying index
    	columns=a.columns) # re-applying column names


# plt.figure(figsize=(15, 5))
# plt.ylabel("Returns")
# plt.plot(a_returns)
# plt.show()

# plt.figure(figsize=(15, 5))
# plt.ylabel("Log Value")
# plt.plot(a_ts)
# plt.show()

from statsmodels.tsa.api import VAR

model = VAR(a_diff[:'2016-01-01'])
model.select_order() # uses information criteria to select the order of the model
reg = model.fit(5) # number of AR terms to include


sample = a_diff[:'2016-01-04'].values
fcast = reg.forecast(y = sample, steps = 10)


# plt.plot(fcast[:,3])
reg.plot_forecast(20)


def dediff(end, forecast):
    future = np.copy(forecast)
    for i in range(np.shape(forecast)[0]):
        if (i==0):
            future[i] = end + forecast[0]
        else:
            future[i] = future[i-1] + forecast[i]
            
    return future



nextPer = pd.DataFrame(dediff(a['2016-01-04':'2016-01-04'], fcast), index=pd.DatetimeIndex(start=datetime(2016,1,5), freq='B', periods=10), columns=a.columns)
rNext = a['2016-01-05':'2016-01-19']


#Volume Plot
plt.figure(figsize=(12, 8))
plt.plot(nextPer['Volume'], '--', label='Forecast')
plt.plot(rNext['Volume'], '--', label='Truth')
plt.title('Volume Forecast')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#Close Plot
plt.figure(figsize=(12, 8))
plt.plot(nextPer['Close'], '--', label='Forecast')
plt.plot(rNext['Close'], '--', label='Truth')
plt.title('Closing Price Forecast')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# All plots compared
plt.figure(figsize=(12, 8))
plt.plot(nextPer.drop('Volume', axis=1), '--')
plt.plot(rNext.drop('Volume', axis=1), '-')
plt.title('Stock Forecasts w/o Volume')
plt.xticks(rotation=45)
plt.show()






