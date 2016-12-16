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


import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARIMA


acf, aint=st.acf(a_ts, nlags=10, alpha=.05)
plt.figure(figsize=(15,7))
plt.stem(acf[1:])
plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
plt.title("ACF Plot")
plt.show()


pacf, pint=st.pacf(a_ts, nlags=10, alpha=.05)
plt.figure(figsize=(15,7))
plt.stem(pacf[1:])
plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
plt.title("PACF Plot")
plt.show()


model = ARIMA(a_ts, (2,2,1))

reg = model.fit()

res = reg.resid

acfr, aintr=st.acf(res, nlags=10, alpha=.05)
plt.figure(figsize=(15,7))
plt.stem(acfr[1:])
plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
plt.title("Residual ACF Plot")
plt.show()

pacfr, pintr=st.pacf(res, nlags=10, alpha=.05)
plt.figure(figsize=(15,7))
plt.stem(pacfr[1:])
plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
plt.title("Residual PACF Plot")
plt.show()

