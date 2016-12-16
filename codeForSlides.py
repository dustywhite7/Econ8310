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


import pyflux as pf

pf.acf_plot(a_returns.values.T[0])
pf.acf_plot(np.square(a_returns.values.T[0]))


my_model = pf.GARCH(p=1,q=1, data=a_returns)

print(my_model.latent_variables)

my_model.adjust_prior(1, pf.TruncatedNormal(0.01, 0.5, lower=0.0, upper=1.0))
my_model.adjust_prior(1, pf.TruncatedNormal(0.97, 0.5, lower=0.0, upper=1.0))

result = my_model.fit('M-H', nsims=20000)

my_model.plot_z([1,2])

my_model.plot_fit(figsize=[15,5])

my_model.plot_sample(nsims=10, figsize=(15,7))

from scipy.stats import kurtosis

my_model.plot_ppc(T=kurtosis)


my_model.plot_predict(h=30, figsize=(15,5))
