import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime

import patsy as pt
from bokeh.plotting import figure, show

a = DataReader('JPM',  'yahoo', datetime(2006,6,1), datetime(2016,6,1))
a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
a_returns.columns = ["Returns"]
a_returns['date'] = a.index.values[1:a.index.values.shape[0]]


a_ts = pd.DataFrame(np.log(a['Adj Close'].values))
a_ts.columns = ["Index"]
a_ts['date'] = a.index.values
a_ts.index = a_ts['date']


# plt.figure(figsize=(15, 5))
# plt.ylabel("Returns")
# plt.plot(a_returns)
# plt.show()

# Plot returns

p = figure(plot_width = 1200, plot_height=400,
        y_axis_label="Returns",
        x_axis_label="Date",
        x_axis_type="datetime")
p.line(a_ts['date'][1:], np.diff(a_ts["Index"])[1:])
show(p)

# Plot logged value

p = figure(plot_width = 1200, plot_height=400,
        y_axis_label="Log Value",
        x_axis_label="Date",
        x_axis_type="datetime")
p.line(a_ts['date'], a_ts['Index'])
show(p)

# plt.figure(figsize=(15, 5))
# plt.ylabel("Log Value")
# plt.plot(a_ts["Index"])
# plt.show()


import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARIMA


acf, aint=st.acf(a_ts['Index'], nlags=10, alpha=.05)
# plt.figure(figsize=(15,7))
# plt.stem(acf[1:])
# plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
# plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
# plt.title("ACF Plot")
# plt.show()

p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,11)), width = 0.5, top = acf[1:], bottom = 0)
p.line(list(range(1,11)), [1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
p.line(list(range(1,11)), [-1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
show(p)

pacf, pint=st.pacf(a_ts['Index'], nlags=10, alpha=.05)
# plt.figure(figsize=(15,7))
# plt.stem(pacf[1:])
# plt.plot([1/np.sqrt(len(a_ts))]*10, 'k--')
# plt.plot([-1/np.sqrt(len(a_ts))]*10, 'k--')
# plt.title("PACF Plot")
# plt.show()

p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,11)), width = 0.5, top = pacf[1:], bottom = 0)
p.line(list(range(1,11)), [1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
p.line(list(range(1,11)), [-1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
show(p)



model = ARIMA(a_ts['Index'], (1,1,1))

reg = model.fit()

res = reg.resid

acfr, aintr=st.acf(res, nlags=10, alpha=.05)
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,11)), width = 0.5, top = acfr[1:], bottom = 0)
p.line(list(range(1,11)), [1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
p.line(list(range(1,11)), [-1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
show(p)

pacfr, pintr=st.pacf(res, nlags=10, alpha=.05)
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,11)), width = 0.5, top = pacfr[1:], bottom = 0)
p.line(list(range(1,11)), [1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
p.line(list(range(1,11)), [-1/np.sqrt(len(a_ts))]*10, color = 'black', line_dash = "dashed")
show(p)

fcst = reg.forecast(steps=10)
future = pd.DatetimeIndex(start=datetime(2016,6,2), freq='D', periods=10)
predicted = pd.DataFrame(fcst[0], columns = ['Index'], index = future)
upper = fcst[2][:,1]
lower = fcst[2][:,0]

# plt.figure(figsize=(15,7))
# plt.plot(a_ts[-100:])
# plt.plot(predicted)
# plt.fill_between(x=future, y1 = lower, y2=upper, alpha=0.2)
# plt.show()

p = figure(plot_width = 1200, plot_height=600,
        y_axis_label="Log Value",
        x_axis_label="Date")
p.line(list(range(-98,0)), a_ts['Index'][-98:], legend="Past Observations")
rng = list(range(0,10))
p.line(rng, predicted['Index'], color = 'red', legend="Forecast")
p.line(rng, upper, color = 'red', line_dash = 'dashed', legend="95% Confidence Interval")
p.line(rng, lower, color = 'red', line_dash = 'dashed')
p.legend.location="top_left"

show(p)

