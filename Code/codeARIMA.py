# Import pandas, numpy, and libraries for ARIMA models, 
#     for tools such as ACF and PACF functions, plotting,
#     and for using datetime formatting
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from plotly.offline import plot
import plotly.graph_objs as go

# Collect data - Deprecated by Yahoo.... :(
data = pd.read_csv("/home/dusty/Econ8310/DataSets/pollutionBeijing.csv")

format = '%Y-%m-%d %H:%M:%S'
data['datetime'] = pd.to_datetime(data['datetime'], format=format)

data.set_index(pd.DatetimeIndex(data['datetime']), inplace=True)

# Plot the data

trace = go.Scatter(
#    x = data['datetime'][1:], # differenced
#    y = np.diff(np.log(data['pm2.5']))[1:], # differenced
    x = data['datetime'], # non-diff
    y = np.log(data['pm2.5']), # non-diff
    mode = 'lines',
    )

pdata = go.Data([trace])

layout = go.Layout(
    title=None,
    xaxis = dict(title = 'Date', type='date'),
    yaxis = dict(title = 'Pollution Level')
    )

plot(go.Figure(data=pdata, layout=layout))

# Drop missing values
data.dropna(inplace=True)

# Generate plot from ACF
acf, aint=st.acf(data['pm2.5'], nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = acf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))

# Generate plot from PACF
pacf, paint=st.pacf(data['pm2.5'], nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = pacf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))



# Generate plot from ACF (DIFFERENCED)
acf, aint=st.acf(np.diff(data['pm2.5'])[1:], nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = acf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))

# Generate plot from PACF (DIFFERENCED)
pacf, paint=st.pacf(np.diff(data['pm2.5'])[1:], nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = pacf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))


# Specify and Fit a Model

model = ARIMA(data["pm2.5"], (1,1,0)) 
		  # specifying an ARIMA(1,1,0) model
reg = model.fit() # Fit the model using standard params
res = reg.resid   # store the residuals as res


# Generate plot from residual ACF
acf, aint=st.acf(res, nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = acf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))


# Generate plot from residual PACF
pacf, paint=st.pacf(res, nlags=30, alpha=.05)

trace = go.Scatter(
    x = list(range(1,31)),
    y = [1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace1 = go.Scatter(
    x = list(range(1,31)),
    y = [-1/np.sqrt(len(data))]*30,
    line = dict(dash='dash', color='black')
    )

trace2 = go.Bar(
    x = list(range(1,31)),
    y = pacf[1:],
    marker = dict(color='grey')
    )

pdata = go.Data([trace, trace1, trace2])

plot(go.Figure(data=pdata))


# Generating our Forecast
fcst = reg.forecast(steps=10) # Generate forecast
upper = fcst[2][:,1] # Specify upper 95% CI
lower = fcst[2][:,0] # Specify lower 95% CI

#Plotting a forecast
trace = go.Scatter(
    x = list(range(0,10)),
    y = upper,
    mode = 'lines',
    line = dict(dash='dash', color='grey'),
    name = '95% Confidence Interval'
    )

trace1 = go.Scatter(
    x = list(range(0,10)),
    y = lower,
    mode = 'lines',
    line = dict(dash='dash', color='grey'),
    name = '95% Confidence Interval',
    showlegend = False
    )

trace2 = go.Scatter(
    x = list(range(0,10)),
    y = fcst[0],
    mode = 'lines',
    line = dict(dash='dash', color='black'),
    name = 'Forecast'
    )

trace3 = go.Scatter(
    x = list(range(-98,0)),
    y = data['pm2.5'][-98:],
    line = dict(color='black'),
    name = 'Data'
    )

pdata = go.Data([trace, trace1, trace2, trace3])

layout = go.Layout(
    xaxis=dict(title="Days After End of Data"),
    yaxis=dict(title="Pollution Level"),
    width=1200,
    height=400
    )

plot(go.Figure(data=pdata, layout=layout))


########################################################


# Import pandas, numpy, and libraries for ARIMA models, 
#     for tools such as ACF and PACF functions, plotting,
#     and for using datetime formatting
import pandas as pd
import numpy as np
import patsy as pt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as st
from plotly.offline import plot
import plotly.graph_objs as go

data = pd.read_csv("/home/dusty/Econ8310/DataSets/omahaNOAA.csv")[-(365*24):]
		# We are keeping only the last 365 days

#p = figure(plot_width = 1200, plot_height=400,
#        y_axis_label="Temperature",
#        x_axis_label="Date/Time")
#p.line(data.index.values, data.HOURLYDRYBULBTEMPF,
#	legend="Past Observations")
#show(p)


trace = go.Scatter(
    x = data.DATE,
    y = data.HOURLYDRYBULBTEMPF,
    line = dict(color='black'),
    name = 'Data'
    )

pdata = go.Data([trace])

layout = go.Layout(
    xaxis=dict(title="Date/Time", type='date'),
    yaxis=dict(title="Temperature (F)"),
    width=1200,
    height=400
    )

plot(go.Figure(data=pdata, layout=layout))


data = data[data.HOURLYDRYBULBTEMPF!=0]

### ARIMAX

# First, let's difference our data TWICE
data['HOURLYDRYBULBTEMPF'] = data['HOURLYDRYBULBTEMPF'].diff(periods=1)
data['HOURLYDRYBULBTEMPF'] = data['HOURLYDRYBULBTEMPF'].diff(periods=24)

eqn = "HOURLYDRYBULBTEMPF ~ HOURLYWindSpeed + HOURLYStationPressure + HOURLYPrecip"
        
y, x = pt.dmatrices(eqn, data = data)

# The exog argument permits us to include exogenous vars
model = ARIMA(y, order=(1,1,0), exog=x)
reg = model.fit(trend='nc', method='mle', 
		maxiter=500, solver='nm')
reg.summary()

# Generating our Forecast
fcst = reg.forecast(steps=10, exog=x[-10:]) # Generate forecast
upper = fcst[2][:,1] # Specify upper 95% CI
lower = fcst[2][:,0] # Specify lower 95% CI

