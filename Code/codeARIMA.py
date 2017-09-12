# Import pandas, numpy, and libraries for ARIMA models, 
#     for tools such as ACF and PACF functions, plotting,
#     and for using datetime formatting
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as st
from bokeh.plotting import figure, show
from datetime import datetime

# Collect data - Deprecated by Yahoo.... :(
data = pd.read_csv("/home/dusty/Econ8310/DataSets/pollutionBeijing.csv")

# data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

# data.drop(['No','year','month','day','hour'], axis=1, inplace=True)

# data.to_csv("/home/dusty/Econ8310/DataSets/pollutionBeijing.csv", index=False)

format = '%Y-%m-%d %H:%M:%S'
data['datetime'] = pd.to_datetime(data['datetime'], format=format)

data.set_index(pd.DatetimeIndex(data['datetime']), inplace=True)

# Plot the data
p = figure(plot_width = 1200, plot_height=400,
        y_axis_label="Pollution Level",
        x_axis_label="Date",
        x_axis_type="datetime")
p.line(data.index.values[1:], np.diff(np.log(data['pm2.5']))[1:])
show(p)

data.dropna(inplace=True)

# Generate plot from ACF
acf, aint=st.acf(data['pm2.5'], nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = acf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)


# Generate plot from PACF
pacf, paint=st.pacf(data['pm2.5'], nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = pacf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)



# Generate plot from ACF (DIFFERENCED)
acf, aint=st.acf(np.diff(data['pm2.5'])[1:], nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = acf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)


# Generate plot from PACF (DIFFERENCED)
pacf, paint=st.pacf(np.diff(data['pm2.5'])[1:], nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = pacf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)


model = ARIMA(data["pm2.5"], (1,1,0)) 
		  # specifying an ARIMA(1,1,0) model
reg = model.fit() # Fit the model using standard params
res = reg.resid   # store the residuals as res


# Generate plot from residual ACF
acf, aint=st.acf(res, nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = acf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)


# Generate plot from residual PACF
pacf, paint=st.pacf(res, nlags=30, alpha=.05)
# Create figure, add ACF values
p = figure(plot_width = 800, plot_height = 600)
p.vbar(x = list(range(1,31)), width = 0.5, top = pacf[1:],
	bottom = 0)
# Confidence Intervals
p.line(list(range(1,31)), [1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
p.line(list(range(1,31)), [-1/np.sqrt(len(data))]*30, 
	color = 'black', line_dash = "dashed")
show(p)


# Generating our Forecast
fcst = reg.forecast(steps=10) # Generate forecast
upper = fcst[2][:,1] # Specify upper 95% CI
lower = fcst[2][:,0] # Specify lower 95% CI

#Plotting a forecast
p = figure(plot_width = 1200, plot_height=400,
        y_axis_label="Pollution Level",
        x_axis_label="Date")
p.line(list(range(-98,0)), data['pm2.5'][-98:], 
    legend="Past Observations")
rng = list(range(0,10))
p.line(rng, fcst[0], color = 'red', 
    legend="Forecast")
p.line(rng, upper, color = 'red', line_dash = 'dashed', 
    legend="95% Confidence Interval")
p.line(rng, lower, color = 'red', line_dash = 'dashed')
p.legend.location="top_left"
show(p)


########################################################


# Import pandas, numpy, and libraries for ARIMA models, 
#     for tools such as ACF and PACF functions, plotting,
#     and for using datetime formatting
import pandas as pd
import numpy as np
import patsy as pt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as st
from bokeh.plotting import figure, show
from datetime import datetime

data = pd.read_csv("/home/dusty/Econ8310/DataSets/omahaNOAA.csv")[-(365*24):]
		# We are keeping only the last 365 days

p = figure(plot_width = 1200, plot_height=400,
        y_axis_label="Temperature",
        x_axis_label="Date/Time")
p.line(data.index.values, data.HOURLYDRYBULBTEMPF,
	legend="Past Observations")
show(p)



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

