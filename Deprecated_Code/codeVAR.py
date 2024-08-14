import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import statsmodels.tsa.stattools as st
from bokeh.plotting import figure, show
from datetime import datetime

# Collect data
data = pd.read_csv("/home/dusty/Econ8310/DataSets/pollutionBeijing.csv")

format = '%Y-%m-%d %H:%M:%S'
data['datetime'] = pd.to_datetime(data['datetime'], format=format)
data.set_index(pd.DatetimeIndex(data['datetime']), inplace=True)


# Select variables for VAR model
varData = data[['pm2.5','TEMP','PRES', 'Iws']].dropna()[:-50]
test = data[['pm2.5','TEMP','PRES', 'Iws']].dropna()[-50:]

# endVal = varData.loc["2014-01-04 00:00:00"]
# varData = varData.diff(1)


model = VAR(varData) # define the model and data
# model.select_order() # uses information criteria to select
		     # model order
reg = model.fit(30) # order chosen based on BIC criterion



# Forecasting

fcast = reg.forecast(varData['2013-01-04':].values, steps = 50)


def dediff(todaysVal, forecast):
    future = forecast
    for i in range(np.shape(forecast)[0]):
        if (i==0):
            future[i] = todaysVal + forecast[0]
        else:
            future[i] = future[i-1] + forecast[i]
            
    return future

nextPer = pd.DataFrame(fcast,
		#dediff(endVal, fcast),
            	pd.DatetimeIndex(start=datetime(2014,12,29,22),
                freq='H', periods=50),
                columns=varData.columns)
			

#Pm 2.5 Plot
p = figure(plot_width=800, plot_height=600, 
	x_axis_type='datetime')
p.line(nextPer.index.values, nextPer['pm2.5'], 
	color = 'red', line_width=3,
    	line_dash='dashed', alpha=0.5, 
    	legend='Forecast')
p.line(test.index.values, test['pm2.5'], 
	color = 'blue', line_width=3,
        alpha=0.5, legend='Truth')
show(p)

# Temperature Plot
p = figure(plot_width=800, plot_height=600, 
	x_axis_type='datetime')
p.line(nextPer.index.values, nextPer['TEMP'], 
	color = 'red', line_width=3,
    	line_dash='dashed', alpha=0.5, 
    	legend='Forecast')
p.line(test.index.values, test['TEMP'], 
	color = 'blue', line_width=3,
        alpha=0.5, legend='Truth')
show(p)

# Pressure Plot
p = figure(plot_width=800, plot_height=600, 
	x_axis_type='datetime')
p.line(nextPer.index.values, nextPer['PRES'], 
	color = 'red', line_width=3,
    	line_dash='dashed', alpha=0.5, 
    	legend='Forecast')
p.line(test.index.values, test['PRES'], 
	color = 'blue', line_width=3,
        alpha=0.5, legend='Truth')
show(p)


# Iws Plot
p = figure(plot_width=800, plot_height=600, 
	x_axis_type='datetime')
p.line(nextPer.index.values, nextPer['Iws'], 
	color = 'red', line_width=3,
    	line_dash='dashed', alpha=0.5, 
    	legend='Forecast')
p.line(test.index.values, test['Iws'], 
	color = 'blue', line_width=3,
        alpha=0.5, legend='Truth')
show(p)



irf = reg.irf(10) # 10-period Impulse Response Fn
irf.plot(impulse = 'Iws') # Plot volume change impact
irf.plot_cum_effects(impulse = 'Iws') # Plot cumulative effects

