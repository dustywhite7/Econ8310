---
marp: true
title: Week 2,3 - (S)ARIMA(X)
theme: default
class: default
size: 4:3
---


# Lectures 2 & 3: Time Series, ARIMA Models 
##### This lesson is based on material by [Robert Nau, Duke University](http://people.duke.edu/~rnau/forecasting.htm)

---

# Refresher:
### Using Statsmodels to implement OLS

<br>

```python
import statsmodels.api as sm
import patsy as pt

# NOTE: Make the file a single line again!
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/blob/master/
	DataSets/pollutionBeijing.csv?raw=true")
y, x = pt.dmatrices("Q('pm2.5') ~ TEMP", data=data)
model = sm.ols(exog=x, endog=y)
reg = model.fit() # Fit the model using standard params
```

---

# Time Series Data


A time series consists of repeated observations of a single variable, $y$, at various times, $t$.

$$\mathbf{y}=\{y_1, y_2, y_3, ..., y_t\}  $$

We seek to predict $y_{t+1}$ using the information from previous observations $\mathbf{y}$.


---

# Time Series Data

In order to estimate $y_{t+1}$, we need to find the effect of previous observations of $y$ on the upcoming period. We might write this model as

<br>

$$ y_{t+1}=\alpha + \sum_{s=1}^t\beta_s\cdot y_s + \epsilon $$

---

# Time Series Data

If we choose to base our model solely on the previous period, then the model would be written

<br>

$$ y_{t+1}=\alpha + \beta_t \cdot y_t + \epsilon $$

<br>

Critically, OLS estimates of this model are invalid.

---

# Autocorrelation

One of the primary assumptions of the OLS model is that

$$Cov(\epsilon_t,\epsilon_s) = 0, \;\forall\; t \neq s $$

This assumption is clearly **not** valid in the case of time series data.

Let's look at some data to find out why.

---

# Autocorrelation

![](autocorrPlot.png)

---

# Autocorrelation

![w:400](autocorrPlot.png)

We need to find a model that can eliminate the autocorrelation almost always seen in time series data.

---

# Autoregressive Models

AR models are based on the premise that  deviation from the underlying trend in the data persists in **all future observations**.


$$ y_{t} = \alpha + \sum_{i=1}^p \rho_i \cdot y_{t-i} + \epsilon_t $$


Here $\rho$ is the correlation term between periods and $\epsilon$ is an error (shock) term

---

# AR Models

- We need to consider lagged observations of $y$ in order to predict future outcomes
- The number of lags that we include is the **order** of our AR model
	- The model is an AR(p) Model, where p is the order of the model

---

# AR Models

- The AR coefficients tell us how quickly a model returns to its mean
	- If the coefficients on AR variables add up to close to 1, then the model reverts to its mean **slowly**
	- If the coefficients sum to near zero, then the model reverts to its mean **quickly**


---

# Integrated Models

Integration occurs when a process is non-stationary. A non-stationary process is one that contains a linear time trend. One example might be a long-term series of stock prices:

![w:550](autocorrPlot.png)

---

# Integrated Models

We need to ensure that our data is stationary. To do so, we need to remove any time-trend from the data.
- This is typically done through differencing

$$ y^s_i = y_i - y_{i-1} $$

where $y^s_t$ is the stationary time series based on the original series $y_t$


---

# Integrated Models

Here,  the time trend has been differenced out of the data:


![](stationary.png)


---

# Integrated Models

The Integration term $d$ represents the number of differencing operations performed on the data:
- I(1): $y^s_t = y_t - y_{t-1}$
- I(2): $y^s_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})$

Where an I(2) model is analogous to a standard difference-in-differences model applied to time-series data.

---

# Moving Average Models

While an AR($\cdot$) model accounts for previous values of the dependent variable, MA($\cdot$) models account for previous values of the **error** terms:

$$ AR(p) = \alpha + \sum_{i=1}^p \rho_i\cdot y_{t-i} + \epsilon_t $$
$$ MA(q) = \alpha + \sum_{i=1}^q \theta_i\cdot \epsilon_{t-i} + \epsilon_t $$

---

# Moving Average Models

An MA model suggests that the current value of a time-series depends linearly on previous error terms.
- Current value depends on how far away from the underlying trend previous periods fell
- The larger $\theta$ becomes, the more persistent those error terms are

---

# Moving Average Models

- AR models' effects last infinitely far into the future
	- Each observation is dependent on the observation before
- In an MA model, the effect of previous periods only persist for $q$ periods 
	- Because each error is uncorrelated with previous errors


---

# Putting it Together

In order to account for all the problems that we might encounter in time series data, we can make use of ARIMA models.

**A**uto**R**egressive **I**ntegrated **M**oving **A**verage models allow us to
- Include lags of the dependent variable
- Take differences to eliminate trends
- Include lagged error terms


---

# The ARIMA Model

ARIMA models are often referred to as 
ARIMA($p,d,q$) models, where $p$, $d$, and $q$ are the parameters denoting the order of the autoregressive terms, integration terms, and moving average terms, respectively.
- It is often a matter of guessing and checking to find the correct specification for a model

---

# ARIMA in Python

```python
# Import needed libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
import plotly.express as px

# Read data, then set the index to be the date
# NOTE: make the file a single line!!
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/blob/master/
DataSets/pollutionBeijing.csv?raw=true")

data['datetime'] = pd.to_datetime(data['datetime'], 
	format='%Y-%m-%d %H:%M:%S')
data.set_index(pd.DatetimeIndex(data['datetime']), 
	inplace=True)
data['logpm'] = np.log(data['pm2.5'])
```
---

# ARIMA in Python
```python
# Plot the data
px.line(data, x='datetime', y='logpm',
       labels = {
           'datetime' : 'Date',
           'logpm' : 'Logged Pollution Level'
       })
```

---

### ARIMA in Python

<br>

![](nonStationary.png)


---

# ARIMA in Python
```python
# Plot the DIFFERENCED data

data['pmdiff'] = data['logpm'].diff()


px.line(data, x='datetime', y='pmdiff',
       labels = {
           'datetime' : 'Date',
           'pmdiff' : 'Differenced and Logged Pollution Level'
       })

```

---

# ARIMA in Python

![](stationary.png)


---

# Testing for Stationarity

We can use the **Augmented Dickey-Fuller Test** to determine whether or not our data is stationary.

- H$_0$: A unit root is present in our data
- H$_A$: The data is stationary

This can help us to determine whether or not differencing our data is required or sufficient for inducing stationarity.

---

# Testing for Stationarity

We can use the **Augmented Dickey-Fuller Test** to determine whether or not our data is stationary.

```python
>>> st.adfuller(
>>> 	data['pm2.5'][-250:], maxlag=12)

(-3.1576359480752445, # The test statistic
 0.022571607041567278, # The p-value
 2, # Number of AR lags in model
 247, # Number of obvservations
 {'1%': -3.4571053097263209, 
  '10%': -2.5730443824681606, # The 1%, 5%, and 10%
  '5%': -2.873313676101283},  # thresholds
 2272.5419900847974) # The model information criterion
```

In this case, we can reject the unit-root hypothesis!

---

# Fitting the ARIMA model

```python
import statsmodels.api as sm

model = sm.tsa.ARIMA(np.log(data["pm2.5"]), (1,1,0)) 
		  # specifying an ARIMA(1,1,0) model
reg = model.fit() # Fit the model using standard params
res = reg.resid   # store the residuals as res
```

Once we fit the ARIMA model using our selected specification, we can then explore the goodness of fit of the model using our model residuals (forecast errors). We will focus on this next week.


---

# Part 2 (Lecture 3) - Choosing your model

---


# Finding the Right Fit

- Time series models are unique in Econometrics: we will nearly always **visually** diagnose the proper specifications for our model
	- This takes practice
	- This takes repetition and iteration for any given model


---

# The Autocorrelation Function (ACF)

The ACF illustrates the correlation between a dependent variable and its lags.
- Choose how many lags to explore (based on nature of data)
- **Reminder**: correlations will vary between -1 and 1, with 1 being perfect correlation, and -1 being perfect inverse correlation
- Correlation can be cyclical!

---

### The Autocorrelation Function (ACF)

![w:600](rawACF.png)

---

# The Partial Autocorrelation Function

The PACF illustrates the correlation between a dependent variable and its lags, **after controlling for lower-order lags**.
- Choose how many lags to explore (based on nature of data)

---

# The Partial Autocorrelation Function (PACF)

![w:600](rawPACF.png)

---

# Building the Model

1. Make the series **stationary**
	- When the ACF falls "quickly" to zero at higher lags, the series is stationary
	- Can also use a **unit root test** to check for stationarity


---

# Building the Model

1. Make the series **stationary**
2. Use ACF and PACF plots to decide if you should include **AR** or **MA** terms in your model
	- Remember that we typically do not use both in the same model


---

# Building the Model

Signatures of **AR** and **MA** models:


**AR** Model: ACF dies out gradually, and the PACF cuts off sharply after a few lags

**MA** Model: ACF cuts off sharply, and PACF dies off more gradually (remember that **MA** models are based on previous *errors*)


---

# Building the Model

1. Make the series **stationary**
2. Use ACF and PACF plots to decide if you should include **AR** or **MA** terms in your model
3. Fit the model, and check residual ACF and PACF for lingering significance
4. If there are significant terms in residual ACF or PACF, add **AR** or **MA** terms, and try again


---


# ARIMA Diagnostics - ACF

```python
# Generate plot from ACF

from statsmodels.graphics.tsaplots import plot_acf

fig = plot_acf(data['pm2.5'], lags=10)
fig.show()
```

---

### ACF Plot

![w:600](rawACF.png)
This is a clear indication that we do NOT have stationary data (yet)

---



# ARIMA Diagnostics - PACF

```python
# Generate plot from PACF

fig = plot_pacf(data['pm2.5'], lags=10)
fig.show()
```

---

### PACF Plot

![w:600](rawPACF.png)

---


### Differenced ACF Plot

```py
plt = sm.graphics.tsa.plot_acf(data['pm2.5'].diff().dropna(), lags=30)
plt.show()
```

![w:600](differencedACF.png)
Differencing our data reduces the amount of structure that remains in the ACF.



---

# Time to Model!

Once we have 
- Utilized our ACF and PACF plots to diagnose our model
- Discovered the amount of differencing required by our data (to make our data stationary)

It is time to fit our model using the ```arima``` command we learned last week. 

We can then validate our model by examining the residual ACF and PACF plots.

---


# Residual ACF 

```python
import statsmodels.api as sm

arima = sm.tsa.ARIMA(data['pm2.5'], order=(1, 0, 0)).fit()

fig = plot_acf(arima.resid, lags=10)
fig.show()
```

---

# Residual ACF 

![w:600](residACF.png)

---

# Residual ACF 

```python
fig = plot_pacf(arima.resid, lags=10)
fig.show()
```

---

# Residual PACF 
Nearly identical to the ACF plot (and is very small, cyclical)

![w:600](residPACF.png)


---

# Looking Ahead

Now that we have a fitted model, we can start to make predictions

```python
fcst = reg.forecast(steps=10) # Generate forecast
```

We make our out-of-sample forecast, and store it as an object. It contains three arrays:
1) The forecast
2) The standard errors
3) Upper and lower confidence intervals

---

# Looking Ahead - Plotting

```python
fig = reg.plot_predict(start=len(data)-100, end=len(data)+100)
fig.show()
```

---

# Looking Ahead - Plotting


![w:600](forecastPlotARIMA.png)

---


# ARIMAX Models and Seasonal ARIMA models (SARIMAX)


---


# ARIMA + X


We can improve on the ARIMA model in many cases if we use ARIMA**X** (ARIMA with e**X**ogenous variables) models to include exogenous regressors in our estimations!

---

# ARIMAX

Let's use some weather data to get started:

```python
import pandas as pd
import numpy as np
import patsy as pt
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
import plotly.express as px

# NOTE: Make the string a single line again!
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/
blob/master/DataSets/omahaNOAA.csv?raw=true")[-(365*24):]
		# We are keeping only the last 365 days
```

---

# ARIMAX

```python
px.line(data, x='DATE', y='HOURLYDRYBULBTEMPF',
       labels = {
           'DATE' : 'Date',
           'HOURLYDRYBULBTEMPF' : 'Hourly Temperature (F)'
       })
```

---

# ARIMAX

![w:1000](temperatureRaw.png)

---

# ARIMAX

We have a lot of erroneous entries, and they're all recorded as 0!

```python
data['cleanTemp'] = data['HOURLYDRYBULBTEMPF'].replace(0, method='pad')

px.line(data, x='DATE', y='cleanTemp',
       labels = {
           'DATE' : 'Date',
           'cleanTemp' : 'Hourly Temperature (F)'
       })
```

---

# Much Better!

![w:1000](temperatureClean.png)

---

# ARIMAX


```python
eqn = "HOURLYDRYBULBTEMPF ~ HOURLYWindSpeed + " + 
"HOURLYStationPressure + HOURLYPrecip"
        
y, x = pt.dmatrices(eqn, data = data)

# The exog argument permits us to include exogenous vars
model = sm.tsa.ARIMA(y, order=(1,1,0), exog=x)
reg = model.fit(trend='nc', method='mle', 
		maxiter=500, solver='nm')
reg.summary()
```

---

![](summaryARIMAX.png)

---

# SARIMAX

Where can we go when we have cyclical data?
- We can introduce "seasonality" into our model

The Seasonal Autoregressive Integrated Moving Average Model with Exogenous Regressors (SARIMAX) is designed to deal with this kind of data and model.


---

# SARIMAX

We know that temperatures fluctuate daily (even though we have attempted to difference this out)

```python
model = sm.tsa.SARIMAX(y, order=(1,1,0),
		seasonal_order=(1,1,0,24), exog=x)
# trend='c' indicates that we want to include a 
#   constant/intercept term in our regression
reg = model.fit(trend='c', maxiter=500, solver='nm')
reg.summary()
```

Here, we need to include terms for our **seasonal** AR, I, and MA terms, as well as the periodicity of our data (24 observations per day).

---

![h:700](summarySARIMAX.png)

---

# Forecasting ARIMAX/SARIMAX

When we forecast based on models with exogenous variables, we need to include those variables as an argument to the forecast method.

```python
# Generating our Forecast
fcst = reg.forecast(steps=10, exog=x[-10:]) 
		     # Generate forecast
```

---

# Review

- We can use diagnostic plots to determine the order of our model, and to determine the processes involved (AR vs MA, etc.)
- ARIMAX allows for the use of exogenous variables to help explain our model
- SARIMAX adds seasonality to the model, allowing us to better account for cyclicality in our data.
