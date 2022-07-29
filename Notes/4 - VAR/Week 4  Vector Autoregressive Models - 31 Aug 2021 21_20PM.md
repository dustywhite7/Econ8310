# Vector Autoregressive Models (VAR)


When we work with time series, we are often working with multiple series. As we saw with our ARIMA models, we will frequently want to include one or more additional factors as regressors in our regression equation to try and explain the fluctuation in our dependent variable.

This works when the additional series are truly exogenous. Is this always the case, though? In many contexts, we may deal with multiple time series which contain feedback loops. While $x$ can be shown to drive changes in $y$, we also have to consider that $y$ may be driving changes in $x$ as well. In this case, both variables are endogenous, and we have an endogeneity problem in our model.

On one hand, this is bad because endogeneity is a violation of the OLS assumptions. On the other hand, this isn't TOO bad, because we have models that can rescue us from our data.

Before we get to the time series versions, though, we need to understand how endogenous regressors can be handled generally. We can then take some of our time series techniques and apply them to our new model.

## Seemingly Unrelated Regressions (SUR)

When we need to estimate two unknowns in a system of equations, we need two different equations (at least!). The solution to our regression problem, then, is to create a series of regression equations that can be estimated simultaneously as a **system of equations**, with the resulting model providing predictions for each dependent variable relative to the other(s).

We can write these equations as follows:

$$ Y_{j} = X_j\beta_j + \epsilon_j $$

where $Y_j$, and $\epsilon_j$ are $N \times 1$, $X_j$ is $N \times K$,  and $\beta_j$ is $K \times 1$. Additionally, there are $j$ regression equations. Note that there isn't really anything special about the equations, per se, the only new thing going on here is that there is **more than one** regression equation.

Additionally, our error terms are correlated, because our dependent variables are to some extent related to one another. Thus, inaccuracy in one model is correlated with inaccuracy in the other model(s). We can express this correlation mathematically:

$$ Cov(\epsilon_{ij}, \epsilon_{ik})=\sigma_{ij}$$
and 
$$ Cov(\epsilon_{ij}, \epsilon_{i'k})=0, \;\;\forall \;i \neq i'$$

In a nutshell, this just tells us that errors may be correlated from one dependent variable to another across regression equations **for the same observation**. We assume that correlation across observations is 0, so that our observations remain (in theory) independent and identically distributed.

At this point, we stack up all of our regression equations into a bunch of really big matrices:


$$ \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_{N}\end{bmatrix} = \begin{bmatrix} X & \mathbf{0} & ... & \mathbf{0}  \\ \mathbf{0} & X & ... & \mathbf{0}  \\ \vdots & \vdots & \ddots & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & X  \end{bmatrix} \begin{bmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_{N}\end{bmatrix} + \begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_{N}\end{bmatrix}$$

Where $Y_j$ is a vector of length $N$, and $X$ is an $N \times K$ matrix, so that the full array of $Y$s are $(N\cdot J) \times 1$, and the enlarged $X$ matrix is  $(N\cdot J) \times (K\cdot J) $

The FGLS (Feasible Generalized Least Squares) estimator of the system is 

$$ \hat{\beta}_{FGLS} = \left( X'\left(\hat{\Sigma} \otimes I_N\right)X  \right)^{-1} X'\left(\hat{\Sigma} \otimes I_N\right)Y $$

<br>

Where $\hat{\Sigma} = [\hat{\sigma}_{ij}]$, and

$$ \hat{\sigma}_{ij} = \frac{1}{N}\left(y_i - X_i\beta_i\right)'\left(y_j - X_j\beta_j\right) $$

Ignore that math, though. All we are doing is jointly conducting least squares regression on a group of regression equations while allowing for the fact that several of the variables are endogenously determined. The end result is essentially just a **collection** of regression tables, with one table for each of the specified dependent variables.

## Back to a Time Series

So what is all of this for? It turns out that one of the most common places to see dependent variables affecting one another is in a time series context. Think of the pollution data: temperature affects air pressure and also wind speed, but wind speed may affect temperature, and air pressure may also affect temperature. As one variable changes, so do the others. 

As we seek to predict one or more of these values, we also need to predict the others. To do this, we need to create a time series version of the SUR model. The simplest version of this model is called the **VAR model, or Vector Autoregressive model**. A small extension (including possible moving average terms) leads us to the **VARMA** model.

Let's talk about how these models work.

## VAR(MA) Models

Remember from our lessons on ARIMA models that most time series $y$ suffer from autocorrelation. This means that $y_t$ is correlated with $y_{t-1}$, and through it all other previous observations of the series. If we have a group of time series variables, then each of the variables are likely to have this same characteristic.

Thus, our elementary SUR model, where a single observation is correlated across dependent variables but not across observations is invalid. Our model now requires us to account for correlation both across dependent variables, as well as over time (ie across observations). What we end up with is the **VARMA** (Vector Autoregressive Moving Average) model. The best way to start describing a VARMA model is to simply write it out mathematically. Let's start by writing a VARMA(1, 0) or VAR(1) model, where there is a single AR term for each dependent variable:

$$ \mathbf{y}_{i,t} = \mathbf{\mu_i} + \mathbf{\Gamma}_{i,1} \mathbf{y}_{i,t-1} + \sum_{j=1}^{J}\mathbf{\Gamma}_{j,1} \mathbf{y}_{i, t-1} + \mathbf{\epsilon}_{i,t} $$ where $i \neq j$. That means that for any specific $y$ function, we consider its one-period lag, as well as the first lagged value of all other $y$'s as explanatory variables.

By creating this special case of the SUR model to account for the tendencies of time series data, we are able to allow endogeneity across variables, as well as autocorrelation across time. 

This approach will become even more impressive as we look at the models themselves, and at how they enable us to explore the impact of possible shocks to the system through our statistical model.

## Implementing a VARMA model

As we build out our statistical models, there is one important distinction between VARMA and ARIMA models that we should consider: an ARIMA model allows us to deal with non-stationary data **through** our model. A VARMA model provides no such mechanism. This is because a VARMA model deals with many endogenous variables simultaneously. Just because one variable is non-stationary does not mean that we should difference **every** dependent variable!

In practice, this means that we must check our variables for stationarity before implementing a VARMA model. If we find our time series to be non-stationary, then we must create a differenced time series prior to incorporating a variable into our model.

And now, to the code:


```python
# Getting started by importing modules and data

import pandas as pd, numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
import plotly.express as px
from datetime import datetime

# Collect data, set index

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/pollutionBeijing.csv")
                   
format = '%Y-%m-%d %H:%M:%S'
data['datetime'] = pd.to_datetime(data['datetime'], format=format)
data.set_index(pd.DatetimeIndex(data['datetime']), inplace=True)

# Select variables for VAR model
varData = data[['pm2.5','TEMP','PRES','Iws']].dropna()[:-50]
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm


Note that we do not use patsy when preparing data for our VARMA model. This is because we must pass an **array** of dependent variables to our model, and this can be more easily done by selecting pre-processed columns from our data frame. 

Additionally, we are dropping missing values and the last fifty observations (the last 50 will be used to evaluate our forecast).

With many variables, it becomes burdenson to visually diagnose each individual time series based on its visual characteristics using ACF and PACF plots. We will take this opportunity to use the built-in functionality of `statsmodels` to calculate the AIC- and BIC-indicated optimal orders for our data. In order to do this, we must assume that our model has no Moving Average components. We then use the `VAR` model's inbuilt functionality as follows:


```python
model = sm.tsa.VAR(varData) # define the model and data
print(model.select_order().summary()) # uses information criteria to select the model order
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:219: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)


     VAR Order Selection (* highlights the minimums)  
    ==================================================
           AIC         BIC         FPE         HQIC   
    --------------------------------------------------
    0        25.21       25.21   8.841e+10       25.21
    1        11.97       11.97   1.578e+05       11.97
    2        11.73       11.74   1.247e+05       11.74
    3        11.59       11.60   1.079e+05       11.59
    4        11.55       11.57   1.040e+05       11.56
    5        11.54       11.56   1.029e+05       11.55
    6        11.53       11.55   1.014e+05       11.53
    7        11.52       11.54   1.003e+05       11.52
    8        11.50       11.53   9.889e+04       11.51
    9        11.48       11.51   9.705e+04       11.49
    10       11.46       11.49   9.481e+04       11.47
    11       11.43       11.47   9.206e+04       11.44
    12       11.40       11.44   8.926e+04       11.41
    13       11.37       11.42   8.690e+04       11.39
    14       11.35       11.40   8.480e+04       11.36
    15       11.32       11.37   8.268e+04       11.34
    16       11.30       11.35   8.087e+04       11.32
    17       11.28       11.34   7.945e+04       11.30
    18       11.27       11.33   7.818e+04       11.29
    19       11.25       11.31   7.687e+04       11.27
    20       11.23       11.30   7.570e+04       11.26
    21       11.22       11.29   7.474e+04       11.24
    22       11.21       11.29   7.412e+04       11.24
    23       11.20       11.28   7.338e+04       11.23
    24       11.19       11.27   7.227e+04       11.21
    25       11.17       11.25   7.064e+04       11.19
    26       11.16       11.24   7.011e+04       11.19
    27       11.16       11.25   7.008e+04       11.19
    28       11.15       11.25   6.979e+04       11.18
    29       11.15       11.25   6.943e+04       11.18
    30       11.14      11.24*   6.911e+04       11.18
    31       11.14       11.25   6.904e+04       11.18
    32       11.14       11.25   6.907e+04       11.18
    33       11.14       11.25   6.908e+04       11.18
    34       11.14       11.26   6.908e+04       11.18
    35       11.14       11.26   6.909e+04       11.18
    36       11.14       11.26   6.908e+04       11.18
    37       11.14       11.27   6.903e+04       11.18
    38       11.14       11.27   6.902e+04       11.18
    39       11.14       11.27   6.901e+04       11.18
    40       11.14       11.27   6.895e+04       11.18
    41       11.14       11.28   6.889e+04       11.18
    42       11.14       11.28   6.883e+04       11.18
    43       11.14       11.28   6.878e+04       11.18
    44       11.14       11.28   6.866e+04       11.18
    45       11.14       11.29   6.854e+04       11.18
    46       11.13       11.29   6.845e+04       11.18
    47       11.13       11.29   6.840e+04       11.18
    48       11.13       11.29   6.820e+04       11.18
    49       11.12       11.29   6.772e+04       11.17
    50       11.12       11.29   6.760e+04      11.17*
    51       11.12       11.29   6.760e+04       11.18
    52       11.12       11.29   6.754e+04       11.18
    53       11.12       11.30   6.743e+04       11.17
    54      11.12*       11.30  6.741e+04*       11.18
    --------------------------------------------------


Given that three of the criteria suggest a lag order near 50, I will choose 48 lags (hourly data, so 2 days of lags), and fit my model accordingly:


```python
modelFit = model.fit(48) 
modelFit.summary()
```


Now, the tables are MASSIVE, because each table has 48 lags of each of the four variables! While this doesn't make for particularly useful inference based on reading tables, our model has been generated, and now we can create some pretty cool forecasts of future periods, and even conduct some exercises that will help us to understand how a shock to the system will impact our simultaneous estimation of these four variables.


```python
forecastData = data[['pm2.5','TEMP','PRES','Iws']].dropna()[-100:-50]

fcast = pd.DataFrame(modelFit.forecast(y = forecastData.values,steps=50), columns = ['pm2.5','TEMP','PRES','Iws'])
```

Now we can create a vector of "truth", and compare our data from the forecast with the observed data from those future time periods.


```python
truth = data[['pm2.5','TEMP','PRES','Iws']].dropna()[-50:]

# Create and format the figure
fig = px.line(x = truth.index, 
		y=[fcast['pm2.5'], truth['pm2.5']],
        title = 'Particulate Matter Forecast',
		labels = {
			'value' : 'Particulate Matter',
			'x' : 'Date',
			'variable' : 'Series'
		})

# Renaming the series
fig.data[0].name = "Forecast"
fig.data[1].name = "Truth"

# Render the plot
fig.show()
```



```python
# Create and format the figure
fig = px.line(x = truth.index, 
		y=[fcast['Iws'], truth['Iws']],
        title = 'Wind Speed Forecast',
		labels = {
			'value' : 'Wind Speed',
			'x' : 'Date',
			'variable' : 'Series'
		})

# Renaming the series
fig.data[0].name = "Forecast"
fig.data[1].name = "Truth"

# Render the plot
fig.show()
```

```python
# Create and format the figure
fig = px.line(x = truth.index, 
		y=[fcast['TEMP'], truth['TEMP']],
        title = 'Temperature Forecast',
		labels = {
			'value' : 'Temperature (C)',
			'x' : 'Date',
			'variable' : 'Series'
		})

# Renaming the series
fig.data[0].name = "Forecast"
fig.data[1].name = "Truth"

# Render the plot
fig.show()
```

```python
# Create and format the figure
fig = px.line(x = truth.index, 
		y=[fcast['PRES'], truth['PRES']],
        title = 'Air Pressure Forecast',
		labels = {
			'value' : 'Air Pressure',
			'x' : 'Date',
			'variable' : 'Series'
		})

# Renaming the series
fig.data[0].name = "Forecast"
fig.data[1].name = "Truth"

# Render the plot
fig.show()
```

Our VAR model does pretty well with this data! In the short term we can see that three of the four variables remain fairly close to their true values, while `pm2.5` deviates to a larger extent. Now that we have our forecasts in hand, let's add some exogenous factors to our model via the VARMAX model extension.

## VARMAX, or VARMA with Exogenous Variables

Just like the ARIMA, we can extend our VAR/VARMA models by adding exogenous variables into the mix. Using the VARMAX model, this is straightforward:


```python
# Prep our data
varData = data[['pm2.5','TEMP','PRES','Iws', 'DEWP']].dropna()[-500:-50]
exog = varData['DEWP']
varData.drop('DEWP', axis=1, inplace=True)

# Create and fit model
model = sm.tsa.VARMAX(endog = varData.values, exog=exog.values, order=(48,0)) # define the order here for VARMAX!
modelFit = model.fit() 
modelFit.summary()
```

Cool! Now we have all of the tools to build out VAR models of several flavors. With these tools in hand we can progress to impulse responses, and learning how to analyze the impact of a single shock to our system.

## Impulse Response Functions

What happens to temperature if it rains? What happens to wind speed? And pollution? When we utilize impulse response functions (IRFs) we can analyze the expected changes across our system that result from a single shock to one of our endogenous variables. The best way to learn is by doing, so let's make an IRF of increased rainfall and determine its effect on our system:


```python
# Recreate our VAR model (though we can also use VARMAX models) with rainfall as a variable
# Select variables for VAR model
varData = data[['pm2.5','TEMP','PRES','Iws', 'Ir']].dropna()[:-50]

model = sm.tsa.VAR(varData) # define the model and data
modelFit = model.fit(48) 

# Create the impulse response functions
irf = modelFit.irf(24)
plt = irf.plot(impulse="Ir")
plt.show()
```


![png](output_22_1.png)


I personally wasn't surprised by the IRFs for rainfall. My general impression over my life has been that rain tends to clear the air (`pm2.5` falls after rain), and the temperature declines. The IRF is intriguing in the sense that it tells us the marginal effect of rainfall at each point in time following the original shock (at `t=0`).

Can we see the overall impact of rainfall? We can! There is also a Cumulative Response Function (CRF), which aggregates the overall impact of the shock, so that we can see where the "new normal" resides. This kind of study is particularly valuable when exploring economic outcomes, since events such as a demand shock will often leave a market at a new equilibrium, and we can estimate that outcome using a CRF.


```python
# Create the cumulative response functions
plt = irf.plot_cum_effects(impulse="Ir")
plt.show()
```


![png](output_24_0.png)


**Reading Assignment**

In the cell below, describe a case in which you might use VAR models in your work/education. How would you create the model? What data would you need? What would you learn if you used an IRF or CRF based on your model?

