# AutoRegressive Integrated Moving Average Models
#### Or ARIMA, for short

## You know what they say about assumptions...

When we use OLS models, we are making five assumptions, which are often called the **standard assumptions**. These assumptions are required in order to mathematically derive the OLS regression solution. These rules are (according to *Econometric Analysis* by William Greene):

1) There must be a linear relationship between the dependent variable and any independent variable (that's why it's called **linear** regression!!)

2) Full rank (you can't recreate any $x$ variables through a linear combination of other $x$ variables)

3) Errors have mean zero. The error term for one observation $\epsilon_i$ cannot be predicted by $x$ variables. 

4) Every error term $\epsilon_i$ is unrelated to any other error term $\epsilon_j$, and each error term has the same variance $\sigma$. Violations of these combined conditions are called autocorrelation and heteroskedasticity, respectively. Autocorrelation, in particular, is crucial in a time-series context.

5) Errors are normally distributed. Probably the easiest assumption to violate. Kind of like a speeding ticket...

<br>

While these assumptions are important, they are NOT required in order to perform regression! In fact, they are often more important (and interesting) in the usefulness of their violations (and the solutions to those new models) than they are in a list like this one. It's just important that you know they exist before we go breaking them and making newer and more exciting models.

### More like guidelines than real rules

In many (most?) cases, we want to work with data that demonstrably violate some of these assumptions. That is fine! Just a few points, though...
- OLS is no longer guaranteed to model "truth" once the assumptions are violated (that seems bad...)
- There are models that have been developed to deal with nearly every possible way of violating these assumptions
- We will discuss those that are most relevant to forecasting (yay!)

<br>

Now that we know this, we just need to figure out whether time-series data (the kind of data we are going to focus on in the first half of this course) violates any of these rules. Knowing what is broken enables us to focus on finding a way to model our data that does not depend on the violated assumption. 

### So how about time-series data?

Let's start by understanding what time-series data actually is. Strictly speaking, time-series data is data focused on a **single variable**, and tracking the value of that variable over time. We also frequently call data time-series data if it is a collection of variables tracked over time. Let's take a look at some time-series data:


```python
import pandas as pd
import plotly.express as px

# The full data set is ~30 Mb so this might not be fast...
# Grab the last year of the data
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/omahaNOAA.csv")
# Clean it up
data = data.loc[len(data)-365*24:, ['DATE', 'HOURLYDRYBULBTEMPC']]
data.columns = ['date', 'temp_c']
data = data.loc[data['temp_c']!=0] # temp=0 is a 'missing value', which is annoying but fixable
data['date'] = pd.to_datetime(data['date'])
# And plot it
px.scatter(data, x='date', y='temp_c')


What is the first thing you notice about this plotted data? Does it help if we look at a smaller subset?


```python
px.scatter(data[-100:], x='date', y='temp_c')
```


What stands out to me about this data is that there is a **pattern** from one observation to the next. This might seem obvious, but it is a really important point about time-series data. Whenever our data is more than noise that has been sampled over time, our time-series data will have a pattern.

In math speak, we might say something like

$$ Corr(x_t, x_{t+1}) \neq 0 $$

This is a lot like what we assumed does NOT happen in our data under assumption (4) earlier! We can also write this relationship in a way that helps us to start understanding how we might implement regression with time-series data:

$$ y_{t} = \rho \cdot y_{t-1} + \epsilon_t $$

Where $\rho$ is the correlation between one observation and the next, and $\epsilon_t$ is the error or noise term.

Let's describe this in plain English. When we work with time-series data, we frequently observe that one observation is correlated with the next observation in the sequence. Because observations are correlated, our data is **not** independent and identically distributed, and therefore the standard assumptions of OLS **do not hold**. Without the standard assumptions OLS is no longer assured to represent our best approximation of truth. We can do better.

## Upgrading OLS for time-series

One of the best ways to account for violations of the standard assumptions is to eliminate the violation of the assumption from our data, and then use OLS. We are going to construct a new model that will do exactly that in order to deal with time-series data. This will enable us to take advantage of the interpretability of OLS models, while also using more interesting data to make forecasts.

### AutoRegressive Models

Our weather data is clearly correlated from one period to the next. The temperature in an hour is highly correlated with the temperature right now. The temperature tomorrow is also correlated (less strongly) with the current temperature. This suggests that the best way to describe our data-generating process is with the equation from above:

$$ y_{t} = \rho \cdot y_{t-1} + \epsilon_t $$

Which also **implicitly** mandates that there be correlation between the current period and **all past time periods**:

$$ y_{t} = \rho \cdot y_{t-1} + \epsilon_t = \rho \cdot (\rho \cdot y_{t-2}) + \epsilon_t $$
$$ = \rho \cdot (\rho \cdot (\rho \cdot y_{t-3})) + \epsilon_t = ... = \alpha + \rho^i \cdot y_{t-i} + \epsilon_t $$

Today's weather is correlated with the weather in every time period that has ever happened before.

The solution to this particular problem with our data is to use an __A__uto**R**egressive (AR) model. AR models are specified to contain a chosen number of lagged observations of $y$ as explanatory variables (they become our $x$ variables), and to use those lags to predict the next value of $y$ in the time-series.

By choosing the number of lags in an AR model, we are specifying how quickly we expect a time-series to return to its mean value. The fewer lagged terms we include, the quicker we expect the mean reversion to occur. The number of lagged observations is called the **order** of the model, and is denoted ($p$). When we describe models as AR($p$), we say that they are AutoRegressive models of order $p$:

$$ AR(p) \implies y_t = \alpha + \sum_{i=1}^p \rho_i \cdot y_{t-i} + \epsilon _t$$

In practice we allow $\rho_i$ to be estimated independently of all other $\rho$ values. We will wait to estimate AR models until we have a more complete picture of time series data.

### Moving Average Models

Brace yourselves. The Moving Average (MA) model may look almost exactly like an AR model, but its subtle differences can be very valuable additions to a time-series model.

AR models assume that the current value of $y$ is correlated with past values of $y$. In an MA model, $y_t$ is instead correlated with the **past error term** $\epsilon_{t-1}$. We can express this mathematically:

$$ y_t = \theta \cdot \epsilon_{t-1} + \epsilon_t $$

What is the difference? We know that $\epsilon_t$ is statistical noise. It represents the deviation of truth from our expectations in time $t$. We also know that it has an expected value of zero, so that our expectations should be correct **on average**. I $\epsilon$ is built from $y$, then how is this different from an AR model?

MA models derive their predictions of tomorrow from the errors of today. Because of this, our model **does not** incorporate persistent information from every previous period, like an AR model. The information about deviation from yesterday is sufficient.

Like an AR model, we can choose the **order** of our MA model by incorporating additional error terms from past periods into our model. In the case of the MA model, the order is denoted $q$.

$$ MA(q) \implies y_t = \alpha + \sum_{i=1}^q \theta_i \cdot \epsilon_{t-i} + \epsilon_t $$

Because errors are uncorrelated with one another, we will estimate each $\theta$ term independently in our model. We just need to discuss one more building block before we start building an actual model!

## Integration in time-series


One of the most pervasive problems in time-series data is a time **trend**. I mean, isn't that kind of what we are hoping to find: a pattern that explains the movement in the series? It turns out that a time-series with a time trend is what we call non-stationary, with stationarity being an important element of assumption 4. A non-stationary model is one with non-uniform mean or variance over time (ie - across observations).

The good news is that this is an easy problem to fix. We can remove the time-trend from a model by using differenced (integrated) models. The integration term ($d$) denotes the number of differences that must be taken before a series is considered stationary. Typically we will only consider the cases in which $d \in \{1,2\}$.

When $d=1$:

$$ \bar{y}_t = y_t - y_{t-1}$$

When $d=2$

$$ \bar{y}_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})$$

so that second degree integrated models resemble a difference-in-differences model or second derivative rather than simply a subtration of two previous periods.

# The ARIMA Model

As statisticians/economists/analysts, we put these three common time-series problems together to form one of the most-used time-series models around: the Auto Regressive Integrated Moving Average (ARIMA) model. 

An ARIMA model is said to have order $(p, d, q)$, representing the orders of the contained Autoregressive, Integration, and Moving Average parameters, respetively. We will be able to adjust these parameters as we design our model in order to optimize our ability to both forecase and conduct inferential analysis.

Let's use `statsmodels` to generate an ARIMA model!


```python
import statsmodels.api as sm

# Can't have missing data, but also don't want to drop hours, so we will
#   fill the data with last known temperature as our best guess of missing
#   data
data=data.fillna(method='pad')

# Implementing an ARIMA(1,0,0) model
arima = sm.tsa.ARIMA(data['temp_c'], order=(1, 0, 0)).fit()
arima.summary()
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:215: ValueWarning:
    
    An unsupported index was provided and will be ignored when e.g. forecasting.
    





<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>temp_c</td>      <th>  No. Observations:  </th>    <td>8303</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(1, 0)</td>    <th>  Log Likelihood     </th> <td>-12670.208</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>    <td>1.113</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 20 Aug 2021</td> <th>  AIC                </th>  <td>25346.415</td>
</tr>
<tr>
  <th>Time:</th>              <td>20:17:31</td>     <th>  BIC                </th>  <td>25367.488</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th>  <td>25353.615</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>    8.8094</td> <td>    2.131</td> <td>    4.134</td> <td> 0.000</td> <td>    4.633</td> <td>   12.986</td>
</tr>
<tr>
  <th>ar.L1.temp_c</th> <td>    0.9944</td> <td>    0.001</td> <td>  851.434</td> <td> 0.000</td> <td>    0.992</td> <td>    0.997</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.0056</td> <td>          +0.0000j</td> <td>           1.0056</td> <td>           0.0000</td>
</tr>
</table>



Just like in any `statsmodels` regression, we are able to quickly generate a summary table based on our regression model. We can experiment with different `order` parameters to determine the ideal model for our data:


```python
arima = sm.tsa.ARIMA(data['temp_c'], order=(3, 1, 0)).fit()
arima.summary()
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:215: ValueWarning:
    
    An unsupported index was provided and will be ignored when e.g. forecasting.
    
    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:215: ValueWarning:
    
    An unsupported index was provided and will be ignored when e.g. forecasting.
    





<table class="simpletable">
<caption>ARIMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>D.temp_c</td>     <th>  No. Observations:  </th>    <td>8302</td>   
</tr>
<tr>
  <th>Model:</th>          <td>ARIMA(3, 1, 0)</td>  <th>  Log Likelihood     </th> <td>-11726.716</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>    <td>0.994</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 20 Aug 2021</td> <th>  AIC                </th>  <td>23463.432</td>
</tr>
<tr>
  <th>Time:</th>              <td>20:26:49</td>     <th>  BIC                </th>  <td>23498.553</td>
</tr>
<tr>
  <th>Sample:</th>                <td>1</td>        <th>  HQIC               </th>  <td>23475.431</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>          <td>   -0.0034</td> <td>    0.025</td> <td>   -0.136</td> <td> 0.892</td> <td>   -0.052</td> <td>    0.045</td>
</tr>
<tr>
  <th>ar.L1.D.temp_c</th> <td>    0.2922</td> <td>    0.011</td> <td>   26.662</td> <td> 0.000</td> <td>    0.271</td> <td>    0.314</td>
</tr>
<tr>
  <th>ar.L2.D.temp_c</th> <td>    0.2168</td> <td>    0.011</td> <td>   19.406</td> <td> 0.000</td> <td>    0.195</td> <td>    0.239</td>
</tr>
<tr>
  <th>ar.L3.D.temp_c</th> <td>    0.0515</td> <td>    0.011</td> <td>    4.697</td> <td> 0.000</td> <td>    0.030</td> <td>    0.073</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.4204</td> <td>          -0.0000j</td> <td>           1.4204</td> <td>          -0.0000</td>
</tr>
<tr>
  <th>AR.2</th> <td>          -2.8152</td> <td>          -2.3972j</td> <td>           3.6976</td> <td>          -0.3877</td>
</tr>
<tr>
  <th>AR.3</th> <td>          -2.8152</td> <td>          +2.3972j</td> <td>           3.6976</td> <td>           0.3877</td>
</tr>
</table>



In addition to simply generating a regression summary table, we can also create forecasts using our ARIMA model, and can even plot those observations:


```python
from datetime import timedelta
import plotly.graph_objects as go

# Generate forecast of next 10 hours

fcast, sd, cint = arima.forecast(steps=10)

# Generate data frame based on forecast
times = [data.iloc[-1, 0] + timedelta(hours=i) for i in range(1, 11)]

forecast = pd.DataFrame([times, fcast]).T
forecast.columns = data.columns = ['date', 'temp_c']

# Plot forecast with original data

fig = px.line(data[-100:], x='date', y='temp_c')
fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["temp_c"], mode='markers', name='Forecast'))
fig.show()
```


We will learn more about diagnosing ARIMA models to choose the optimal parameters next lesson. We will also discuss the addition of exogenous factors to our model, and the inclusion of seasonality to create even more accurate models.
