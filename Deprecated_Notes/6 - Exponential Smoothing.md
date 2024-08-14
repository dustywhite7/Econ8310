# Exponential Smoothing

Most of the material in this notebook is adapted from [Forecasting: Principles and Practice](https://otexts.com/fpp2/expsmooth.html)

How do you handle forecasts when you just don't have that much information? Why do so many forecasting methods lag behind the time series data that they seek to predict? If you want the solution to these problems, as well as a model that is often chosen by practitioners over our other time series analysis tools, then you want **exponential smoothing** models!

## Simple and Smooth

Let's start off with a simple time series: $ y_1, y_2, ..., y_{t-1}, y_t $. If we want to predict $y_{t+1}$, then we need to choose some method of using the information that we have about previous periods to make our best guess at the following period. In this case, we are not going to use any parametric assumptions, we are simply going to choose weights for each observation, and then create a predicted outcome based on those weights.

### The Naive Model

The simplest weighting scheme is to assign all importance to the first observation, and no importance to any other observation in the time series. This model, called the **naive model**, will simply use the most recent observation of a time series as the prediction for the next event. You can write it like this:

$$ y_{t+1} = y_t + \epsilon_{t+1} $$

Probably not a great model, since there is almost certainly information to be gained by utilizing more than a single observation to predict our next step. This model does have the advantage of being available at the start of data collection, however. We only need a single past observation to begin making **some** forecast!

### Just make an average

Another weighting scheme we might choose is a simple average. Every past observation is equally weighted, and the average value of past periods is our predicted value for the subsequent period.

$$ y_{t+1} = \frac{1}{t}\sum_{i=1}^t y_i$$

In some cases, this might be a useful predictor, but for many time series, this simply isn't enough. We need to adjust the **importance** of our observations to match their relevance for the next time period.

### Exponential Decay

One simple way to create a model wherein each observation has less importance as we move further into the past is to use an **exponential decay** function to weight our observations. Like our average model from above, the overall weight of all observations will still sum to 1 (meaning that this is just a weighted average model), with the most recent observations being vastly (exponentially??) more important to our prediction than the earliest observations. The equation representing this kind of weighted average looks like this:

$$ y_{t+1} = \alpha \cdot y_t + \alpha \cdot(1-\alpha) \cdot y_{t-1} + \alpha \cdot(1-\alpha)^2 \cdot y_{t-2} + ... + \alpha \cdot(1-\alpha)^n \cdot y_{t-n} $$


A simple table (taken from [here](https://otexts.com/fpp2/ses.html)) reflecting the weights that would be assigned to various observations using this kind of weighting scheme helps to clarify a bit.

|    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       | $\alpha=0.2$  | $\alpha=0.4$  | $\alpha=0.6$  | $\alpha=0.8$  |
| --- | --- | --- | --- | --- |
| $y_t$       | 0.2000 | 0.4000 | 0.6000 | 0.8000 |
| $y_{t−1}$      | 0.1600 | 0.2400 | 0.2400 | 0.1600 |
| $y_{t−2}$      | 0.1280 | 0.1440 | 0.0960 | 0.0320 |
| $y_{t−3}$      | 0.1024 | 0.0864 | 0.0384 | 0.0064 |
| $y_{t−4}$      | 0.0819 | 0.0518 | 0.0154 | 0.0013 |
| $y_{t−5}$ | &nbsp;&nbsp;&nbsp;&nbsp;0.0655 | &nbsp;&nbsp;&nbsp;&nbsp;0.0311 | &nbsp;&nbsp;&nbsp;&nbsp;0.0061 | &nbsp;&nbsp;&nbsp;&nbsp;0.0003 |

What is $\alpha$? It's called our **smoothing parameter**, and it dictates the speed at which our weights fall over time. For very high $\alpha$ values, the weight is primarily placed on more recent observations, and for low $\alpha$ values, the weight is more evenly spread, though more recent values still receive greater "attention" than observations from the more distant past. $\alpha$ values need to be between 0 and 1, with 0 denoting the average model, and 1 the naive model.

Another way to write the model is to break it into its components. In this case, we have a simple model with only two components: the level ($l_t$) and forecast components ($\hat{y}_{t+1}$).

| Component | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Term |
| --- | --- |
| Forecast | $ \hat{y}_{t+1} = l_t $ |
| Level | $ l_t = \alpha \cdot y_t + 
(1-\alpha) \cdot l_{t-1}$|

Essentially, in our basic exponential smoothing model, we calculate the average level of the time series, and use this to make our prediction of the next period (or periods). Because we are already using all information that we believe is relevant to the problem, the level does not change as we move further into the future. This means that predictions of **any** future period will be the same as the prediction for $t+1$, until we accumulate further observations.

### Improving the forecast with trends

We can improve on our preliminary exponential smoothing model by incorporating trend information. Trend information is a simple estimate of the most recent direction and magnitude of movement, and can be readily incorporated into our smoothing model through a third component included in our model. Our first option is to simply include a linear trend in the model. The linear trend has the advantage of simple implementation and ease of interpretation.

| Component | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Term |
| --- | --- |
| Forecast | $ \hat{y}_{t+h} = l_t + h\cdot b_t$ |
| Level | $ l_t = \alpha \cdot y_t + (1-\alpha) \cdot (l_{t-1} + b_{t-1})$|
| Trend | $ b_t = \beta \cdot (l_t - l_{t-1}) + (1-\beta) \cdot b_{t-1} $|

Thus our model with linear trend is the same as our model from before, with the inclusion of a trend component. The trend component is the weighted average of the difference between the current and previous level, and the slope term from the previous period. We smooth the trend, just like we smooth the level of our model. This allows us to adjust for the most recent data, but also to include information from previous periods, since those also contain information about possible future values of our time series.

A linear trend is unlikely to persist far into the future. Many exponential smoothing models accomodate the unlikely progression of linear trends through **damping** techniques. Essentially, the model has a built-in decay function to dampen the effect of the trend as we look further into the future. The equations would be adapted as follows:

| Component | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Term |
| --- | --- |
| Forecast | $ \hat{y}_{t+h} = l_t + (\phi + \phi^2 + ... + \phi^h) \cdot b_t$ |
| Level | $ l_t = \alpha \cdot y_t + (1-\alpha) \cdot (l_{t-1} + \phi \cdot b_{t-1})$|
| Trend | $ b_t = \beta \cdot (l_t - l_{t-1}) + (1-\beta) \cdot  \phi \cdot b_{t-1} $|

These updated trend equations incorporate the decay term ($\phi$), which denotes the speed with which the linear trend decays to 0 in future observations.

### Seasonality and exponential smoothing

Our last addition to the exponential smoothing model (at least in this course) is to incorporate seasonality. In order to do so, we will incorporate one more term into our model. At the same time we also have to provide information on the number of observations observed per seasonal cycle. If our data is quarterly, and seasons happen over the course of a single year, then we would say that our seasonal term ($m$) is 4 (or four observations). If we have data that displays daily seasonality with hourly observations, then $m=24$.

| Component | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Term |
| --- | --- |
| Forecast | $ \hat{y}_{t+h} = l_t + (\phi + \phi^2 + ... + \phi^h) \cdot b_t + s_{t+h-m(k+1)}$ |
| Level | $ l_t = \alpha \cdot (y_t - s_{t-m}) + (1-\alpha) \cdot (l_{t-1} + \phi \cdot b_{t-1})$|
| Trend | $ b_t = \beta \cdot (l_t - l_{t-1}) + (1-\beta) \cdot  \phi \cdot b_{t-1} $|
| Seasonality | $s_t = \gamma \cdot (y_t - l_{t-1} - \phi \cdot b_{t-1}) + (1-\gamma) \cdot s_{t-m} $ | 

In essence, we incorporate a smoothed term that accounts for behavior at the same point in the **previous cycle**. Thus, we can account for the position in the current cycle, as well as the behavior within our current cycle, and any recent fluctuations that we have observed from one period to the next.

This is a **lot** of math speak. Let's implement these models, and see if we can gain more intuition about their function.

## Implementing Exponential Smoothing

Let's start by importing some US economic data. We will focus on forecasting non-farm payroll figures over time.


```python
import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/RecessionForecasting.csv")
data['DATE'] = pd.to_datetime(data['DATE'])

px.line(data, x="DATE", y='CivEmpLevel')
```


Once we have our data imported, we can start to implement the simple exponential smoothing model we described first, which will just be a weighted average, with a level term doing all the work of creating our forecasts. Thus, our forecasts should be totally flat.


```python
employment = data['CivEmpLevel']
employment.index = data['DATE']
employment.index.freq = employment.index.inferred_freq

alpha020 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.2,
                                        optimized=False)

alpha050 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.5,
                                        optimized=False)

alpha080 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.8,
                                        optimized=False)

forecast020 = alpha020.forecast(3)
forecast050 = alpha050.forecast(3)
forecast080 = alpha080.forecast(3)
```


```python
import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([employment.values, alpha020.fittedvalues.values,  alpha050.fittedvalues.values,  alpha080.fittedvalues.values]).T
smoothData.columns = ['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8']
smoothData.index = employment.index

fig = px.line(smoothData, y = ['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'alpha=0.2': 'red', 
                            'alpha=0.5':'green', 
                            'alpha=0.8':'purple'}
       )

fig.update_xaxes(range=[smoothData.index[-50], forecast020.index[-1]])
fig.update_yaxes(range=[142000, 153000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast020.index, y = forecast020.values, name='Forecast alpha=0.2', line={'color':'red'}))
fig.add_trace(go.Scatter(x=forecast050.index, y = forecast050.values, name='Forecast alpha=0.5', line={'color':'green'}))
fig.add_trace(go.Scatter(x=forecast080.index, y = forecast080.values, name='Forecast alpha=0.8', line={'color':'purple'}))
```

Above, we fitted the model with three different levels of smoothing, so that we can see the difference as we choose varied levels of $\alpha$. If we want to simply choose the **best** smoothing parameter given the data that we can observe, we can simply choose to optimzie our model instead:


```python
# Streamlined Modeling

alphaBest = SimpleExpSmoothing(employment).fit()
forecast = alphaBest.forecast(3)
```


```python
import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([employment.values, alphaBest.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Best Fit Model']
smoothData.index = employment.index

fig = px.line(smoothData, y = ['Truth', 'Best Fit Model'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Best Fit Model': 'red'}
       )

fig.update_xaxes(range=[smoothData.index[-50], forecast.index[-1]])
fig.update_yaxes(range=[142000, 153000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast.index, y = forecast.values, name='Forecast', line={'color':'red'}))
```

Most of the time, though, we don't want just flat forecasts. Let's create a model with a trend component, now!


```python
# Linear trend
trend = ExponentialSmoothing(employment, trend='add').fit()
# Linear trend with damping
dampedTrend = ExponentialSmoothing(employment, trend='add', damped=True).fit(use_boxcox=True,use_brute=True)

forecast_t = trend.forecast(10)
forecast_dt = dampedTrend.forecast(10)
```


```python
import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([employment.values, trend.fittedvalues.values, dampedTrend.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Trend', 'Damped Trend']
smoothData.index = employment.index

fig = px.line(smoothData, y = ['Truth', 'Trend', 'Damped Trend'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Trend': 'red',
                            'Damped Trend': 'green'
                           },
              title='Linear and Damped Trends'
       )

fig.update_xaxes(range=[smoothData.index[-50], forecast_t.index[-1]])
fig.update_yaxes(range=[142000, 154000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast_t.index, y = forecast_t.values, name='Forecast Trend', line={'color':'red'}))
fig.add_trace(go.Scatter(x=forecast_dt.index, y = forecast_dt.values, name='Forecast Damped Trend', line={'color':'green'}))
```

Our model might also benefit from seasonal adjustments, which we can also easily incorporate into our model:


```python
# Linear trend
trend = ExponentialSmoothing(employment, trend='add', seasonal='add').fit()
# Linear trend with damping
dampedTrend = ExponentialSmoothing(employment, trend='mul', seasonal='add', damped=True).fit(use_boxcox=True,use_brute=True)

forecast_t = trend.forecast(10)
forecast_dt = dampedTrend.forecast(10)
```


```python
import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([employment.values, trend.fittedvalues.values, dampedTrend.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Trend', 'Damped Trend']
smoothData.index = employment.index

fig = px.line(smoothData, y = ['Truth', 'Trend', 'Damped Trend'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Trend': 'red',
                            'Damped Trend': 'green'
                           },
              title='With Seasonality'
       )

fig.update_xaxes(range=[smoothData.index[-50], forecast_t.index[-1]])
fig.update_yaxes(range=[142000, 157000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast_t.index, y = forecast_t.values, name='Forecast Trend', line={'color':'red'}))
fig.add_trace(go.Scatter(x=forecast_dt.index, y = forecast_dt.values, name='Forecast Damped Trend', line={'color':'green'}))
```

You'll notice (or I'll just point it out to help you notice) that most of the code above is formatting for the plots. It is actually very simple to create our Exponential Smoothing models, and even easier to forecast with them. 

As one of the most powerful tools in a practitioner's toolkit, it is great to have such a straightforward model with such simple implementation.

