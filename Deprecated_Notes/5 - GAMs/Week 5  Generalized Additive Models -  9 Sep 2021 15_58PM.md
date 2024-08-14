# Generalized Additive Models (GAM)

While linear models are valuable, sometimes we need a bit more flexibility. We have a few options to overcome linearity constraints in our statistical models:

1. Adding higher order parameters to our linear model
2. Strictly non-parametric modeling
3. Semi-parametric models

## Higher Orders in OLS

When estimating the effect of age (or experience) on earnings for workers, we frequently discuss a pattern that is called the "earnings-experience profile". This pattern looks like the following chart: 

<div>
<img src="https://www.payscale.com/data/wp-content/uploads/sites/8/2019/05/Earnings-Growth-by-Gender-in-Management-Occupations.png" width="500"/>
</div>

It is clear that this graph does not demonstrate a linear relationship between age (a proxy for experience) and earnings. It looks much more like a quadratic equation to me. How can we incorporate that functional form into a linear model?

It turns out that OLS's assumptions state that the relationship between a paramter and the dependent variabe be linear, but not necessarily that the parameter itself be linear. What we can do to sidestep this problem is to simply **include non-linear terms in our regression model**. Let's look at a simple regression equation:

$$ y_i = \alpha + \beta \cdot x_i + \epsilon_i $$

The REQUIRED linearity stipulates that $\beta \cdot x_i$ be a linear relationship, or in other words that $\beta$ be a number (not some other type of functional form). If we want to consider a non-linear relationship between $y$ and $x$, then we can adjust our regression model like this:

$$ y_i = \alpha + \beta_1 \cdot x_{i} + \beta_2 \cdot x^2_{i} + \epsilon_i $$

Now, we still have a linear relationshipe between each parameter and our dependent variable ($y$), but we have created a second parameter based on our original $x$ variable: $x^2$. Thus, we can estimate the quadratic relationship between $x$ and $y$ using linear regression. The marginal effect of increasing $x$ by one unit in this model is no longer $\beta(x_{old}-x_{new})$. Now, it is $\beta_1(x_{old}-x_{new}) + \beta_2(x_{old}^2-x_{new}^2)$, and depends on the **current $x$ values**!


## Non-parametric Modeling

Another alternative is to simply abandon the parametric nature of OLS, and allow our model to assume **any** functional form. Thus, the relationship between $y$ and $x$ could be represented as

$$ y_i = f(x_i) + \epsilon_i $$

where $f()$ can be any functional form. We can then have ANY relationship between $x$ and $y$, and are not hindered by any restrictions or assumptions in our model.

Why don't we do this? I'll give you two reasons. First, it quickly becomes computationally intensive. The most common method for estimation of non-parametric models is through **kernel regression** (you can also use **regression trees** and several other tools, though). These models require large amounts of data in order to estimate, because each portion of the resulting model is in effect calculated based on the nearest observations. This also means that each individual point in the model needs to be separately estimated. While this is reasonable for simple regressions, it becomes complex and slow as the number of variables in a model increase.

Second, when we estimate an entirely non-parametric model, we are likely to generate interaction terms between variables. This means that the effects of one variable cannot be easily separated from the effects of other variables included in our model. If we have two $x$ parameters ($x_1, x_2$), our model can be written

$$ y_i = f(x_1, x_2) + \epsilon_i $$

and the marginal effects of these parameters given the model specified will be

$$ ME_1 = \frac{\partial f}{\partial x_1} $$
and

$$ ME_2 = \frac{\partial f}{\partial x_2} .$$

At this point, we have lost the ability to quickly interpret our marginal effects, and must generate estimates **at any specific observation** of the marginal effect of changes to inputs. In order to have a more flexible model, we have sacrificed both ease of calculation as well as ease of interpretability.

What we really need is some sort of middle ground, and this can be achieved through **Generalized Additive Models**.

## Generalized Additive Models (GAMs)

Generalized Additive Models, or GAMs, provide a nice middle ground between linear models and non-parametric modeling by opening a subclass of non-parametric models to analyze data, while also attempting to preserve interpretability. The restricted flexibility of GAMs is rooted in the restriction to **additively separable** functional forms.

What does that mean? Each explanatory variable must have an **additively separable** form in the regression model. To simplify even further, each variable can have ANY functional form so long as that form **does not interact with the functional forms of other variables**. In math speak, it's like this:

$$ y_i = \sum_{j=1}^K f_j(x_{j,i}) + \epsilon_i $$

If we had two variables again ($x_1, x_2$), then our model would be

$$ y_i = f_1(x_{1,i}) + f_2(x_{2,i}) + \epsilon_i. $$

Our functional form for each variable can take any shape, but must be totally separate for each $x$! This is what we call additively separable, because each function can be added together to become the overall estimate of $y$. Additionally, this makes for some really nice properties when interpreting the model.

- Each functional form can be separately visualized!
- Each functional form relates to only one variable, and so the slope estimates are much easier to calculate when estimating marginal effects.

Let's dive into creating GAMs in Python, and use that experience to learn a bit more about how they work.

## Using GAMs



### Facebook's `Prophet`

There are two libraries that we will cover as we explore the use of GAMs for forecasting. The first is the [`Prophet`](https://facebook.github.io/prophet/) library from Facebook. Using this library, we can create time series-focused GAMs.

`Prophet` expects our data to be structured in very specific ways. Let's import some data and take a look:


```python
# !pip install prophet # Only use this line if prophet is not already installed

import pandas as pd
from prophet import Prophet

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/chicagoBusRiders.csv")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>route</th>
      <th>date</th>
      <th>daytype</th>
      <th>rides</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>01/01/2001</td>
      <td>U</td>
      <td>7354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>01/01/2001</td>
      <td>U</td>
      <td>9288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>01/01/2001</td>
      <td>U</td>
      <td>6048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>01/01/2001</td>
      <td>U</td>
      <td>6309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>01/01/2001</td>
      <td>U</td>
      <td>11207</td>
    </tr>
  </tbody>
</table>
</div>



Now, looking at our `data.head()` values, we can see that there are several columns. `Prophet` wants data to be provided in two columns, named `ds` and `y`, respectively. `ds` should contain the timestamp, and `y` should contain the time series that we want to use.

To fit our data to this structure, let's pick a single bus route, and then reshape our data:


```python
data_p = data.loc[data['route']=='3', ['date', 'rides']]
data_p.columns = ['ds', 'y']

data_p.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/01/2001</td>
      <td>7354</td>
    </tr>
    <tr>
      <th>87</th>
      <td>01/02/2001</td>
      <td>16697</td>
    </tr>
    <tr>
      <th>216</th>
      <td>01/03/2001</td>
      <td>18195</td>
    </tr>
    <tr>
      <th>344</th>
      <td>01/04/2001</td>
      <td>19424</td>
    </tr>
    <tr>
      <th>472</th>
      <td>01/05/2001</td>
      <td>21221</td>
    </tr>
  </tbody>
</table>
</div>



At this point our data is ready to be used in a `Prophet` model.


```python
m = Prophet()
m.fit(data_p)
```


First, we created a `Prophet` model object, and then we used its `fit` method to fit our model to the data for route 3. This fitted model can then be used to create forecasts through which we explore the various seasonal patterns that we might expect to observe on route 3 over time.


```python
# Create an empty dataframe with dates for future periods
future = m.make_future_dataframe(periods=365)
# Fill in dataframe wtih forecasts of `y` for the future periods
forecast = m.predict(future)

forecast.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>19215.353695</td>
      <td>13249.199456</td>
      <td>19122.297441</td>
      <td>19215.353695</td>
      <td>19215.353695</td>
      <td>-3035.353849</td>
      <td>-3035.353849</td>
      <td>-3035.353849</td>
      <td>1451.914814</td>
      <td>1451.914814</td>
      <td>1451.914814</td>
      <td>-4487.268663</td>
      <td>-4487.268663</td>
      <td>-4487.268663</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16179.999846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-01-02</td>
      <td>19215.926422</td>
      <td>14836.148444</td>
      <td>20788.351595</td>
      <td>19215.926422</td>
      <td>19215.926422</td>
      <td>-1567.609337</td>
      <td>-1567.609337</td>
      <td>-1567.609337</td>
      <td>2832.429296</td>
      <td>2832.429296</td>
      <td>2832.429296</td>
      <td>-4400.038633</td>
      <td>-4400.038633</td>
      <td>-4400.038633</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17648.317085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-01-03</td>
      <td>19216.499149</td>
      <td>14523.227096</td>
      <td>20686.022151</td>
      <td>19216.499149</td>
      <td>19216.499149</td>
      <td>-1550.516253</td>
      <td>-1550.516253</td>
      <td>-1550.516253</td>
      <td>2729.388648</td>
      <td>2729.388648</td>
      <td>2729.388648</td>
      <td>-4279.904900</td>
      <td>-4279.904900</td>
      <td>-4279.904900</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17665.982896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-01-04</td>
      <td>19217.071875</td>
      <td>14496.121355</td>
      <td>20429.264514</td>
      <td>19217.071875</td>
      <td>19217.071875</td>
      <td>-1758.318527</td>
      <td>-1758.318527</td>
      <td>-1758.318527</td>
      <td>2371.290426</td>
      <td>2371.290426</td>
      <td>2371.290426</td>
      <td>-4129.608952</td>
      <td>-4129.608952</td>
      <td>-4129.608952</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17458.753349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-01-05</td>
      <td>19217.644602</td>
      <td>14952.612673</td>
      <td>21135.747844</td>
      <td>19217.644602</td>
      <td>19217.644602</td>
      <td>-1183.922585</td>
      <td>-1183.922585</td>
      <td>-1183.922585</td>
      <td>2768.476279</td>
      <td>2768.476279</td>
      <td>2768.476279</td>
      <td>-3952.398864</td>
      <td>-3952.398864</td>
      <td>-3952.398864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18033.722017</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot our model together with the forecast
fig = m.plot(forecast)
fig.show()
```


![png](output_12_0.png)



```python
# Plot the components of the forecast
fig = m.plot_components(forecast)
fig.show()
```


![png](output_13_0.png)


Using these figures, we can explore how our model performs in-sample, what projections look like for future periods, and how different types of cyclicality or seasonality apply to our current model. If we want to explore the raw forecast numbers for our model, we can just use our `forecast` DataFrame to extract that information:


```python
forecast[['ds','yhat']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>16179.999846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-01-02</td>
      <td>17648.317085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-01-03</td>
      <td>17665.982896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-01-04</td>
      <td>17458.753349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-01-05</td>
      <td>18033.722017</td>
    </tr>
  </tbody>
</table>
</div>



While useful for time series data, `Prophet` lacks the kind of flexibility that would allow us to easily incorporate both time series and exogenous factors into our model. Let's look at a more flexible implementation of the GAM model.

### `pyGAM`'s alternative implementation

My preferred GAM library is the `pyGAM` library. It allows for really neat and flexible solutions, using GAMs not only for time series analysis, but also for more general data analysis. Let's get an example going and take a look at why.


```python
# Import libraries/functions
from pygam import LinearGAM, s, f, l
import pandas as pd

# Read in data, select route 3
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/chicagoBusRiders.csv")
data_p = data.loc[data['route']=='3']
data_p['date'] = pd.to_datetime(data_p['date'])
data_p['year'] = data_p['date'].dt.year
data_p['month'] = data_p['date'].dt.month
data_p['day'] = data_p['date'].dt.weekday


```


```python
data_p.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>route</th>
      <th>date</th>
      <th>daytype</th>
      <th>rides</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>2001-01-01</td>
      <td>U</td>
      <td>7354</td>
      <td>2001</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>3</td>
      <td>2001-01-02</td>
      <td>W</td>
      <td>16697</td>
      <td>2001</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>216</th>
      <td>3</td>
      <td>2001-01-03</td>
      <td>W</td>
      <td>18195</td>
      <td>2001</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>344</th>
      <td>3</td>
      <td>2001-01-04</td>
      <td>W</td>
      <td>19424</td>
      <td>2001</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>472</th>
      <td>3</td>
      <td>2001-01-05</td>
      <td>W</td>
      <td>21221</td>
      <td>2001</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = data_p[['year', 'month', 'day']]
y = data_p['rides']
```

With `pyGAM`, we do need to do all of the data processing manually, so we are creating the parsed time data using `pandas`' built-in functions to create columns for year, month, and day of the week. Once we have our data processed, we can fit our model.


```python
gam = LinearGAM(s(0) + s(1) + f(2))
gam = gam.gridsearch(x.values, y)
```

    100% (11 of 11) |########################| Elapsed Time: 0:00:00 Time:  0:00:00


There are a few things going on here. First, we need to create our actual GAM model. This is done by creating an instance of the `LinearGAM` class. The argument that we provide to this class is similar to a regression equation. What we need to provide are the **functional forms** of the parameters to be included in our model.

When we imported `pyGAM`, we also imported `s`, `f`, and `l`. `s` stands for **spline**, and represents our most general functional form. Splines can be used to estimate functions of any shape, and so they are the key component of a GAM as enablers of highly flexible functional form.

`f` creates a factor-based functional form, and is useful where data is encoded as numbers which are categorial (rather than ordinal) in nature. This allows us to see the different level effects of each encoded factor in the column.

`l` creates a linear functional form. This can be useful where we expect the relationship between $x$ and $y$ by constraining the relationship to be linear rather than smooth. Remember that linear functional forms are by far the easiest to interpret, so we want to use them wherever they are representative of truth!

In our case, we want to see the spline-based relationship between year and rides and month and rides, but days are better treated as factors. This means that the most effective "regression equation" is `s(0) + f(1) + f(2)`, with 0, 1, and 2 denoting the respective columns in our `x` data array.

Now, let's make ourselves a nice plot of the results:


```python
# Import plotly tools to create a grid of subplots (figures) that work together
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

# Name each figure
titles = ['year', 'month', 'day']

# Create the subplots in a single-row grid
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=titles)
# Dictate the size of the figure, title, etc.
fig['layout'].update(height=500, width=1000, title='pyGAM', showlegend=False)

# Loop over the titles, and create the corresponding figures
for i, title in enumerate(titles):
    # Create the grid over which to estimate the effect of parameters
    XX = gam.generate_X_grid(term=i)
    # Calculate the value and 95% confidence intervals for each parameter
    # This will become the expected effect on the dependent variable for a given value of x
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    
    # Create the effect and confidence interval traces (there are 3 total)
    trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
    ci1 = go.Scatter(x = XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2 = go.Scatter(x = XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')

    # Add each of the three traces to the figure in the relevant grid position
    fig.append_trace(trace, 1, i+1)
    fig.append_trace(ci1, 1, i+1)
    fig.append_trace(ci2, 1, i+1)

#Plot the figure
py.iplot(fig)
```

This looks a lot more complicated than `Prophet`, and it is! The added complexity is due to increased flexibility. That flexibility becomes apparent when we want to add more than the same factors that we could use in Prophet. Let's try to model something different and see what happens.


```python
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/auto-mpg.csv")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>



In this case, our model has nothing to do with time-series analysis, but we will still be able to create a nice GAM. We can also create GAMs that combine time series patterns as well as other types of factors using `pyGAM`, and can use more varied data types as we do so.


```python
titles = ['modelYear', 'acceleration', 'weight', 'horsepower', 'displacement','cylinders']

x = data[titles]
y = data['mpg']

# Creating a factor for cylinders since it is not continuous like the other information
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
gam = gam.gridsearch(x.values, y)

# Create the subplots in a single-row grid
fig = tools.make_subplots(rows=2, cols=3, subplot_titles=titles)
# Dictate the size of the figure, title, etc.
fig['layout'].update(height=500, width=1000, title='pyGAM', showlegend=False)

# Loop over the titles, and create the corresponding figures
for i, title in enumerate(titles):
    # Create the grid over which to estimate the effect of parameters
    XX = gam.generate_X_grid(term=i)
    # Calculate the value and 95% confidence intervals for each parameter
    # This will become the expected effect on the dependent variable for a given value of x
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    
    # Create the effect and confidence interval traces (there are 3 total)
    trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
    ci1 = go.Scatter(x = XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2 = go.Scatter(x = XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')

    if i<3:
        fig.append_trace(trace, 1, i+1)
        fig.append_trace(ci1, 1, i+1)
        fig.append_trace(ci2, 1, i+1)
    else:
        fig.append_trace(trace, 2, i-2)
        fig.append_trace(ci1, 2, i-2)
        fig.append_trace(ci2, 2, i-2)

#Plot the figure
py.iplot(fig)
```


Even though it took some effort to set up the original model, now that we have template code, we can quickly adapt it to pretty much any data, and get some really interesting figures back. These figures are the basis for our ability to interpret GAMs. The biggest reason for visualization is the power it has to translate our data to intuitive results for non-experts.

We can, of course, also create forecasts using our `pyGAM` models:


```python
pred = gam.predict(data.loc[3, titles].values.reshape(1,6))

print("Truth was {0:.2f}, prediction was {1:.2f}, the model was off by {2:.2f}.".format(data.loc[3,'mpg'], float(pred), data.loc[3,'mpg']-float(pred)))
```
