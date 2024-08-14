# Linear Regression

Let's learn about our first statistical modeling technique. Linear regression is sometimes considered the most basic machine learning algorithm. Strictly speaking, this is true; linear regression takes data as an input, and updates its rule set based on that data to model an outcome. This is the basis for machine learning! On the other hand, linear regression as a statistical tool definitely predates the machine learning movement, and has been used for over 200 years. It is well understood, and is readily accepted as a means of analysis.

Whether or not you think of it as machine learning, it is a great place to get started when learning about data modeling through statistics. Let's build up some intuition for how linear regression works before we learn to use it.

## Cause and Effect

The goal of statistical modeling is to understand the inputs that **cause** some specific outcome that we want to study. The big catch with statistical models is statistical models do not successfully identify **causation**. Statistical models instead identify **correlation**, and leave **causation** to domain expertise.

- **Correlation**: a mutual relationship or connection between two or more things. In statistics, we even use measures such as the correlation coefficient to describe the intensity of the relationship between two variables. In order for the relationship between two variables to be a **causal** relationship, there must first be **correlation** between those variables.

- **Causation**: the act of causing something. This is the relationship that statistical models want to measure! Unfortunately, statistics alone can't get us to **causation**. In order to establish a **causal** relationship, we must combine correlation between two variables with an explanation of *how* one of those variables can lead to changes in the other.

## Questioning Causality

When we hypothesize a causal relationship (that $x$ causes $y$), it is important to ask ourselves several questions:

1. Is it possible that $y$ causes $x$ instead?
2. Is it possible that $z$ (a new factor that we haven't considered before) is causing both $x$ and $y$?
3. Could the relationship have been observed by chance?

If we can demonstrate that each of these questions is unlikely to be answered in the affirmative, and we observe correlation between two variables, then we can begin to assert that the relationship may be causal.

## Establishing Causality

In order to establish causality, we need to meet several conditions:

- We can explain **why** $x$ causes $y$
- We can demonstrate that **nothing else is driving the changes** (within reason)
- We can show that there is a **correlation** between $x$ and $y$

In other words, we need a way to statistically isolate the relationship between two variables, even when there are other "moving parts" in our model.

## RCT

One way to establish causality is through Randomized controlled trials (RCTs). In the context of an RCT, the experiment is designed such that only one variable is experimented upon. By assigning random individuals (or entities, groups, etc.) to the treatment and control groups, the researcher can use univariate statistical tests to determine the relationship between the variable of interest (called the treatment variable) and the outcome (dependent variable).

Unfortunately, there are **many** contexts in which creating an RCT is not feasible. This may be due to the data collection method, it may be due to ethical concerns, or some other internal or external factor. Where we cannot perform an RCT, regression analysis becomes our next best option.

## Regression Analysis

If you have ever created a trend line in Excel or Tableau (or any similar software), then you have implemented a form of regression analysis. The beauty of regression analysis is in its ability to be **more** than just a trend line on a plot. While a trendline is valuable, it is only helpful when describing the relationship between two (and maybe three) variables, regression analysis makes it possible to create the analog to a trendline using **as many variables as you would like** (so long as you have more observations than variables).

The underlying concept (we won't go into the math here) is that we want to statistically separate the effect of each variable on the outcome. In other words, what happens to our outcome when we vary only one of our many variables at a time? All of the math behind regression analysis is designed to answer this question. This means that regression analysis

- Allows us to **act as if nothing else were changing**
- Mathematicaly isolates the effect of each individual **variable** on the outcome of interest
    - Variables are the factors that we want to include in our model
    
When we look at the output of a regression, then, we are looking at an equation that tells us how to add up the impacts of each variable to estimate the value of our dependent variable! Not only can we understand the impact of each individual variable, but we can also forecast the dependent variable for use in predictive modeling.


### Regression Terms

As we move forward, it will be helpful to keep in mind the following definitions:

- **Coefficient**: This is the effect of changing a variable by one unit (from “untreated” to “treated”)
- **Standard Error (Standard Deviation)**: Measures how noisy the effect of the independent variable is on the dependent variable
    - Larger numbers indicate more noise
- **Confidence Interval**: Assuming our regression analysis includes all relevant information, we expect that the true coefficient (treatment effect) lies within this range 95% of the time (for a 95% confidence interval)

- **Statistical Significance**: When the Average Treatment Effect has a confidence interval (at 95% or 99% levels, typically) that does not include 0



### Regression Assumptions

It is important to note that the statistical models underlying linear regression depend on several assumptions:

1. Effects are Linear (there are some workarounds)
2. Errors are normally distributed (bell-shaped)
3. Variables are not Collinear
4. No Autocorrelation (problematic for time series data)
5. Homoskedasticity (errors are shaped the same across all observations)

While the study of these assumptions is essential for a practitioner, and frequently occupies an entire semester-long course, it is sufficient for us to state these assumptions, and be aware of the fact that these assumptions are baked into the models. It is equally important to know that adaptations of regression analysis exist to work around violations of each assumption where needed.

Most regressions implemented in the real world must account for violations of one or more assumptions.

## When should we use regression, then?

Regression Analysis is most useful when you care about WHY a particular outcome occurs. Regressions are very powerful transparent models, by which I mean that it is straightforward to see how each variable leads to the predicted outcome. Whenever we want to establish causality (and can't put together an RCT), regression models are the *de facto* standard for understanding how one variable causes the other to change.

If, on the other hand, you want to just predict WHAT will happen next, there exist much better tools for you! We will spend many class sessions discussing these models during the remainder of this course.

## Implementing Linear Regression in Python


Now let's talk about actually **doing** linear regression. In order to perform regression analysis, we will utilize the `statsmodels` library, which is capable of performing most types of regression modeling. While the library is very robust, we will focus on running linear regression under standard assumptions during this class. Let's start by importing the library and some data.


```python
# Import our libraries

import statsmodels.formula.api as smf
import pandas as pd

# Import data to estimate the weight of fish sold at market
data = pd.read_csv("https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/fishWeight.csv")
```

Note that when we import `statsmodels`, we do so with a slightly different syntax (`import statsmodels.formula.api as smf`) rather than just importing the whole library. `statsmodels` provides the option to import two distinct APIs (application programming interfaces). We are using the formulas API, which will allow us to write intuitive regression equations that more easily permit us to modify our data as we run regressions.

Now, we really don't have much work left to do before we can run a regression! Let's look at our data really quickly, and then try out some regression analysis.


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
      <th>Species</th>
      <th>Weight</th>
      <th>Length1</th>
      <th>Length2</th>
      <th>Length3</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>23.2</td>
      <td>25.4</td>
      <td>30.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bream</td>
      <td>290.0</td>
      <td>24.0</td>
      <td>26.3</td>
      <td>31.2</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bream</td>
      <td>340.0</td>
      <td>23.9</td>
      <td>26.5</td>
      <td>31.1</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>363.0</td>
      <td>26.3</td>
      <td>29.0</td>
      <td>33.5</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bream</td>
      <td>430.0</td>
      <td>26.5</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>12.4440</td>
      <td>5.1340</td>
    </tr>
  </tbody>
</table>
</div>



As we can see above, we don't have too many variables. We have three measures of length (two are diagonal measures of the fish), we have height and width, then we have species, and finally, our dependent variable: weight. Let's see how well `Length1` predicts weight:


```python
reg = smf.ols("Weight ~ Length1", data=data)

reg = reg.fit()
```

Done! We have just implemented our first regression model! The function `smf.ols` is the function to implement OLS, or Ordinary Least Squares Regression. OLS is what is typically meant when someone says that they are going to use regression analysis, although other kinds of regressions certainly exist.

We provide two arguments to our regression function:
- A formula for the regression model
- The data set

Our formula always includes the dependent variable as the leftmost element, and is separated from independent variables by the `~` symbol. After we create our model, we have to use the `.fit()` method to calculate the optimal weights for our regression model.

Now, we can call the `.summary()` method on the fitted model in order to see how our model is structured and its anticipated performance:


```python
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Weight</td>      <th>  R-squared:         </th> <td>   0.839</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.837</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   815.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 09 Jun 2020</td> <th>  Prob (F-statistic):</th> <td>4.75e-64</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:09:35</td>     <th>  Log-Likelihood:    </th> <td> -1015.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   159</td>      <th>  AIC:               </th> <td>   2034.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   157</td>      <th>  BIC:               </th> <td>   2040.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> -462.3751</td> <td>   32.243</td> <td>  -14.340</td> <td> 0.000</td> <td> -526.061</td> <td> -398.690</td>
</tr>
<tr>
  <th>Length1</th>   <td>   32.7922</td> <td>    1.148</td> <td>   28.554</td> <td> 0.000</td> <td>   30.524</td> <td>   35.061</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.385</td> <th>  Durbin-Watson:     </th> <td>   0.369</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.009</td> <th>  Jarque-Bera (JB):  </th> <td>   9.768</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.489</td> <th>  Prob(JB):          </th> <td> 0.00757</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.721</td> <th>  Cond. No.          </th> <td>    79.2</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The most important features on this regression table are the `R-squared` and its adjusted form, as well as the table of coefficients and standard errors. In the above model, our $R^2$ is about .83, indicating that our model (including only an intercept term and the length of the fish) explains 83% of the variation in weight of the fish! That is pretty good! But I bet we can do better if we include more terms in our model.


```python
reg = smf.ols("Weight ~ Length1 + C(Species)", data=data)

reg = reg.fit()

reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Weight</td>      <th>  R-squared:         </th> <td>   0.930</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.927</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   286.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 09 Jun 2020</td> <th>  Prob (F-statistic):</th> <td>7.78e-84</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:12:28</td>     <th>  Log-Likelihood:    </th> <td> -948.61</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   159</td>      <th>  AIC:               </th> <td>   1913.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   151</td>      <th>  BIC:               </th> <td>   1938.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>               <td> -668.1044</td> <td>   40.472</td> <td>  -16.508</td> <td> 0.000</td> <td> -748.069</td> <td> -588.139</td>
</tr>
<tr>
  <th>C(Species)[T.Parkki]</th>    <td>   28.2864</td> <td>   36.336</td> <td>    0.778</td> <td> 0.438</td> <td>  -43.506</td> <td>  100.078</td>
</tr>
<tr>
  <th>C(Species)[T.Perch]</th>     <td>  -41.6749</td> <td>   21.598</td> <td>   -1.930</td> <td> 0.056</td> <td>  -84.349</td> <td>    0.999</td>
</tr>
<tr>
  <th>C(Species)[T.Pike]</th>      <td> -415.5526</td> <td>   32.256</td> <td>  -12.883</td> <td> 0.000</td> <td> -479.283</td> <td> -351.822</td>
</tr>
<tr>
  <th>C(Species)[T.Roach]</th>     <td>  -55.8549</td> <td>   29.596</td> <td>   -1.887</td> <td> 0.061</td> <td> -114.331</td> <td>    2.621</td>
</tr>
<tr>
  <th>C(Species)[T.Smelt]</th>     <td>  201.6195</td> <td>   38.457</td> <td>    5.243</td> <td> 0.000</td> <td>  125.637</td> <td>  277.602</td>
</tr>
<tr>
  <th>C(Species)[T.Whitefish]</th> <td>  -22.9381</td> <td>   42.825</td> <td>   -0.536</td> <td> 0.593</td> <td> -107.552</td> <td>   61.676</td>
</tr>
<tr>
  <th>Length1</th>                 <td>   42.4320</td> <td>    1.221</td> <td>   34.741</td> <td> 0.000</td> <td>   40.019</td> <td>   44.845</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>30.449</td> <th>  Durbin-Watson:     </th> <td>   0.862</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  55.264</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.910</td> <th>  Prob(JB):          </th> <td>9.99e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.242</td> <th>  Cond. No.          </th> <td>    230.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



When we add a new independent variable to our regression model, we separate each independent variable from the others using the `+` symbol. Additionally, we can use very simple syntax to include categorical variables by wrapping a variable name in the `C()` syntax. That variable is then automatically separated into separate binary variables for each class within the column!

Adding this single variable increased our $R^2$ from .83 to .93!

**Solve it [1 point]**:

Using the wage data provided [here](https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/wagePanelData.csv), create a linear regression model to explain and/or predict wages. Your fitted model should be stored as `reg`. If you do not name the model correctly, you won't get any points!

All code needed to implement your model should be contained in the cell below:



**Solve it [1 point]**:

In the cell below, explain why you settled on the model that you chose above. (Just type your explanation into the cell) 
- Why did you believe that the variables selected cause changes in wage? 
- How well does your model perform? 
- If you could collect more data, what data would you want?


# Logistic Regression

Linear regression is typically used to describe outcomes in which the value is continuous. This could be a model of the amount of profit generated through various business practices, the population density of a region, or any other metric that can be measured on a more or less continuous scale. Less frequently, linear regression is used to model outcomes that represent discrete outcomes such as success or failure, or to model the probability of success or failure. This is called a **linear probability model (LPM)**.

Why might LPMs not be used very often? Let's think about probability, as well as the properties of a linear regression model. A probability is a measure of likelihood, and is typically measured on the scale of $[0,1]$, or from 0% to 100% probability. A probability of 0 (0%) indicates that there is no likelihood that an event will take place. On the other hand, a probability of 1 (100%) would suggest absolute certainty that an outcome will occur. The important point, no matter which measurement of probability we choose to use, is that probabilities are **bounded** (cannot go beyond) absolute certainty of failure or absolute certainty of success.

Remember linear regression? One important part of any linear regression model is the **linearity** of the model. While this seems obvious, it is this part of the linear regression that gets us in trouble as we use LPMs. Any linear function with a nonzero slope will by definition be **unbounded**, and will NOT remain within the $[0, 1]$ interval. This means that an LPM will inevitably provide predictions that are non-sensical! Through our LPM, we will get some predictions that fit into each of the following categories:

- Totally normal predictions, with probabilities between 0 and 1
- Weird probabilities #1, with probability above 1 (or a likelihood of greater than 100%, which doesn't make sense!)
- Weird probabilities #2, with probability below 0 (or a negative likelihood, which again doesn't make sense!)

## Non-Linear, But in a Good Way

Is there a better way to model probabilities using regression analysis? You bet there is! It is called **logistic regression**, and we will learn how to do it, right here, right now. But first, let's explore why it is an improvement over LPMs. Remember that the big problem with LPMs is their linearity, right? It is the part of a regression that makes regression analysis so simple to understand, but also the part that breaks our probability model. We want to redesign our regression model to **resemble** a linear model, but stay within the $[0, 1]$ interval.

When we use a linear regression model, we end up with a series of coefficients. Those are the slope parameters for a linear equation resulting in our prediction of the dependent variable (outcome). We typically refer to each coefficient as a $\beta$ (beta) term, with subscripts ($\beta_i$) denoting which coefficient we are referring to. If we use the same subscripts to describe our $x$ variables, then we can write our regression equation (with $k$ variables) as follows:

$$ y = \beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 ... + \beta_k \cdot x_k$$

Obviously, as our $x$ values increase, our $y$ either increases or decreases, depending on the sign and magnitude of the associated $\beta$ coefficient. If those $x$'s get sufficiently large, so does $y$! In fact, it can go as high as $\infty$ and as low as $-\infty$. This is a problem now we are dealing with probability.

To fix our regression equation, we make a really simple transformation called the **logistic transformation**. After the transformation, our regression equation is written as follows:

$$ y = \frac{exp(\beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 ... + \beta_k \cdot x_k)}{1+ exp(\beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 ... + \beta_k \cdot x_k)} $$

where $exp()$ represents Euler's number raised to the power of the internal element (in our case, our original linear regression function).

Why do we choose this function? Several reasons:

- It is a simple transformation
- It leads to interpretable coefficients (remember that we want those!)
- It is bounded by $[0,1]$ like we want

How is it bounded? Remember that our linear regression function can go to either $\infty$ or $-\infty$. If the linear function takes the value $\infty$, then our logistic function becomes

$$ y = \frac{\infty}{1+\infty} \approx 1$$

because $exp(\infty)=\infty$. When the linear function takes the value $-\infty$, then $exp(-\infty)=0$, so our logistic function becomes

$$ y = \frac{0}{1} = 0 $$

and we remain within our probability bounds!

## Interpreting Coefficients

The confusing part of logistic regression stems from understanding the difference between coefficients in linear regression and in logistic regression. In linear regression, a coefficient describes the change in the dependent variable resulting from a one unit **increase** in the independent variable associated with that coefficient. If, for example, the coefficient of age on income (in Euros) is &#128;1,000, then an individual would be expected to earn &#128;1,000 more if he or she were 1 year older, or &#128;1,000 less if he or she were 1 year younger.

In a logistic regression, our coefficient is not a linear treatment effect. Instead, coefficients represent the **log odds ratio**, which can be written as

$$ \beta = ln\left(\frac{p}{1-p}\right) $$

If the **log odds ratio** is greater than 0, then a one unit increase in the associated variable would lead to an increased likelihood of success in the dependent variable. If the value is below 0, then the likelihood of success diminishes. The effects are not linear, but they do reflect the direction of the trend and can be interpreted to understand the relationship that each variable has with the outcome.

## Implementing Logistic Regression

Implementing logistic regression is nearly identical to the implementation of linear regression, thanks to the ease of use provided through the `statsmodels` library. Let's import some data and create a model:


```python
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv("https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/roomOccupancy.csv")

reg = smf.logit("Occupancy ~ Light + CO2", data=data).fit()

reg.summary()
```

    Optimization terminated successfully.
             Current function value: 0.067159
             Iterations 10





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Occupancy</td>    <th>  No. Observations:  </th>  <td>  8143</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  8140</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     2</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 10 Jun 2020</td> <th>  Pseudo R-squ.:     </th>  <td>0.8701</td> 
</tr>
<tr>
  <th>Time:</th>                <td>18:39:43</td>     <th>  Log-Likelihood:    </th> <td> -546.87</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -4210.2</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -9.2601</td> <td>    0.354</td> <td>  -26.156</td> <td> 0.000</td> <td>   -9.954</td> <td>   -8.566</td>
</tr>
<tr>
  <th>Light</th>     <td>    0.0184</td> <td>    0.001</td> <td>   25.309</td> <td> 0.000</td> <td>    0.017</td> <td>    0.020</td>
</tr>
<tr>
  <th>CO2</th>       <td>    0.0041</td> <td>    0.000</td> <td>   12.701</td> <td> 0.000</td> <td>    0.003</td> <td>    0.005</td>
</tr>
</table>



There are only two differences between this code and the code needed for a linear regression:

1. The dependent variable is binary (or exists within the range of 0 to 1)
2. We call the `smf.logit` function instead of `smf.ols`

We can use the same regression equation syntax to describe our regression model as we did with OLS, and we can use the same functions to fit our model. Do note, however, that the fitting process that occurs behind the scenes is different, and logistic regressions may take a significant amount of time to finish estimation if there are a large number of parameters and/or observations.

**Solve it [1 point]**:

For this assignment, you will perform a similar task to last class. Import the pass/fail data for students in Portugal found [here](https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/passFailTrain.csv), and create a logistic regression model using statsmodels that can estimate the likelihood of students passing or failing class. This information is contained in the cell called `G3`, which takes the value 1 when the student has a passing final grade, and 0 otherwise.

Call your fitted model `reg`, and place all necessary code to create and fit your model in the cell below:



**Solve it [1 point]**:

In the cell below, explain why you settled on the model that you chose above. (Just type your explanation into the cell)

- Why did you believe that the variables selected cause changes in wage?
- How well does your model perform?
- If you could collect more data, what data would you want?
