# Lasso Regression Models - A First Predictor

Welcome to the part of class where we talk about predictive models. Like we did with time-series modeling, we will start with a model based on OLS. Why do we keep doing this? Because OLS is computationally inexpensive, easy to explain, and well-accepted across many disciplines.

To get started with our OLS derivative this time, let's think about one of the problems that inspired our Exponential Smoothing models: low-frequency data, or cases in which we have very few observations but need to be able to say something about future (or past!) observations. 

## Identification Requirements in Statistics

In order for a statistical model to be valid, it must be **identified**. Identification addresses our ability to mathematically determine the value of variables (unknowns). In order for a statistical model to be identified, we need the number of parameters in the equation ($y$ and all $x$ parameters) to be equal to or greater than the number of observations. We usually write this as

$$ N-1 \geq K $$

where $N$ is the number of observations and $K$ is the number of $x$ parameters in the model (the $-1$ comes from the $y$ variable which occupies a single degree of freedom).

Think about this in the same way that we think about systems of equations in algebra. For every unknown, we need an equation. In statistics, we need at least one degree of freedom per statistical unknown.

### Implications

The result of the identification constraint is that we cannot have large $K$ for small $N$. How can we determine the correct specification of a model where we are limited in our ability to include controls?

1. We can use our business understanding
2. We can use statistical constraints to limit $x$
3. Principal Component Anaysis (PCA) and its relatives

While solution (1) might be the superior option, a practitioner may not be able to winnow a model sufficiently to achieve identification without statistical inspiration. Ideally, no practitioner is so uninformed about the data source that they simply choose to use statistical constraints without considering the business implications of each variable included in the data source being used to predict outcomes. 

I personally tend to advocate against PCA in most contexts. In a statistical sense, it sounds awesome to be able to reduce the dimensionality of your data by extracting the truly orthogonal parameters from the variables in $x$. In reality, this makes it difficult to understand which variables are driving each outcome. This limitation prevents many types of business decision-making (choosing policies, allocating resources, etc.), and should be used with caution.

## Lasso as a Solution

Since learning the business context of every data set you might encounter in your career is well beyond our current scope (and I don't like PCA), let's explore a statistical model that helps us to choose $x$ subject to $N-1\geq K$. The model that we need is called a **Lasso regression** (short for least absolute shrinkage and selection operator, but who wants to say that ever again?).

The lasso regression is a simple extention of OLS. In previous work, we have utilized the algebraic solution to OLS. This solution is actually derived from the optimization of maximum likelihood estimation (MLE) under the standard assumptions of OLS. This is important right now because a lasso regression requires us to step back to the MLE problem. The original OLS log-likelihood function is

$$  ln(\theta|y, x) = -\frac{n}{2}ln(2\pi) -\frac{n}{2}ln(\sigma^2) - \frac{1}{2\sigma^2}(y-x\beta)'(y-x\beta)  $$

indicating that we can solve the problem by choosing betas (and $\sigma^2$) to minimize the sum of squared errors. It is also structured based on the assumption that errors in the model are normall distributed. 

In order to adapt our model to fit the problem of selecting the appropriate $x$ to accomodate our low $N$ value (and thus requirement for low $K$), we need to add an additional constraint parameter to our likelihood function:

$$ L(\theta)_{LASSO} = L(\theta)_{OLS} - \lambda ||\theta||_1 $$

$L(\theta)_{LASSO}$ is the log-likelihood function for the lasso model, and $L(\theta)_{OLS}$ is the log-likelihod function for OLS. The additional term in the model is a **penalty term**. The $||\theta||_1$ term is simply notation for the L1 norm, or Manhattan Distance. This is measured as the sum of the magnitude in each dimension of $\theta$. Theta ($\theta$) just represents our regression parameters (beta ($\beta$) and sigma ($\sigma$) together).

The last part of the penalty term is lambda ($\lambda$). As $\lambda$ increases, the penalty becomes more severe.

The net result of this penalty term is to penalize any model for having lots of coefficients that are not zero. For some models, the importance of the coefficients is sufficiently large to justify keeping many parameters active in the model (ie, having non-zero coefficients). In other models, additional parameters do not provide sufficient information to our model to justify their inclusion after the penalty is included in the model.

By incrementally increasing $\lambda$, we can force a model to reduce its dimensionality from $K$ to any number between $K$ and 0.

### Seeing the effect

Let's look at the graph below. It includes the magnitude of various parameters in a lasso regression (this is the y value in the figure), and increasing values of $\lambda$ on the x-axis.

![](https://github.com/dustywhite7/Econ8310/raw/master/SlidesCode/lassoProcess.png)

As the penalty term increases, each of the coefficients is driven to 0. The first variables to reach zero are those with the smallest effects on the outcome. Slowly, however, each variable is pushed to 0. By looking at a graph like this (or the numbers behind it), we can choose $\lambda$ specific to our model and constraints. This enables us to have the exact number of coefficients in our model that we are willing to tolerate from an identification (or just a complexity) perspective.

## Implementing Lasso Regressions in Python

Let's create a model. Lasso models function nearly identically to OLS. We will use the `sklearn` implementation, although `statsmodels` also includes lasso implementations for OLS. As we use the `sklearn` implementations of these models, you can start to get familiar with `sklearn` as we prepare to use it for the remainder of the course.


```python
import pandas as pd
from sklearn.linear_model import Lasso
import patsy as pt

#Import randy johnson pitch data
data = pd.read_csv("https://github.com/dustywhite7/Econ4350/blob/master/Data/randy_johnson_08_09.csv?raw=true")

# Create regression arrays
y, x = pt.dmatrices("release_speed ~ pitch_name + balls + strikes + home_score + away_score", data = data)

# Create an OLS Model with alpha term (the penalty term)
model = Lasso(alpha = 1/100).fit(x, y)

# Print variable and coefficient table
print("{0} {1}".format('Variable'.ljust(30), 'Coefficient'))
print("{0}".format('-'*50))
for i, j in zip(x.design_info.column_names, model.coef_):
    print("{0} {1}".format(i.ljust(30), j))
```

To see what happens when you change the penalty term, simply adjust the `alpha` value in the code above. A fraction that is closer to 0 will drive more coefficients toward 0, while a smaller penalty will loosen the model constraints and allow for more non-zero coefficients.

To illustrate this point, let's create a Data Frame of coefficients as we vary the `alpha` level:


```python
import plotly.express as px

data = []
for alpha in [i/200 for i in range(201)]:
    model = Lasso(alpha = alpha).fit(x, y)
    row = list(model.coef_) + [alpha]
    data.append(row)
    
plot_data = pd.DataFrame(data, columns = x.design_info.column_names + ['alpha']).melt(id_vars='alpha')

px.line(plot_data, x='alpha', y='value', color='variable', 
        labels = {
                    'alpha':'Penalty Term',
                    'value':'Coefficient'
                },
       title="Changing Lasso Penalty")
```

As `alpha` increases, we see various coefficients fall to 0. The first variables to disappear are pitch types with very low frequency (in other words, unimportant variables in the scheme of predicting pitch velocity). Eventually, though, even the more important variables fall to 0.

## Lasso Regressions with Binary Outcomes

`sklearn` also enables us to utilize Lasso regression with binary outcomes, by including a penalty term in the likelihood estimator for logistic regression. The code change is trivial to implement:


```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
import patsy as pt

#Import car insurance data
data = pd.read_csv("https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/carInsuranceTrain.csv")
```


```python
data.columns
```




    Index(['Id', 'Age', 'Job', 'Marital', 'Education', 'Default', 'Balance',
           'HHInsurance', 'CarLoan', 'Communication', 'LastContactDay',
           'LastContactMonth', 'NoOfContacts', 'DaysPassed', 'PrevAttempts',
           'Outcome', 'CallStart', 'CallEnd', 'CarInsurance'],
          dtype='object')




```python
import numpy as np

# Create regression arrays
y, x = pt.dmatrices("CarInsurance ~ Age + Job + Marital + Education + Balance + HHInsurance + CarLoan", data = data)

# Create an OLS Model with L1 penalty (lasso) and C term (the inverse penalty term)
model = LogisticRegression(penalty='l1', C =1/10).fit(x, np.ravel(y))

# Print variable and coefficient table
print("{0} {1}".format('Variable'.ljust(30), 'Coefficient'))
print("{0}".format('-'*50))
for i, j in zip(x.design_info.column_names, np.ravel(model.coef_)):
    print("{0} {1}".format(i.ljust(30), j))
```

    Variable                       Coefficient
    --------------------------------------------------
    Intercept                      0.0
    Job[T.blue-collar]             -0.1434961688350271
    Job[T.entrepreneur]            -0.051163186164107186
    Job[T.housemaid]               0.0
    Job[T.management]              0.0
    Job[T.retired]                 0.46013798527652344
    Job[T.self-employed]           0.0
    Job[T.services]                0.0
    Job[T.student]                 0.2877032182612834
    Job[T.technician]              0.0
    Job[T.unemployed]              0.29609059576121766
    Marital[T.married]             -0.3000661213101173
    Marital[T.single]              0.03868109974417657
    Education[T.secondary]         0.0
    Education[T.tertiary]          0.24902305217755524
    Age                            0.00010231159665363767
    Balance                        5.129428618981847e-06
    HHInsurance                    -0.6812722588964261
    CarLoan                        -0.3122834971367438


    /opt/conda/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning:
    
    Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
    


Two notes on the Logistic Regression version of lasso:

1. You must specify that the `penalty` is `'l1'`, or you won't actually run a lasso regression
2. The penalty Term (`C`) is an inverse penalty term. Instead of higher values being more restrictive, lower values are more restrictive

Experiment by changing the C value to see when different coefficients become 0, or by plotting the movement of the variables like we did above for the OLS lasso example.

