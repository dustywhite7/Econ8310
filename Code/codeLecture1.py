import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime

import patsy as pt
import seaborn as sns
import mpld3

sns.set_context("poster")


x = np.linspace(-1, 1, 101)
y = 2 * (x + np.random.rand(101))



sns.regplot(x,y, fit_reg=False)



xs = np.concatenate((np.ones(101).reshape(101,1),
		     x.reshape(101,1)), axis=1)
             
beta = np.linalg.solve(np.dot(xs.T, xs), np.dot(xs.T, y))

yhat = beta[0] + beta[1]*x



for a in [y, yhat]:
    sns.regplot(x, a, fit_reg=False)
    