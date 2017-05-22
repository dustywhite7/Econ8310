# Import Libraries
import pandas as import pd
import numpy as np
import statsmodels.formula.api as sm

# Import Data
data = pd.read_csv(
	'/home/dusty/DatasetsDA/firmInvestmentPanel.csv')

# We now need to de-mean our data
vars = ['I_','F_','C_']
for i in data.FIRM.unique():
    data.loc[data.FIRM==1, vars] = \
    	data.loc[data.FIRM==1, vars] -\
        data.loc[data.FIRM==1, vars].mean()


# Specify regression
reg = sm.ols("I_ ~ F_ + C_ + YEAR + I(YEAR**2) + C(FIRM)",
	data=data[data.YEAR<1954]) # Last year saved as
                                   # forecast
# Fit regression with clustered robust standard errors
# Note: We need to have same year restriction on groups
fit = reg.fit().get_robustcov_results(cov_type='cluster',
	groups=data.loc[data.YEAR<1954, 'FIRM'])
# Print results
print(fit.summary())

# Store predictions and truth
pred = fit.predict(data[data.YEAR==1954])
truth = data.loc[data.YEAR==1954, "I_"]
# Store errors
errors = pred - truth
# Calculate Absolute Percentage Error
pce = np.abs(errors/truth)*100
# Print MSE, Mean Absolute Error, and Mean Abs Percentage Error
print("Mean Squared Error: %s" % str(np.mean(errors**2)))
print("Mean Absolute Error: %s" % str(np.mean(np.abs(errors))))
print("Mean Absolute Percentage Error: %s" % str(np.mean(pce)))
