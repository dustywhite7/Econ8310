# Assignment 3

This assignment will make use of the models covered in Weeks 9 to 11. Models include:

- Lasso Regression
- Decision Trees
- Random Forest

Your job will be to forecast whether or not an individual purchased a "meal" at Stedman's Café. You may need to restructure the data in order to make appropriate forecasts, but I have done significant preparation of the data to streamline the cleaning process. You will be graded based on the following:

- Your code executing without errors
- Storing your models to make predictions on new data
- Making reasonable predictions based on the data provided
- Forecasting a score for each team during each week

To complete this assignment, your code will need to contain the following:

1. A forecasting algorithm named "model" using the statsmodels or sklearn implementation of one of the models covered in weeks 9 to 11. Your dependent variable is the `meal` variable (indicating whether or not a purchase was classified as a meal or not), and may or may not use exogenous variables from the remainder of the dataset. The training data is stored at the following URL: https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv
2. A fitted model named "modelFit". This should be a model instance capable of generating forecasts using new data in the same shape as the data used in part (1).
3. A list or series of predictions using the data from the test period named "pred". Your data should simply contain binary values for whether or not you expect that an individual purchased a meal.
4. Make sure that your forecast includes predicted points for EACH observation! There are 1000 observations in the prediction timeframe. The data from which to generate predictions is stored at the following URL: https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv

Note: While all models from weeks 9 to 11 are available to you, they may not all be good fits for the data. I recommend considering the data carefully, then choosing models to try. See which models seem to perform best on this data, and implement the best performer for the final submission of the project.
