# Assignment 1

This assignment will make use of the models covered in Weeks 1 to 3. Models include:

- Ordinary Least Squares (OLS) models
- AutoRegressive Integrated Moving Average (ARIMA) models
- ARIMA models with exogenous variables
- Seasonal ARIMA models

Your job will be to forecast the average price of honey (in the column `avg_price_pound`) from 2016 to 2020. The necessary data is available on GitHub: https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment1/honey_production_data_2015.csv. There is also a data dictionary: https://github.com/dustywhite7/Econ8310/blob/master/AssignmentData/Assignment1/honey_production_description.docx?raw=true

Your grade will be assigned based on the performance of your code, and will be based primarily on:

- Your code executing without errors
- Storing your models to make predictions on new data
- Making reasonable predictions based on the data provided
- The data is available at 

To complete this assignment, your code will need to contain the following:

- A valid model from among the model options. One OLS model, one ARIMA model, one ARIMAX model, or one SARIMAX model. For your model, be sure that you structure the model as follows:
- A forecasting algorithm named "model" using the statsmodels implementation of one of the four models covered in weeks 1 to 3. This model will use the average price of honey as the dependent variable, and may or may not use exogenous variables from the remainder of the dataset. The training data is contained in the lab1train.csv file.
- A fitted model named "modelFit". This should be a model instance capable of generating forecasts using new data in the same shape as the data used in part (1).
- A vector of forecasts using the data from the test period named "pred". You should have forecasts for 5 future periods.

### While you can use as many cells as you wish to work on your code, all final code (from import statements to predictions) should be pasted into the "Graded" cell at the bottom of this notebook. Any other code will not be considered when your assignment is graded

**Note**: While all models from weeks 1 to 3 are available to you, they may not all be good fits to the data. I recommend considering the data carefully, then choosing 2-3 models to try. See which models seem to perform best on this data, and implement the best choice for the final submission of the project.




