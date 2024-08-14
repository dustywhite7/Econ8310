# Assignment 2

This assignment will make use of the models covered in Weeks 4 to 6. Models include:

- Vector Autoregressive (VAR) models
- Generalized Additive Models (GAMs)
- Exponential Smoothing models

Your job will be to forecast the number of taxi trips requested during each hour in a week in New York City, utilizing past data about taxi trips in New York City. Your grade will be assigned based on the performance of your code, and will be based primarily on:

- Your code executing without errors
- Storing your models to make predictions on new data
- Making reasonable predictions based on the data provided

The data is available at [https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2.csv](https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2.csv)

To complete this assignment, your code will need to contain the following:

- One valid model. This can be one VAR model, one GAM model (please use pyGAM, prophet breaks the grading machine), or one exponential smoothing model. For your model, be sure that you structure the model code as follows:

    - A forecasting algorithm named "model" using the implementation of one of the four models covered in weeks 4 to 6 (don't use other libraries, since I can't keep track of all of them). This model will use the number of trips in an hour as the dependent variable, and may or may not use exogenous variables from the remainder of the dataset. The training data is contained in the lab1train.csv file.
    - A fitted model named "modelFit". This should be a model instance capable of generating forecasts using new data in the same shape as the data used in part (1).
    - A vector of forecasts using the data from the test period named "pred". You should have forecasts for **744 future periods**. The data from which to generate forecasts (if you used exogenous variables) is contained in the lab1forecast.csv file.
    
    To make predictions, you can use the test data set found at the following link: [https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2test.csv](https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2test.csv)

### While you can use as many cells as you wish to work on your code, all final code (from import statements to predictions) should be pasted into the "Graded" cell at the bottom of this notebook. Any other code will not be considered when your assignment is graded

**Note:** While all models from weeks 4 to 6 are available to you, they may not all be good fits to the data. I recommend considering the data carefully, then choosing 2-3 models to try. See which models seem to perform best on this data, and implement the best choice for the final submission of the project.

