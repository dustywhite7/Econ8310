# Random Forests

## How does a Random Forest work?

Random forests are an **ensemble** method of machine learning based on the decision tree/CART models that we have already discussed. **Ensembles** are an exciting concept within machine learning. Some models (perhaps CARTS) are very simple and fast, but tend to overfit to the data. This means that a single decision tree might learn the in-sample data very well, but struggle to make accurate predictions out-of-sample. In order to counter the tendency to overfit a model to the data, we can use an **ensemble** of models. This means that we create *many* similar models, and then aggregate their predictions to come up with a single model based on all of the lower-level models.

My favorite conceptual example of an ensemble model is voting in elections. We all know people who have extreme views (in one direction or another). For the most part, those extreme views are unhelpful to society. But what if your crazy uncle got to be the only person to choose the outcome of an election? This kind of behavior used to happen (and still happens in some places), and typically results in troubling outcomes. Instead, most developed nations rely on elections to determine the form that government and laws should take. Why?

We participate in democratic elections because they have a tendency to eliminate extreme outcomes. In order to gain sufficient votes to enter government or become policy, the candidate or policy must clear a significant threshold of votes. Extreme views on each end will be cancelled out by the extreme views that oppose them. Then, the sensible, and typically moderate, outcomes are favored over the more extreme outcomes. This results in more stable outcomes, both in elections and in machine learning.

## Building a Random Forest

In order to construct a random forest, we start with decision trees, and we make lots of them. These decision trees have several key differences from the decision trees that we might implement in isolation, though! When we create a random forest, we need to make use of **randomness** to improve our model. Each decision tree is given the following:
1. A random sample of the available data
2. A random sample of variables to choose from at each stage of the tree-building process

Why? The random sample of data allows each tree to see the data slightly differently, and thereby increase the robustness of the overall model. The random samples of variables force each tree to choose from different variables, increasing the number of variables that are utilized in our model. This way, we don't just have a bunch of trees with the same "world-view". Some trees use really crappy variables, and some use the best variables (just like when you and your uncle vote!).

Once we have a large forest of trees, it is time to implement the voting procedure. Each tree has been trained to make predictions on the same set of data, so we now ask each tree to make its prediction. Then, we aggregate those predictions to determine the most likely outcome based on the collective opinions of all of the tree models. In this way, the poor trees are "voted out", and the highly informed trees provide the basis of our predictive model.

Finally, it is worth noting that building a random forest does mean a decrease in transparency. While we can still determine the variables which provide the most information to our model, we can no longer draw a single tree to present to our stakeholders. Instead, we can use representative trees from our forest, or describe the forest process. As we increase the complexity of our models, we have to work to increase the trust from stakeholders in models that are more difficult to understand.

## Implementing Random Forests

Random forests are computationally more complex than decision trees, but in practice are no more complicated to implement in `sklearn` than any other model. They do, however, have a few more arguments that we can use to change how they are fitted.


```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data, then separate x and y variables
data = pd.read_csv("https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/roomOccupancy.csv")
y = data['Occupancy']
x = data[['Temperature','Humidity','Light','CO2','HumidityRatio']]

# Randomly sample our data --> 70% to train with, and 30% for testing
x, xt, y, yt = train_test_split(x, y, test_size=0.3)

# Create the model and fit it using the train data
clf = RF(n_estimators=100, n_jobs=-1, max_depth=5)
clf.fit(x, y)

# Test our model using the testing data
pred = clf.predict(xt)
acc = accuracy_score(yt, pred)

print("Model accuracy is {}%.".format(acc*100))
```

    Model accuracy is 99.01760130986492%.


We break our data into training and testing sets, then implement our random forest model (imported as `RF` to make typing easier). We use `n_estimators` to specify the number of trees we want to include in our model. `n_jobs` is used to specify the number of cores in our processor that we would like to use to estimate these trees. Each tree is independent of the others, so trees can be trained in parallel wherever there are multiple processors available. Providing a value of `n_jobs=-1` tells `sklearn` to make use of all available processing cores, in order to accelerate the training process as much as possible. Note that this is likely to make it hard to do other things on your computer while the model trains.

Finally, we are able to use our tree-specific arguments, such as `max_depth`, just like we did with our decision trees. These arguments will be applied to all of the trees in the forest.

## Introducing MNIST

The Modified National Institute of Standards and Technology database (MNIST) is a database of handwritten single-digit numbers that are frequently used as an introduction to machine vision. We can train a machine learning model to recognize patterns in an image that will help that model to discern between the possible classes that the images belong to.

Each image is 28 x 28 pixels (784 variables), and has a buffer of empty pixels around the outside, so that the image is already cleaned and centered. The best way to understand the data, though, is to visualize some of it. Let's load a [sample of the data](https://github.com/dustywhite7/pythonMikkeli/blob/master/exampleData/mnistTrain.csv?raw=true).


```python
# Import plotting library, numpy to shape our data
import plotly.express as px
import numpy as np

# Read in the MNIST sample
mnist = pd.read_csv("https://github.com/dustywhite7/pythonMikkeli/blob/master/exampleData/mnistTrain.csv?raw=true")

# Randomly choose a row, ignore first column (the label), transform into an array
image = np.asarray(mnist.iloc[np.random.randint(1000),1:])

# Reshape the data into a 28x28 square
image = np.reshape(image, (28,28))

# Render the image
px.imshow(image, color_continuous_scale ='Greys')
```

With the MNIST dataset, we can test our ability to identify/classify handwritten digits using algorithms ranging from Decision Trees to Random Forests to neural networks! In this class, we will keep the data in a flat structure (with one row of 784 pixel variables for each image) in order to allow our models to try to classify which of the ten integers each image represents. 

This well-known dataset is both challenging and engaging, since it forms the basis for the kinds of models that are necessary to identify other items in images (such as people), and that can provide the inputs required for such use cases as self-driving cars or identifying farm yields based on satellite images!

**Solve it**:

Create a Random Forest model that predicts the handwritten digit from the [MNIST Dataset](https://github.com/dustywhite7/pythonMikkeli/raw/master/exampleData/mnistTrain.csv). Name your model `writingForest`. You will be awarded points for the following:
 - Creating a working Random Forest model named `writingForest` **[2 points]**
 - `writingForest` the 87% accuracy threshold on data it has never seen before (from the same data source) **[2 points]**
 - `writingForest` passes the 92% accuracy threshold on data it has never seen before (from the same data source) **[2 points]**
 
The outcome of interest is stored in the variable `Label` (this should be your `y` variable). All other variables should be included in your `x` array (remember that the decision tree algorithm will pick the best variables for you!). 

All necessary code for your model to be trained and tested should be placed in the cell below:

