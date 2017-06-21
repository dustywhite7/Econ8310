## ECON 8310 / BSAD 8080 -- Business Forecasting
Days and Times: Thursdays, 6:00 to 8:40 PM<br>
Classroom: Mammel Hall 115

The course will cover forecasting tools and applications applied to business settings. We will cover traditional Econometric forecasting methods in the first half of the class. In the second half of the course, we will focus on models in predictive analytics and machine learning, since these models are quickly becoming critical tools for forecasters in many settings. The course will include lecture and lab time, and labs will be focused on teaching students how to implement the models discussed in lectures.

#### Office Hours
I will hold office hours on Mondays and Thursdays from 5-6 PM, and by appointment. I would love to get to know you and your goals, so please come by and talk to me.

#### Grading
This course will be graded as follows:
<ul>
  <li> 500 points of your grade will be based on the assignments that make up lab.</li>
  <li> 250 points will be based on an in-class, two day midterm project/presentation. More details will be provided in class</li>
  <li> 250 points will be based on an in-class, two day final project/presentation. More details will be provided in class</li>
</ul>
Final grades will be based on the total points you earn, and distributed according to the following scale.

| Letter | Percent |
| ------- | ------ |
| A | 940-1000 |
| A- | 900-939 |
| B+ | 870 - 899 |
| B | 840 - 869 |
| B- | 800-839 |
| C+ | 770-799 |
| C | 740-769|
| C- | 700-739|
| D+ | 660-699|
| D | 600-659 |
| F | < 600 |

### Projects
The exams in this course will be two projects, for which you will be given two class periods to prepare and present. The best way to learn is to do, and so we will focus on *doing* forecasting. I don't expect you to know how to code when the semester starts, but the course will be based on writing code, so I do expect you to learn as the course progresses. I will help you do so, and will make the process as painless as possible. The primary goal is to help you *do* forecasting. Your entire grade is based on coding projects (and presentations of some of the projects) and will depend heavily on teamwork, so please make sure that you schedule time to remain for *all* of class each week. These projects must be done as part of a group (*recommended*).


### Course Schedule

#### Week 1
**Review of OLS, Tools for Class** -- We will review the requirements of the course, the software that we will be using (mostly the Anaconda distribution of Python), and the principles of least squares regressions.

#### Week 2
**Time Series Models - ARIMA** -- After our refresher session, we will start working with Autoregressive Integrated Moving Average (ARIMA) models, a staple in time series analysis and in forecasting data that contains periodicity, time trends, and noisy outcomes.

#### Week 3
**Time Series Models - ARIMAX** -- While the ARIMA model is univariate, we can extend its application to multivariate problems by using the ARIMA with Exogenous Variables (ARIMAX) model. We can now control for events contained within exogenous variables while constructing our forecast model on the variable of interest.

#### Week 4
**Time Series Models - VAR** -- Unlike the ARIMA model, the VAR model is designed with multivariate analysis in mind from the start. The goal of this model is to determine the way that **all** variables interact with each other over the course of time. This model is very popular in the macroeconomic literature, since it permits us to observe the effect of a shock (or change) in a single variable on all other variables simultaneously. Additionally, we can use the model to generate forecasts of a variable of interest.

#### Week 5
**Time Series Models - GAM** -- All of the models previously considered have been models that assume linear relationships between exogenous and endogenous variables. A Generalized Additive Model allows for non-linear relationships between exogenous and endogenous variables, while also maintaining the additive nature of the overall functional form. This allows for tremendous flexibility while preserving overall ease of interpretation of the effect of each individual exogenous variable.

#### Week 6
**Panel Data Models - Fixed-Effects Model** -- When we utilize time series data of multiple individuals (but still measuring the same variable), we would do well to account for the differences between individuals that may not be observable through the inclusion of available exogenous variables. Panel data regressions allow us to do just that. By including fixed-effects, we are able to gain information about each individual, and to use that information to generate more accurate forecasts for each.

#### Weeks 7 & 8
**Midterm Project** -- Your group from lab will be assigned a problem to solve, that can make use of any of the previously presented methods in class. Your goal is to create a prototype solution to the problem, and then to present your solution to the class. The first period will be devoted to preparation, and the second period will be focused on team presentations.

#### Week 9
**Logistic Regression** -- Regressions typically assume that the dependent variable is continuous (or at least can be treated as such). Logistic regression, however, is a transformation of the linear regression model which confines the dependent variable to the [0,1] interval. This makes logistic regression an excellent tool when seeking to classify binary outcomes in which it is important to identify the effect of individual exogenous variables.

### Week 10
**Classification and Decision Trees, Part 1** -- Decision trees are the first of several "machine learning" models that we will work with. Before we discuss decision trees in depth, however, we must first explore the problems caused by the nature of classification problems with high dimensionality, as well as how we calculate entropy. We will then begin to create decision tree classification models, and discuss their strengths and weaknesses.

### Week 11
**Classification and Decision Trees, Part 2** -- One of the greatest dangers of machine learning algorithms is overfitting. This session, we will explore ways in which we can avoid overfitting our model, and ensure that our predictions can be generalized to populations not included in the original data.

### Week 12
**Ensemble Methods - Random Forests** -- Ensemble methods are a powerful machine learning tool, and one that is regularly able to provide easy improvements to out-of-sample predictive accuracy. We will focus primarily on the Random Forest ensemble classifier, and its improvements over the single decision tree classifier.

### Week 13
**Support Vector Machines** -- Decision trees are restricted in their ability to separate points in space by only separating based on a single variable at a time. Support Vector Machines (SVMs) are able to draw arbitrary separating hyperplanes that distinguish observations by using more flexible functional forms.

### Week 14
**k-Nearest Neighbors** -- Just like a real-estate appraisal, it is frequently true that observations will behave most like their nearest neighbors. We will discuss what it means for two observations to be close together, and then explore the k-Nearest Neighbor algorithm (kNN), and how it uses the distances between observations to generate predictions.

### Weeks 15 & 16
**Final Project** -- Your group from lab will be assigned a problem to solve, that can make use of any of the methods from class. Your goal is to once again create a prototype solution to the problem you are given, and then to present your solution to the class. The first period will be devoted to preparation, and the second period will be focused on team presentations.


### ACADEMIC INTEGRITY

UNO’s requirements for Academic Integrity and Behavior All students are required to adhere to the highest standards of academic integrity and behavior and must satisfy the UNO Academic Integrity Policy [http://www.unomaha.edu/student-life/student-conduct-and-community-standards/policies/academic-integrity.php](http://www.unomaha.edu/student-life/student-conduct-and-community-standards/policies/academic-integrity.php)  and Student Code of Conduct [http://www.unomaha.edu/student-life/student-conduct-and-community-standards/policies/code-of-conduct.php](http://www.unomaha.edu/student-life/student-conduct-and-community-standards/policies/code-of-conduct.php). It is the student’s responsibility to read, understand and abide by these policies.

If I find that you have plagiarized, been dishonest in completing your assignments, or cheated an an exam or assignment, then I reserve the right to award you no points on the entire exam, project, or assignment and to report the behavior to the university. If this behavior is repeated, I reserve the right to award a failing grade, independent of your score on other assignments. Academic integrity is essential to education, and I take it very seriously.
