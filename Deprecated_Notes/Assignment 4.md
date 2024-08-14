# Assignment 4

Given the nature of the content covered by this assignment, the assignment itself will be slightly different than before. All previous assignments have been automatically graded. Because `pymc3` models take so long to run, our grading machine will not be able to run. These models very often take longer than 3 minutes (the max I am allowed to allocate) to complete.

Instead, the assignment will be more open-ended than previous assignments, and will have more flexible grading criteria. You will be able to see each of the grading criteria when you submit (and you can still submit as many times as you would like). I will ONLY grade the final submission. If you make many submissions, and decide you prefer an earlier submission, just submit it again as the last submission, and that will be the one that I grade!

## Your task

How danceable are popular songs? Has their danceability changed over time? Using the Bayesian modeling techniques that we are learning during this portion of class, you should be able to model danceability and track changes in how danceable the typical Top 50 song is over the course of 10 years. The data can be found [here](https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/spotify_2010_2019.csv), and details about the data provided are available on [Kaggle](https://www.kaggle.com/datasets/leonardopena/top-spotify-songs-from-20102019-by-year). 

In addition to creating a model to track danceability, feel free to incorporate any other information you find relevant. Because I will be grading this assignment manually, please make use of comments in your code to justify what you choose to do.

You will be graded on the following:
1. Code running without errors
2. Comments explaining and justifying what you choose to do
3. Explanation of the prior distributions that you choose
4. Results detailing whether or not danceability changed during the ten year span
5. If you find that danceability changed, provide evidence for when the change occurred

Note: The inspiration for this project comes from the SMS example in the class notes. You'll have to make changes, but should be able to base your model on much of the work done in that example! :)
