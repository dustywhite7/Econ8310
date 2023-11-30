---
marp: true
title: Bayesian Statistics Part 3
theme: default
class: default
size: 4:3
---

# Bayesian Statistics and Modeling<br>Part III

---

# Bayesian linear models

![bg right 90%](rrp_frequency.png)

I love (for better or worse) rooting for the Seahawks. In many recent years, they have insisted on running a LOT more than most teams.

(Plot is frequency of Run-Run-Pass sequences by team)

---

# Bayesian Linear Models

Can we determine the likelihood of a play being successful based on various characteristics of that play?

- Are runs more successful than passes? (unlikely, but Pete Carroll thinks so)
- We should probably also account for down and distance


---

# What's the model?

![](football_dag.png)

---

# What's the model?

Let's go look at our code now, and generate a regression model using the Bayesian method

---


# Complete Regression Results

![](bayes_nfl/bayes_nfl.png)

---

# Making sense of choices

What if we want to be able to look at specific contexts?

We write quick function and are off to the races!

---

# 1st and 10...

![](bayes_nfl/nfl_1st_10.png)

---

![](first_down.jpg)

---

# 2nd and 10...

![](bayes_nfl/nfl_2nd_10.png)

---

# 3rd and 10...

![](bayes_nfl/nfl_3rd_10.png)

---

# 4th and 10!!

![](bayes_nfl/nfl_4th_10.png)

---

# Ok, but maybe the Seahawks are different!

Are they? Let's estimate our model with only Seahawks data

---

# Complete Regression Results

![](bayes_nfl/bayes_sea.png)

---

# Making sense of choices

What if we want to be able to look at specific contexts?

We write quick function and are off to the races!

---

# 1st and 10...

![](bayes_nfl/sea_1st_10.png)

---

# 2nd and 10...

![](bayes_nfl/sea_2nd_10.png)

---

# 3rd and 10...

![](bayes_nfl/sea_3rd_10.png)

---

# 4th and 10!!

![](bayes_nfl/sea_4th_10.png)

---

# How does it work?

1. Our model creates a `trace` object
2. Each trace contains however many samples (in this case ~40k) of the estimated parameter
3. We use these to look at the **distribution of parameter values**

---

# Credible intervals

Rather than having Confidence Intervals, we have Credible Intervals in Bayesian statistics.
- 95% of sampled parameter values fall inside a 95% CI
- We can shape them arbitrarily
- We can also just use them to measure the likelihood that one measure exceeds another!
    - For example, our distributions for the seahawks overlap, but that doesn't mean running is EVER better!

---

# More flexibility

We have only scratched the surface, but we are starting to see how we can create flexible models that allow us to ask much more **real** questions than we might with a null-hypothesis framework

---

# Lab Time!