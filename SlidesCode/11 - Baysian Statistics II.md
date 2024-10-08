---
marp: true
title: Bayesian Statistics Part 2
theme: default
class: default
size: 4:3
---

# Bayesian Statistics and Modeling<br>Part II

---

# Time series modeling with `pymc`

###### Based on [this example](https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html)

---

# Modeling airline trends

$$ Passengers \sim \alpha + \beta \cdot t $$

Just a simple linear trend for now. $\alpha$ is an intercept term, and $\beta$ is our slope term

Let's go to the code [here](https://github.com/dustywhite7/Econ8310/blob/master/Code/airlines_example_bayes.py) to start building our model

---

# Prior predictions (WHAT??)

![](airline_prior_1.png)

---

# Prior predictions (WHAT??)

- Look at a large array of possible "reasonable" outcomes given our assumptions about the data
- Gives us an idea of whether our priors make sense
- In this case, we want to make some corrections

---

# Updated priors

![](airline_prior_2.png)

---

# Posterior predictions

- Incorporate our actual data and then compare our model to observed outcomes
- Decide if we think that our model can make reasonable predictions

---

# Posterior predictions

![](airline_posterior_1.png)

---

# Adding seasonality

We add a group of periodic functions (fourier features) to function as our "seasonality splines" (if we think of our model as a GAM). They will get stretched or weighted based on observations.

---

# Seasonality (multiplicative)

$$ Passengers \sim (\alpha + \beta \cdot t) \cdot (1 + seasonality) $$

Our seasonal terms interact with each term in our original model to increase/decrease the expected number of passengers

---

# Seasonal priors

![](airline_prior_3.png)

---

# Seasonal posteriors

![](airline_posterior_2.png)

---

# Modeling baseball outcomes

###### A revised/updated version of [this tutorial](https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/hierarchical_partial_pooling.html)

Follow along with the tweaked code [here](https://github.com/dustywhite7/Econ8310/blob/master/Code/baseball_example_bayes.py)

---

![bg left 70%](process_bayes_2.png)

- $\phi$ (phi) - Our population-level expectation of batting average
- $\kappa$ (kappa) - Population variance in batting average
- $\alpha, \beta$ - Parameters of our beta distribution
- $p_i$ - Individual batting average

---

![bg left 70%](process_bayes_2.png)

$$ \alpha = \phi \cdot \kappa $$

$$ \beta = (1-\phi) \cdot \kappa $$

---

# Beta distribution

![bg left 90%](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Beta_distribution_pdf.svg/650px-Beta_distribution_pdf.svg.png)

- Used where there are binary outcomes (hit or no hit)

- Tilts toward 1 or 0 based on observed outcomes and concentration of those outcomes

---

# Population values

![](baseball_bayes_1.png)

---

# Player with 4 at-bats, no hits

![](baseball_bayes_2.png)

---

# Player with 25 at-bats, no hits

![](baseball_bayes_3.png)

---

# Player with 50 at-bats, no hits

![](baseball_bayes_4.png)

---

# Mariners 2021

![h:600px](baseball_bayes_5.png)

---

# Data Storytelling

> Probabilistic programming will unlock narrative explanations of data, one of the holy grails of business analytics and the unsung hero of scientific persuasion. People think in terms of stories - thus the unreasonable power of the anecdote to drive decision-making, well-founded or not. But existing analytics largely fails to provide this kind of story; instead, numbers seemingly appear out of thin air, with little of the causal context that humans prefer when weighing their options.

-- B. Cronin ([full article](http://radar.oreilly.com/2013/04/probabilistic-programming.html))