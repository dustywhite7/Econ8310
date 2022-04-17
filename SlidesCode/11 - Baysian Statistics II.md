---
marp: true
title: Bayesian Statistics Part 2
theme: default
class: default
size: 4:3
---

# Bayesian Statistics and Modeling<br>Part II

---

# Time series modeling with `pymc3`

###### Based on [this example](https://docs.pymc.io/en/v3/pymc-examples/examples/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html)

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

# Seasonal priors

![](airline_prior_3.png)

---

# Seasonal posteriors

![](airline_posterior_2.png)

---

# Modeling baseball outcomes

###### A revised/updated version of [this tutorial](https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/hierarchical_partial_pooling.html)

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