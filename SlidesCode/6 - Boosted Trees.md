---
marp: true
title: Week 9 - Boosted Trees
theme: default
class: default
size: 4:3
---

# COMING SOON!

# Ensemble Flavors - Extended

---

# Bagging (Refresh)

Bagging (**B**ootstrap **Agg**regation) is a simple way to start creating an ensemble model.

Standard Model:

$$ \hat{f}(x) = f^*(x) $$

<br>

All training data is used to generate our single best estimate of the true functional form, $f(x)$.

---

# Bagging

$$ \hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^B f^*_b(x) $$

<br>

In bagging, each estimate utilizes a bootstrap (random) sample of the training data

The bagged estimate is then based on the weighted average of all of the models

---

# Boosting

If we boost an algorithm using $M$ stages, then we need to define $f_m(x)$ at each stage

$$ \hat{f}_0(x) = 0 $$

At each subsequent stage, we solve for

$$ \hat{f}_m(x) = \hat{f}_{m-1}(x) + f^*_m(x) $$

So that each stage adds more information to our model.

---

# Boosting vs Bagging

**Bagging**:
- An averaged model utilizing bootstrapped samples of the complete dataset

**Boosting**:
- An additive model, where the predictions are incrementally improved

---

# Boosting vs Bagging


**Bagging**:
- Much easier to implement
- Less Overfitting

**Boosting**:
- Better Performance (generally)
- More vulnerable to overfitting

---