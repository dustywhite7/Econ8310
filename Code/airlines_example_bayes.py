#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install pymc3 daft')


# In[15]:


import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use("arviz-darkgrid")

# UTILITY FUNCTION

def _sample(array, n_samples):
    """Little utility function, sample n_samples with replacement"""
    idx = np.random.choice(np.arange(len(array)), n_samples, replace=True)
    return array[idx]


# In[9]:


df = pd.read_csv("https://github.com/facebook/prophet/raw/main/examples/example_air_passengers.csv", parse_dates=["ds"])


# In[10]:


df.plot.scatter(x="ds", y="y", color="k")


# In[12]:


# SCALE THE DATA

# Time
t = (df["ds"] - pd.Timestamp("1900-01-01")).dt.total_seconds().to_numpy()
t_min = np.min(t)
t_max = np.max(t)
t = (t - t_min) / (t_max - t_min)

# Number of passengers
y = df["y"].to_numpy()
y_max = np.max(y)
y = y / y_max


# In[16]:


# WITH LARGE DEFAULT VARIANCE

with pm.Model(check_bounds=False) as linear:
    α = pm.Normal("α", mu=0, sigma=5)
    β = pm.Normal("β", mu=0, sigma=5)
    σ = pm.HalfNormal("σ", sigma=5)
    trend = pm.Deterministic("trend", α + β * t)
    pm.Normal("likelihood", mu=trend, sigma=σ, observed=y)

    linear_prior_predictive = pm.sample_prior_predictive()

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["ds"],
    _sample(linear_prior_predictive["likelihood"], 100).T * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["ds"], _sample(linear_prior_predictive["trend"], 100).T * y_max, color="blue", alpha=0.05
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines");


# In[18]:


# WITH REDUCED VARIANCE IN PRIOR

with pm.Model(check_bounds=False) as linear:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    σ = pm.HalfNormal("σ", sigma=0.1)
    trend = pm.Deterministic("trend", α + β * t)
    pm.Normal("likelihood", mu=trend, sigma=σ, observed=y)

    linear_prior_predictive = pm.sample_prior_predictive(samples=100)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["ds"],
    _sample(linear_prior_predictive["likelihood"], 100).T * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["ds"], _sample(linear_prior_predictive["trend"], 100).T * y_max, color="blue", alpha=0.05
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines");


# In[19]:


# POSTERIOR CHECK

with linear:
    linear_trace = pm.sample(return_inferencedata=True)
    linear_posterior_predictive = pm.sample_posterior_predictive(trace=linear_trace)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["ds"],
    _sample(linear_posterior_predictive["likelihood"], 100).T * y_max,
    color="blue",
    alpha=0.01,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
posterior_trend = linear_trace.posterior["trend"].stack(sample=("draw", "chain")).T
ax[1].plot(df["ds"], _sample(posterior_trend, 100).T * y_max, color="blue", alpha=0.01)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines");


# In[21]:


n_order = 10
periods = df["ds"].dt.dayofyear / 365.25
fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)
fourier_features


# In[22]:


# BASED ON OUR PREFERRED PRIORS

coords = {"fourier_features": np.arange(2 * n_order)}
with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    trend = pm.Deterministic("trend", α + β * t)

    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T)
    )

    μ = trend * (1 + seasonality)
    σ = pm.HalfNormal("σ", sigma=0.1)
    pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)

    linear_with_seasonality_prior_predictive = pm.sample_prior_predictive()

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False)
ax[0].plot(
    df["ds"],
    _sample(linear_with_seasonality_prior_predictive["likelihood"], 100).T * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["ds"],
    _sample(linear_with_seasonality_prior_predictive["trend"], 100).T * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines")
ax[2].plot(
    df["ds"].iloc[:12],
    _sample(linear_with_seasonality_prior_predictive["seasonality"][:, :12], 100).T * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Prior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);


# In[23]:


# POSTERIOR CHECK

with linear_with_seasonality:
    linear_with_seasonality_trace = pm.sample(return_inferencedata=True)
    linear_with_seasonality_posterior_predictive = pm.sample_posterior_predictive(
        trace=linear_with_seasonality_trace
    )

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False)
ax[0].plot(
    df["ds"],
    _sample(linear_with_seasonality_posterior_predictive["likelihood"], 100).T * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
posterior_trend = linear_trace.posterior["trend"].stack(sample=("draw", "chain")).T
ax[1].plot(df["ds"], _sample(posterior_trend, 100).T * y_max, color="blue", alpha=0.05)
df.plot.scatter(x="ds", y="y", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines")
posterior_seasonality = (
    linear_with_seasonality_trace.posterior["seasonality"].stack(sample=("draw", "chain")).T
)
ax[2].plot(
    df["ds"].iloc[:12],
    _sample(posterior_seasonality[:, :12], 100).T * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Posterior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);


# In[ ]:




