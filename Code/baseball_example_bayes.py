#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install pymc3 daft')


# In[2]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")


# In[5]:


data = pd.read_csv("/mnt/c/Users/dusty/Desktop/mariners2021.csv")


# In[6]:


at_bats, hits = data[['at_bats', 'hits']].to_numpy().T


# In[41]:


import daft

# Instantiate the PGM.
pgm = daft.PGM()

# Hierarchical parameters.
pgm.add_node("phi", "$\phi$", 0.5, 3)
pgm.add_node("kappa", r"$\kappa$", 1.5, 3)

pgm.add_node("alpha", r"$\alpha$", 0.5, 1)
pgm.add_node("beta", r"$\beta$", 1.5, 1)

pgm.add_node("p", r"$p_i$", 1, 0, observed=True)

# Add in the edges.
pgm.add_edge("phi", "alpha")
pgm.add_edge("phi", "beta")
pgm.add_edge("kappa", "alpha")
pgm.add_edge("kappa", "beta")
pgm.add_edge("alpha", "p")
pgm.add_edge("beta", "p")


# Render and save.
pgm.render()


# In[20]:


N = len(hits)

with pm.Model() as baseball_model:

    phi = pm.Uniform("phi", lower=0.0, upper=1.0)

    kappa_log = pm.Exponential("kappa_log", lam=1.5)
    kappa = pm.Deterministic("kappa", tt.exp(kappa_log))

    thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, shape=N)
    y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)
    
with baseball_model:

    theta_new = pm.Beta("theta_new", alpha=phi * kappa, beta=(1.0 - phi) * kappa)
    y_new = pm.Binomial("y_new", n=50, p=theta_new, observed=0)
    
with baseball_model:
    trace = pm.sample(2000, tune=2000, chains=2, target_accept=0.95, return_inferencedata=True)

    # check convergence diagnostics
    assert all(az.rhat(trace) < 1.03)
    
az.plot_trace(trace, var_names=["phi", "kappa"]);


# In[21]:


player_names = data['name']

ax = az.plot_forest(trace, var_names=["thetas"])
ax[0].set_yticklabels(player_names.tolist());


# In[19]:


az.plot_trace(trace, var_names=["theta_new"]); # 4 at-bats


# In[16]:


az.plot_trace(trace, var_names=["theta_new"]); # 25 at-bats


# In[22]:


az.plot_trace(trace, var_names=["theta_new"]); # 50 at-bats


# In[ ]:




