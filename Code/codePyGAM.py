from pygam import LinearGAM, s, f
import pandas as pd
import patsy as pt
import numpy as np
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

# Prep the dataset
data = pd.read_csv(
    "/home/dusty/Econ8310/DataSets/HappinessWorld.csv")

# Generate x and y matrices
eqn = """"happiness ~ -1 + freedom + family + 
      year + economy + health + trust"""
y,x = pt.dmatrices(eqn, data=data)

# Initialize and fit the model
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
gam = gam.gridsearch(np.asarray(x), y)

# Specify plot shape
titles = ['freedom', 'family', 'year', 'economy',
          'health', 'trust']

fig = tools.make_subplots(rows=2, cols=3, subplot_titles=titles)
fig['layout'].update(height=800, width=1200, title='pyGAM', showlegend=False)

for i, title in enumerate(titles):
  XX = gam.generate_X_grid(term=i)
  pdep, confi = gam.partial_dependence(term=i, width=.95)
  trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
  ci1 = go.Scatter(x = XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
  ci2 = go.Scatter(x = XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')
  if i<3:
    fig.append_trace(trace, 1, i+1)
    fig.append_trace(ci1, 1, i+1)
    fig.append_trace(ci2, 1, i+1)
  else:
    fig.append_trace(trace, 2, i-2)
    fig.append_trace(ci1, 2, i-2)
    fig.append_trace(ci2, 2, i-2)

py.plot(fig)


# Making a Forecast

# predicting the outcome of the UAE in 2015
gam.predict([[0.64, 1.13, 2015, 1.47, 0.81, 0.38]])