import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go


x = np.linspace(-1, 1, 101)
y = 2 * (x + np.random.rand(101))

xs = np.concatenate((np.ones(101).reshape(101,1),
		     x.reshape(101,1)), axis=1)
             
beta = np.linalg.solve(np.dot(xs.T, xs), np.dot(xs.T, y))

yhat = beta[0] + beta[1]*x

trace1 = go.Scatter(
    x = x,
    y = y,
    mode = "markers"
    )

trace2 = go.Scatter(
    x = x,
    y = yhat,
    mode = "lines"
    )

data = go.Data([trace1, trace2])

plot(data)