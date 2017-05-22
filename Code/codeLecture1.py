import numpy as np
from bokeh.plotting import figure, show


x = np.linspace(-1, 1, 101)
y = 2 * (x + np.random.rand(101))

xs = np.concatenate((np.ones(101).reshape(101,1),
		     x.reshape(101,1)), axis=1)
             
beta = np.linalg.solve(np.dot(xs.T, xs), np.dot(xs.T, y))

yhat = beta[0] + beta[1]*x


p = figure(plot_width = 800, plot_height=500)

p.scatter(x,y, color='green', size = 10, alpha=0.5)
p.line(x, yhat, color='red', line_width=3, alpha=0.5)

show(p)