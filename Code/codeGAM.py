# Our import statements
import pandas as pd
import numpy as np
from pygam import LinearGAM 
from pygam.utils import generate_X_grid
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot, row
import matplotlib.pyplot as plt

# Importing data from the web
path = 'http://www.stat.cmu.edu/~larry/' \
	'all-of-nonpar/=data/rock.dat'

data = pd.read_csv(path, sep=' *', engine='python')

X = data[['peri','shape','perm']]
y = data['area']

adjy = y - np.mean(y)

gam = LinearGAM(n_splines=10).gridsearch(X, y)
XX = generate_X_grid(gam)

# fig, axs = plt.subplots(1, 3)
titles = ['peri', 'shape', 'perm']

# for i, ax in enumerate(axs):
#     pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    
#     ax.scatter(X[X.columns[i]], adjy, color='gray', edgecolors='none')
#     ax.plot(XX[:, i], pdep)
#     ax.plot(XX[:, i], confi[0], c='r', ls='--')
#     ax.set_title(titles[i])
    
    
pdep, confi = gam.partial_dependence(XX, width=.95)
p = list()

for i in range(3):
    p.append(figure(title=titles[i], plot_width=250, toolbar_location=None))
    p[i].scatter(X[X.columns[i]], adjy, color='gray', size=5, alpha=0.5)
    p[i].line(XX[:, i], pdep[:,i], color='blue', line_width=3, alpha=0.5)
    p[i].line(XX[:, i], confi[i][:, 0], color='red', line_width=3, alpha=0.5, line_dash='dashed')
    p[i].line(XX[:, i], confi[i][:, 1], color='red', line_width=3, alpha=0.5, line_dash='dashed')

show(row([p[0],p[1],p[2]]))
