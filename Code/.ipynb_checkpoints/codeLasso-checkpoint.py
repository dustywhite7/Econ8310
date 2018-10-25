import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso 
import patsy as pt

# data = pd.read_csv("/home/dusty/Downloads/obvienceDataHR.tsv", sep='\t')
# data.to_csv("/home/dusty/Downloads/obvienceDataHR.csv", index=False)

data = pd.read_csv("nflValues.csv")

y, x = pt.dmatrices('np.log(OperatingIncome) ~ Expansion + Exp2 + Exp3 + Exp4 + Exp5 + Exp6 + TVDeal + LaborContract + Playoffs + SuperBowl + Revenues + ChangeValue + Value', data=data)
y = np.ravel(y)


coef = []
for i in range(1000):
  model = Lasso(alpha = (i/20000))
  reg = model.fit(x, y)
  coef.append(reg.coef_)
  
coef = pd.DataFrame(coef, columns=x.design_info.column_names)

from plotly.offline import plot
import plotly.graph_objs as go

# Plot the DIFFERENCED data

traces = []
for i in coef.columns:
  traces.append(go.Scatter(
      x = [i/20000 for i in range(1000)],
      y = coef[i],
      mode = 'lines',
      name=i
      ))

pdata = go.Data(traces)

layout = go.Layout(
    title="Dimensionality Reduction via LASSO Estimation",
    xaxis = dict(title = 'Penalty'),
    yaxis = dict(title = 'Coefficient')
    )

plot(go.Figure(data=pdata, layout=layout))


# # Pure Lasso

# model = Lasso(alpha = (i/20000))
# reg = model.fit(x, y)

# results = pd.DataFrame([reg.coef_], 
#             columns = x.design_info.column_names, 
#             index = ['Coefficients']
#             ).T

# # Logistic Lasso
# from sklearn.linear_model import LogisticRegression \
#                                  as Logit 

# model = Logit(penalty = 'l1', C=1/0.05)
# # C is an inverse penalty weight, so that
# #   smaller C = larger penalty
# reg = model.fit(x, y)

# results = pd.DataFrame([reg.coef_], 
#             columns = x.design_info.column_names, 
#             index = ['Coefficients']
#             ).T

# # Implementing PCA

# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# pca.fit(x)

# newX = pca.transform(x)