import pandas as pd
import numpy as npi
import patsy as pt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv("/home/dusty/DatasetsDA/Titanic/train.csv")


model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)

y, x = pt.dmatrices("Survived ~ -1 + Sex + Age + SibSp + Pclass", data=data)

x, xt, y, yt = train_test_split(x, y, test_size=0.33, random_state=42)

res = model.fit(x,y)

print("\n\nIn-sample accuracy: %s%%\n\n" % str(round(100*accuracy_score(y, model.predict(x)), 2)))


from sklearn.tree import export_graphviz

export_graphviz(res, "/home/dusty/Econ8310/Code/tree.dot")


print("\n\nOut-of-sample accuracy: %s%%\n\n" % str(round(100*accuracy_score(yt, model.predict(xt)), 2)))