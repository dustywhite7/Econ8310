import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,\
BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('/home/dusty/DatasetsDA/mnist/trainingFull.csv')

# Y, X = pt.dmatrices("Survived ~ -1 + C(Pclass) + Age + C(Sex) + SibSp + Parch + Fare + C(Embarked)", data=data)
Y = data['Label']
X = data.drop('Label', axis=1)

x, xt, y, yt = train_test_split(X, Y, test_size = 0.9, random_state=42)

tree = DecisionTreeClassifier(max_depth=5,
	min_samples_leaf=10)

tclf = tree.fit(x, y)

tpred = tclf.predict(xt)

print("\nThe decision tree has an accuracy of: %s\n" % str(accuracy_score(tpred, yt)))

forest = RandomForestClassifier(n_estimators=100, n_jobs = -1, random_state=42)

fclf = forest.fit(x, y)

fpred = fclf.predict(xt)

print("The random forest has an accuracy of: %s\n" % str(accuracy_score(fpred, yt)))

boost = GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)

boclf = boost.fit(x, y)

bopred = boclf.predict(xt)

print("The boosting algorithm has an accuracy of: %s\n" % str(accuracy_score(bopred, yt)))

bag = BaggingClassifier(n_estimators=100, n_jobs = -1, random_state=42)

baclf = bag.fit(x, y)

bapred = baclf.predict(xt)

print("The bagging algorithm has an accuracy of: %s\n" % str(accuracy_score(bapred, yt)))
