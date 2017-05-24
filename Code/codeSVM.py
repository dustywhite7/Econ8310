import pandas as pd
import numpy as np
import patsy as pt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("/home/dusty/DatasetsDA/gradeData/CleanedData/passFailTrain.csv")

y, x = pt.dmatrices("G3 ~ -1 + sex + age + C(address) + C(famsize) + C(Pstatus) + studytime + failures + schoolsup + paid + freetime + absences + health + C(Mjob) + C(Fjob) + C(Medu) + C(Fedu) + G1 + G2", data=data)

y = np.ravel(y)

x, xt, y, yt = train_test_split(x, y, test_size = 0.1, random_state=42)

model = SVC(C=0.1, kernel='linear')
reg = model.fit(x, y)
pred = reg.predict(xt)

print("\nModel accuracy is %s%%\n" % str(round(accuracy_score(pred, np.ravel(yt)), 2)*100))
