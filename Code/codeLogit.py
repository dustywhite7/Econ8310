import numpy as np
import patsy as pt
import matplotlib.pyplot as plt
import mpld3
from statsmodels.discrete.discrete_model import Logit


x = np.linspace(0,1, num=1001)
e = np.random.random(1001)
y = x*e
y1 = (y>.5)
y1.dtype = np.int8


plt.scatter(x, y1)
plt.show()



data = pd.read_csv('/home/dusty/Econ8310/Code/passFailTrain.csv')

y, x = pt.dmatrices('G3 ~ G1 + age + goout', data = data)

model = Logit(y, x)

reg = model.fit()

print(reg.summary())

def rsqOut(model, x, y):
    yes = 0
    no = 0
    yesMean = 0
    noMean = 0
    for a in zip(x,y):
        if a[1]>0:
            yes += 1
            yesMean += model.predict(a[0])
        else:
            no += 1
            noMean += model.predict(a[0])
    yesMean = yesMean/yes
    noMean = noMean/no
    return yesMean-noMean
    
print("\nR-squared value: %s\n" % str(rsqOut(reg, x, y)))