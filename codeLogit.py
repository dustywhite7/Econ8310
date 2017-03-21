import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1, num=1001)
e = np.random.random(1001)
y = x*e
y1 = (y>.5)
y1 = np.apply_along_axis(int, y1)
y1.dtype = np.int8


plt.scatter(x, y1)
plt.show()
