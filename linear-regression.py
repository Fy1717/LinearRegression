import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import  matplotlib.pyplot as plt

data = pd.read_csv("linear.csv")
x=data["metrekare"]
y=data["fiyat"]

x=x.values.reshape(99,1)
y=y.values.reshape(99,1)
lineer_regresyon=lr()
lineer_regresyon.fit(x,y)

lineer_regresyon.predict(x)
m=lineer_regresyon.coef_
b=lineer_regresyon.intercept_

a=np.arange(120)

plt.scatter(x,y)
plt.scatter(a,m*a+b)
plt.show()


