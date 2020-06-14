import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_fwf('brain_body.txt')
x = df[['Brain']]
x = np.array(x)
y = df[['Body']]
y = np.array(y)

body_reg = linear_model.LinearRegression()
body_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x, body_reg.predict(x))
plt.show()
