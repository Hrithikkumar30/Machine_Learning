import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import poly

dataset = pd.read_csv("./Regressions/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[: , -1].values

from  sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# print(X_poly)
linea_reg = LinearRegression()
linea_reg.fit(X_poly, Y)

#visualizing the polynomial linear regression model
plt.scatter(X, Y, color = 'red')
plt.plot(X , linea_reg.predict(poly_reg.fit_transform(X)), color = 'blue')


print(linea_reg.predict(poly_reg.fit_transform([[8.2]])))  #prediction for a new value in polynomial regression
plt.show()