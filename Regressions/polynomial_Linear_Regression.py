import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import poly

dataset = pd.read_csv("./Regressions/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[: , -1].values

from  sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
print(X_poly)