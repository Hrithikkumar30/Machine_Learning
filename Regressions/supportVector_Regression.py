import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasets =pd.read_csv('H:\MachineLearning-Learn\Machine_Learning\Regressions\Position_Salaries.csv')
X = datasets.iloc[: , 1:-1].values
y = datasets.iloc[: , -1].values


y = y.reshape(len(y) , 1) #reshape the y vector to a matrix in vertical
# print(y)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

# print(X)
# print(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')  #kernel = 'rbf' is the default kernel helps to avoid overfitting
regressor.fit(X , y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.9]]))))