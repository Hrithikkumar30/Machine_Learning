import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets

datasets = pd.read_csv('H:\MachineLearning-Learn\Machine_Learning\Regressions\Position_Salaries.csv')
X = datasets.iloc[: , 1:-1].values
y = datasets.iloc[: , :-1].values

from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

RFregressor.fit(X,y)

RFregressor.predict([[6.5]]) #random forest prediction

X_grid = np.arrange(min(X) , max(X) ,0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y , color ='red')
plt.plot (X_grid , RFregressor.predict(X_grid),color='blue')
plt.show()