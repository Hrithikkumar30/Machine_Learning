import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets

datasets = pd.read_csv('H:\MachineLearning-Learn\Machine_Learning\Regressions\Position_Salaries.csv')
X = datasets.iloc[: , 1:-1].values
y = datasets.iloc[: , :-1].values

from sklearn.tree import DecisionTreeRegressor
DstReg = DecisionTreeRegressor(random_state = 0)

DstReg.fit(X, y)

print(DstReg.predict([[6]])) 


#low visualization
plt.scatter(X, y, color = 'red')
plt.plot(X , DstReg.predict(X), color = 'blue')
plt.show()


#high reselution visualizetion
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y , color ='red')
plt.plot(X_grid , DstReg.predict(X_grid) , color = 'blue')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

