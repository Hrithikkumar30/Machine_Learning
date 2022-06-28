import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("./Regressions/50_Startups.csv")
X = dataset.iloc[:,:-1].values #independent variable
y = dataset.iloc[:, -1].values #dependent variable

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ColTran = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')  #index of column is wriitten inside onehotencoder(),[]
X = np.array(ColTran.fit_transform(X))     
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

