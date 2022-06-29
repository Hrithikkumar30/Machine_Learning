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
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#trianing multiple linear regression model
from sklearn.linear_model import LinearRegression #importing linear model regression class from sklearn
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
regressor = LinearRegression() #calling LinearRegression class
regressor.fit(X_train, y_train) #fitting the model

#predicting test set results
y_pred = regressor.predict(X_test) #predicting the test set 
np.set_printoptions(precision=2)   #setting precision to 2
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) #concatenating predicted and actual values

# lreg = LinearRegression()
# sfs1 = SFS(lreg, k_features=5, forward=True, verbose=2, scoring='accuracy')
# sfs1 = sfs1.fit(X_train, y_train)
# # print(sfs1.k_feature_idx_)
# print(sfs1.k_score_)