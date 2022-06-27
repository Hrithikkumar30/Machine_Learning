import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import  SimpleImputer
# Importing the dataset
dataset = pd.read_csv('H:\MachineLearning-Learn\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python/Data.csv')
X = dataset.iloc[:,:-1].values #independent variable
y = dataset.iloc[:, -1].values #dependent variable
# print(dataset)
# print(X)


#data cleaning for larger dataset 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean' )
imputer = imputer.fit(X[:, 1:3]) #only fit on the columns that we want to replace
X[:, 1:3] = imputer.transform(X[:, 1:3]) #replace missing values with mean of column 1 and 2
# print(X)

#another method
""" 
trf1 = ColumnTransformer(transformers =[
    ('cat', SimpleImputer(strategy ='most_frequent'), ['sex', 'smoker', 'region']),
    ('num', SimpleImputer(strategy ='median'), ['age', 'bmi', 'children']),
      
], remainder ='passthrough')
"""


#data cleaning for small dataset using pandas
# mean_salary =dataset['Salary'].mean()
# print(mean_salary)
# dataset['Salary'].fillna(mean_salary , inplace=True)
# print(dataset)



#Encoding categorical data

#encoding idependent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ColTran = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ColTran.fit_transform(X))
# print(X)

#Encoding dependent variable
from sklearn.preprocessing import LabelEncoder

LabEnc = LabelEncoder()
y = LabEnc.fit_transform(y) #fit the label encoder to the dependent variable as 0 &1 (0=no, 1=yes)
# print(y) 


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)



#Feature Scalling
from  sklearn.preprocessing import StandardScaler # this package helps to scale the data
sc_X = StandardScaler()
X_train[:,3:] = sc_X.fit_transform(X_train[:,3:]) #fit the scaler to the training set
X_test[:,3:] = sc_X.transform(X_test[:,3:]) #transform the test set using the same scaler
print(X_train)
print(X_test)


