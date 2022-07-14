import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('H:\MachineLearning-Learn\Machine_Learning\classification\Social_Network_Ads.csv')
# print(dataset.head()) 

X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[:, -1].values
# print(Y)
# print(X)


from sklearn.model_selection import train_test_split
X_train ,X_test, Y_train ,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling   
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




from sklearn.svm import SVC
SVM = SVC(kernel = 'linear' , random_state = 0)
SVM.fit(X_train, Y_train)

y_pred = SVM.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1)) #concatenating predicted and actual values

from sklearn.metrics import confusion_matrix , accuracy_score # accuracy score is used to calculate the accuracy of the model
cm = confusion_matrix(Y_test,y_pred)
print(cm)
ac = accuracy_score(Y_test,y_pred)
print(ac)
