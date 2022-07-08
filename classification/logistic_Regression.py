import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('classification\Social_Network_Ads.csv')
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

from sklearn.linear_model import LogisticRegression

LGRClassifier = LogisticRegression(random_state = 0)
LGRClassifier.fit(X_train,Y_train)

# X_predict = LGRClassifier.predict(sc.transform([[30 ,87000]]))
# print(X_predict)

y_pred = LGRClassifier.predict(X_test) #predicting the test set 
# np.set_printoptions(precision=2)   #setting precision to 2
# print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1)) #concatenating predicted and actual values

#making confusion matrix
"""A confusion matrix is a table that is used to define the 
    performance of a classification algorithm. A confusion matrix 
    visualizes and summarizes the performance of a classification 
    algorithm
"""
from sklearn.metrics import confusion_matrix , accuracy_score # accuracy score is used to calculate the accuracy of the model
cm = confusion_matrix(Y_test,y_pred)
print(cm)
ac = accuracy_score(Y_test,y_pred)
print(ac)

#visualizng training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, LGRClassifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = sc.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()