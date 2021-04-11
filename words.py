# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:32:43 2021

@author: Kieru
"""

#Import important libraries
print('Importing important libraries')
print("Done importing libraries!\n############################\n")

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Preview of Data
print('Preview of Data')
data = pd.read_csv('iris.csv')
print(type(data.head()))
print(data.info())
print(data.describe())
print(data['variety'].value_counts())

print(data.describe())
tmp=data
g = sns.pairplot(tmp, hue='variety', markers='+')
plt.show()

g = sns.violinplot(y='variety', x='length', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='nature', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='context', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='age', data=data, inner='quartile')
plt.show()




#Modeling with scikit-learn
print('Modeling with scikit-learn')
X = data.drop(['variety','word'], axis=1)
y = data['variety']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)

#Option 1. Train and test on the same dataset
# experimenting with different n values
print('''
#Option 1. Train and test on the same dataset
# experimenting with different n values''')
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
#Log and check for accuracy of model
print('Log and check for accuracy of model')
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print('accuracy is' + str(metrics.accuracy_score(y, y_pred)))
print('make a prediction for an example of an out-of-sample observation\n')
print(knn.predict([[4.9,3,5.1,0.2]]))
print(knn.predict([[1,1,1,1]]))
print(knn.predict([[5,5,5,5]]))
print(knn.predict([[10,10,10,10]]))

testing=input("Enter a word:")
tlength=len(testing)
context=(10)
nature=(10)
age=(1)

print(knn.predict([[tlength,context,nature,age]]))

print("#"*100)
      
'''      
#Option2. Split the dataset into a training set and a testing set
print('#Option2. Split the dataset into a training set and a testing set')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# experimenting with different n values
k_range = list(range(1,18))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
 
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
#log and check for accuracy of model
print('log and check for accuracy of model')
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#Choosing KNN to Model Iris Species Prediction with k = 12
print('Choosing KNN to Model Iris Species Prediction with k = 12')
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
print('make a prediction for an example of an out-of-sample observation\n')
knn.predict([[4.9,3,5.1,0.2]])
'''
