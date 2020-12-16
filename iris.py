#we will first learn from a model that has been there since the inception of statistical computational analysis,that was used to classify iris .
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
y.shape
X.shape

#1. Import the class representing your model. Here we will use the KNearestNeighbour
from sklearn.neighbors import KNeighborsClassifier
#2. Create an instance of this class with k set to 3.
knn = KNeighborsClassifier(n_neighbors=3)
#3. Train the model with your data.
knn.fit(X, y)
#4. Predict the response for a new observation.
#For this, you use the “predict” method,
#passing it a 1x4 NumPy array containing the same four
#features you used in all of your previous observations:
knn.predict([[3,5,4,2]])
#You get back the following response:
array([1])
#In other words, the model believes the most appropriate
#response would be category 1. Or, if you feed that back
into iris.target_names:
iris.target_names[knn.predict([[3,5,4,2]])]
#you get back:
array([‘versicolor’],
 dtype=’|S10’)
#So, the new flower would most likely be of type “versicolor”

#Data Validatin is done using a technique known as train-split-test.
#And sure enough, scikit-learn provides a way of doing this:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
➥test_size=0.4)
