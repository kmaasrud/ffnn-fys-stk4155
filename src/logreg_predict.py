#Import modules
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#Import logistic regression class and needed functions
from logistic_reg import *
from utils import accuracy

np.random.seed(2020)

#Setting up the data
breastcancer=load_breast_cancer()
y = breastcancer.target
X = breastcancer.data
y = y.reshape(len(y), 1)

# Setting up an overview of the data
cancer=np.count_nonzero(y==1)
not_cancer=np.count_nonzero(y==0)

#Printing the overwiev of the dataset
print(f"The dataset includes {cancer} patients with cancer, which is {np.around(cancer*100/(not_cancer+cancer),2)} %")
print(f"The dataset includes {not_cancer} patients without cancer, which is {np.around(not_cancer*100/(not_cancer+cancer),2)} %")

# Setting up the training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scaling the data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

y_train=np.ravel(y_train)

#Performs logisitc regression using scikit learn
log_reg_scikit= LogisticRegression(solver='liblinear')
y_pred=log_reg_scikit.fit(X_train_scaled, y_train)
accuracy_scikit=accuracy(y_test,y_pred)

# Let X_train and X_test be scaled
X_train = X_train_scaled
X_test = X_test_scaled

#Using logisitc regression to solve
"""
# Add intercept column to the X-data
one_vector = np.ones((len(y_train),1))
X_train1 = np.concatenate((one_vector, X_train), axis = 1)
one_vector = np.ones((len(y_test),1))
X_test1 = np.concatenate((one_vector, X_test), axis = 1)
"""

# Performs logistic regression
log_reg_code = LogReg(X_train, y_train)
log_reg_code.SGD_logreg(epochs=100, mini_batches=40)
pred = log_reg_code.predict(X_test)
accuracy_code = accuracy(y_test, pred)

print(f" Accuracy: logistic regression using the code: {accuracy_code}")
print(f" Accuracy: logistic regression using scikit: {accuracy_scikit}")
