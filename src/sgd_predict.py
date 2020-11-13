#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTE: This code is implemented by following the steps in the example provided
by Arpan Das at https://www.kaggle.com/arpandas65/simple-sgd-implementation-of-linear-regression#SGD-with-optimal-learning-rate
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from franke import franke
from utils import design_matrix

np.random.seed(2020)

# creating data
N = 100
deg = 3
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
X = design_matrix(x, y, deg)
Y = franke(x, y, noise_sigma=0.1, noise=True)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

train_data=pd.DataFrame(x_train)
train_data['z']=y_train

x_test=np.array(x_test)
y_test=np.array(y_test)


# SkLearn SGD classifier
clf_ = SGDRegressor()
clf_.fit(x_train, y_train)

print('SkLearn Mean Squared Error OLS :',mean_squared_error(y_test, clf_.predict(x_test)))

clf_ = Ridge(solver='sag') # Stochastic Average Gradient descent solver
clf_.fit(x_train, y_train)

print('SkLearn Mean Squared Error Ridge :',mean_squared_error(y_test, clf_.predict(x_test)))


# SkLearn SGD classifier predicted weight matrix
sklearn_w=clf_.coef_

# implemented SGD Classifier
def SGD(train_data,eta=0.001,n_epochs=1000,k=10):
    w_cur=np.zeros(shape=(1,train_data.shape[1]-1))
    b_cur=0
    cur_itr=1
    while(cur_itr<=n_epochs):
        w_old=w_cur
        b_old=b_cur
        w_temp=np.zeros(shape=(1,train_data.shape[1]-1))
        b_temp=0
        temp=train_data.sample(k)
        
        y=np.array(temp['z'])
        x=np.array(temp.drop('z',axis=1))
        for i in range(k):
            w_temp+=x[i]*(y[i]-(np.dot(w_old,x[i])+b_old))*(-2/k)
            b_temp+=(y[i]-(np.dot(w_old,x[i])+b_old))*(-2/k)
        w_cur=w_old-eta*w_temp
        b_cur=b_old-eta*b_temp
        if(w_old==w_cur).all():
            break
        cur_itr+=1
    return w_cur,b_cur

def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        y=np.asscalar(np.dot(w,x[i])+b)
        y_pred.append(y)
    return np.array(y_pred)
       
        
# Function to get optimal learning rate on the implemented SGD Classifier
x1_train,x1_test,y1_train,y1_test=train_test_split(X,Y,test_size=0.3)
x1_train,x1_cv,y1_train_,y1_cv_=train_test_split(x1_train,y1_train,test_size=0.3)

x1_train = scaler.transform(x1_train)
x1_cv=scaler.transform(x1_cv)

x1_train_=np.array(x1_train)
x1_train_data=pd.DataFrame(x1_train)
x1_train_data['z']=y1_train_

x1_cv_data=pd.DataFrame(x1_cv)
x1_cv_data['z']=y1_cv_

y1_train_=np.array(y1_train_)
y1_cv_=np.array(y1_cv_)

etas = np.logspace(-1, -6, 6)
def tune_etas():
    train_error=[]
    cv_error=[]
    
    for eta in etas:
        w,b=SGD(x1_train_data,eta=eta,n_epochs=1000)

        y1_pred_train=predict(x1_train_,w,b)
        train_error.append(mean_squared_error(y1_train_,y1_pred_train))
        w,b=SGD(x1_cv_data,eta=eta,n_epochs=1000)
        y1_pred_cv=predict(x1_cv,w,b)
        cv_error.append(mean_squared_error(y1_cv_,y1_pred_cv))
    return train_error,cv_error 

train_error,cv_error=tune_etas()

# plotting obtained values
plt.plot(np.log10(etas),train_error,label='train MSE')
plt.plot(np.log10(etas),cv_error,label='CV MSE')
plt.scatter(np.log10(etas),train_error)
plt.scatter(np.log10(etas),cv_error)
plt.legend()
plt.xlabel('log of learning rate')
plt.ylabel('Mean Squared Error')
plt.tight_layout()
plt.savefig('etas_vs_mse.png', dpi=300, bbox_inches='tight')
plt.show()

# running implemented SGD Classifier with obtained optimal learning rate
w,b=SGD(train_data,eta=0.1,n_epochs=1000)
y_pred=predict(x_test,w,b)

# Errors in implemeted model
print('Implemented Mean Squared Error :', mean_squared_error(y_test,y_pred))

# weight vector obtained from impemented SGD Classifier
custom_w=w

# MSE = mean squared error
x=PrettyTable()
x.field_names=['Model','Weight Vector','MSE']
x.add_row(['sklearn',sklearn_w,mean_squared_error(y_test, clf_.predict(x_test))])
x.add_row(['custom',custom_w,mean_squared_error(y_test,y_pred)])
print(x)

sklearn_pred=clf_.predict(x_test)
implemented_pred=y_pred
x=PrettyTable()
x.field_names=['SKLearn SGD predicted value','Implemented SGD predicted value']
for itr in range(15):
    x.add_row([sklearn_pred[itr],implemented_pred[itr]])
print(x) 

n_epochs = np.arange(2, 2001, 200) 
def tune_epochs():
    train_error=[]
    cv_error=[]
    
    for n in n_epochs:
        w,b=SGD(x1_train_data,eta=0.1,n_epochs=n)

        y1_pred_train=predict(x1_train_,w,b)
        train_error.append(mean_squared_error(y1_train_,y1_pred_train))
        w,b=SGD(x1_cv_data,eta=0.1,n_epochs=n)
        y1_pred_cv=predict(x1_cv,w,b)
        cv_error.append(mean_squared_error(y1_cv_,y1_pred_cv))
    return train_error,cv_error 

train_error,cv_error=tune_epochs()

# plotting obtained values
plt.plot(n_epochs,train_error,label='train MSE')
plt.plot(n_epochs,cv_error,label='CV MSE')
plt.scatter(n_epochs,train_error)
plt.scatter(n_epochs,cv_error)
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('Mean Squared Error')
plt.tight_layout()
plt.savefig('epochs_vs_mse.png', dpi=300, bbox_inches='tight')
plt.show()

