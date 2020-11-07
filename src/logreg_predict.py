#Import modules
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import time

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

#Raveling the y-values
y_train_ravel=np.ravel(y_train)
y_test_ravel=np.ravel(y_test)

# Scaling the data using the scikit learn modules
scaler = StandardScaler();  scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Function that performs logisitc regression using scikit learn
def log_reg_scikit_learn(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_test_ravel=y_test_ravel, y_train_ravel=y_train_ravel):
    start = time.time()
    log_reg_scikit= LogisticRegression(solver='liblinear')
    y_pred=log_reg_scikit.fit(X_train_scaled, y_train_ravel)
    accuracy_scikit=format(log_reg_scikit.score(X_test_scaled,y_test_ravel))

    print(f" Accuracy: logistic regression using the scikit: {accuracy_scikit}")

    end = time.time()

    print(f" The scikit function used {end-start} seconds to run")

    return accuracy_scikit


#Finding the best number of mini_batches with a set amount og epochs
def log_reg_best_mini_batch(epochs = 110, X=X, y=y):
    #Making a figure to plot the functions in
    plt.figure()

    n=10
    #Defining empty lists
    accuracy_test=[]
    accuracy_train=[]
    mini_batches_amount=[]

    #Iterating over the batches
    for i in range(n):
        print(f"{i*10} %")

        #Clearing the lists before calculating a new mini batch
        accuracy_test.clear()
        accuracy_train.clear()
        mini_batches_amount.clear()

        #looping over the mini batches
        for j in np.arange(1,150, n):
            #mini_batches_amount.clear()
            mini_batches_amount.append(j)
            test_accuracy_temp, train_accuracy_temp = CV_log_reg(X, y, epochs=epochs, mini_batches=j)
            accuracy_test.append(test_accuracy_temp)
            accuracy_train.append(train_accuracy_temp)

        #Plotting the accuracy scores for the train and test sets
        plt.plot(mini_batches_amount, accuracy_test, 'tab:red')
        plt.plot(mini_batches_amount, accuracy_train, 'tab:green')

    #Standard ploting commands
    plt.xlabel("Number of minibatches")
    plt.ylabel("Accuracy")
    plt.legend(['Test set', 'Train set'])
    #save_fig("LogRegcancer_accuracy_vs_mini_batches")
    plt.show()

    #Finding the best parameters
    best_index=np.where(accuracy_test==np.nanmax(accuracy_test))
    best_mini_batches=(best_index[0][0])*10+1

    print(f" Best amount of minibatches to use: {best_mini_batches}")

    return best_mini_batches

#Finding the best number of epochs with a set amount og mini_batches
def log_reg_best_epochs(mini_batches = 40, X=X, y=y):

    #Making a figure to plot the functions in
    plt.figure()

    n=10
    #Defining empty lists
    accuracy_test=[]
    accuracy_train=[]
    epochs_amount=[]

    #Iterating over the batches
    for i in range(n):
        print(f"{i*10} %")

        #Clearing the lists before calculating a new mini batch
        accuracy_test.clear()
        accuracy_train.clear()
        epochs_amount.clear()

        #looping over the mini batches
        for j in np.arange(1,200, n):
            #epochs_amount.clear()
            epochs_amount.append(j)
            test_accuracy_temp, train_accuracy_temp = CV_log_reg(X, y, epochs=j, mini_batches=mini_batches)
            accuracy_test.append(test_accuracy_temp)
            accuracy_train.append(train_accuracy_temp)

        #Plotting the accuracy scores for the train and test sets
        plt.plot(epochs_amount, accuracy_test, 'tab:red')
        plt.plot(epochs_amount, accuracy_train, 'tab:green')

    #Standard ploting commands
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Test set', 'Train set'])
    #save_fig("LogRegcancer_accuracy_vs_mini_batches")
    plt.show()

    #Finding the best parameters
    best_index=np.where(accuracy_test==np.nanmax(accuracy_test))
    best_epochs=(best_index[0][0])*10+1

    print(f" Best amount of minibatches to use: {best_epochs}")

    return best_epochs

#Function that performs logisitc regression using using the code
def logistic_reg(epochs=140, mini_batches=35, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_test=y_test, y_train=y_train):
    start = time.time()
    # Performs logistic regression
    log_reg_code = LogReg(X_train_scaled, y_train)
    log_reg_code.SGD_logreg(epochs=epochs, mini_batches=mini_batches)
    pred = log_reg_code.predict(X_test_scaled)
    accuracy_code = accuracy(y_test, pred)

    print(f" Accuracy: logistic regression using the code: {accuracy_code}")

    end = time.time()

    print(f" The self writtten function used {end-start} seconds to run")
    return accuracy_code


#Finding the best number of mini_batches and epochs without the L2 parametrization.
#The chosen epochs and minibatches i only optimal for this specific case
def log_reg_best_mini_batch_epoch(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_test=y_test, y_train=y_train):
    #Defining empty lists
    accuracy_list=[]
    mini_batch_list=[]
    epochs_list=[]

    #Iterating over the batches
    for e in range(40,201, 1):
        print(f"{(e-40)/1.6} %")
        #looping over the mini batches
        for mini in range(1,151, 1):
            #mini_batches_amount.clear()

            log_reg_code = LogReg(X_train_scaled, y_train)
            log_reg_code.SGD_logreg(epochs=e, mini_batches=mini)
            pred = log_reg_code.predict(X_test_scaled)
            accuracy_list.append(accuracy(y_test, pred))
            mini_batch_list.append(mini)
            epochs_list.append(e)

    max_accuracy = max(accuracy_list)
    max_index = accuracy_list.index(max_accuracy)
    best_mini_batch=mini_batch_list[max_index]
    best_epoch=epochs_list[max_index]


    print(f" Best amount of minibatches to use: {best_mini_batch}")
    print(f" Best amount of epochs to use: {best_epoch}")

    print(max_accuracy)
    return


#Calling the functions- Log reg with best parameters is run by running logisitc_reg()
log_reg_scikit_learn()
#log_reg_best_mini_batch()
#log_reg_best_epochs()
logistic_reg()

#Best parameters in this specific case
#log_reg_best_mini_batch_epoch()
