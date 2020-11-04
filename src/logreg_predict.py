#Import modules
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

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

#Raveling the y-values
y_train_ravel=np.ravel(y_train)
y_test_ravel=np.ravel(y_test)


# Performs logistic regression
log_reg_code = LogReg(X_train_scaled, y_train)
log_reg_code.SGD_logreg(epochs=80, mini_batches=20)
pred = log_reg_code.predict(X_test_scaled)
accuracy_code = accuracy(y_test_ravel, pred)


#Performs logisitc regression using scikit learn
log_reg_scikit= LogisticRegression(solver='liblinear')
y_pred=log_reg_scikit.fit(X_train_scaled, y_train_ravel)
#accuracy_scikit=accuracy(y_test,y_pred)
accuracy_scikit=format(log_reg_scikit.score(X_test_scaled,y_test_ravel))


print(f" Accuracy: logistic regression using the code: {accuracy_code}")
print(f" Accuracy: logistic regression using the scikit: {accuracy_scikit}")

#Getting overflow for some reason, will try to fix tomorrow
def LogReg_optimize_n_minibatches(epochs = 100, X=X, y=y):
    print("\nOptimizing number of minibatches for logistic regression...")
    # Insert column of 1's first
    one_vector = np.ones((len(y),1))
    X = np.concatenate((one_vector, X), axis = 1)

    iterations = 10
    plt.figure()
    n_minibatches = np.arange(1,150, 10)
    for iteration in range(iterations):
        print(f"iteration {iteration}")
        acc_scores = np.zeros(len(n_minibatches))
        acc_train_scores = np.zeros(len(n_minibatches))
        for i, n in enumerate(n_minibatches):
            #logreg = LogReg(X_train, y_train)
            #logreg.sgd(n_epochs=epochs, n_minibatches=n)
            #pred = logreg.predict(X_test)
            #acc = accuracy(y_test, pred)
            acc, train_acc = CV_log_reg(X, y, epochs=epochs, mini_batches=n)
            acc_scores[i] = acc
            acc_train_scores[i] = train_acc
        plt.plot(n_minibatches, acc_scores, 'tab:blue')
        plt.plot(n_minibatches, acc_train_scores, 'tab:red')

    plt.xlabel("Number of minibatches in SGD")
    plt.ylabel("Accuracy score")
    plt.legend(['Test set', 'Train set'])
    #save_fig("LogRegcancer_accuracy_vs_n_minibatches")
    plt.show()

    opt_index = np.where(acc_scores == np.nanmax(acc_scores))
    opt_n_minibatches = n_minibatches[opt_index]

    return opt_n_minibatches


LogReg_optimize_n_minibatches()
