# Import modules and neededfunctions
import numpy as np
from utils import sigmoid, accuracy
from scipy.special import expit


from sklearn.preprocessing import scale


# set random seed for reproducibility
np.random.seed(2020)

class LogReg:
    def __init__(self, X_train, y_train, predictor_names = None):
        self.y_train = y_train
        self.X_train = X_train

    def SGD_logreg(self, epochs, mini_batches):
        n = len(self.X_train)

        batch_size = int(n/mini_batches)

        beta = np.random.randn(len(self.X_train[0]), 1)
        train_data = [self.X_train, self.y_train]

        for epoch in range(epochs):
            for mini_batch in range(mini_batches):
                index=np.random.randint(mini_batches)

                start_calc=index*batch_size
                end_calc=index*batch_size+batch_size

                x_temp=self.X_train[start_calc:end_calc]
                y_temp=self.y_train[start_calc:end_calc]

                eksp=np.dot(x_temp,beta)
                sg = sigmoid(eksp)
                temp1=np.transpose(x_temp)
                grad = -np.dot(temp1,y_temp-sg)
                l = self.learning_rate(epoch*mini_batches + mini_batch)
                beta=beta-l*grad
                self.beta = beta
        self.beta=beta
        return beta

    def predict(self,X):
        eksp2=np.dot(X,self.beta)
        #Rounds the elements toward 1 or 0
        prediction = np.round(sigmoid(eksp2))
        #If i ravel it, the accurcay drops about 10%
        prediction=np.ravel(prediction)
        return prediction

    def learning_rate(self, t, t0=5, t1=50):
        return t0/(t+t1)


def k_folds(n, k=5):
    """Imports k_fold function from project 1
    Returns a list with k lists of indexes to be used for test-sets in
    cross-validation. The indexes range from 0 to n, and are randomly distributed
    to the k groups s.t. the groups do not overlap"""
    indexes = np.arange(n)

    min_size = int(n/k)
    extra_points = n % k

    folds = []
    start_index = 0
    for i in range(k):
        if extra_points > 0:
            test_indexes = indexes[start_index: start_index + min_size + 1]
            extra_points -= 1
            start_index += min_size + 1
        else:
            test_indexes = indexes[start_index: start_index + min_size]
            start_index += min_size
        folds.append(test_indexes)
    return folds


def CV_log_reg(x, y, k=5,epochs=100,mini_batches=30):
    """Function that performs k-fold cross-validation with logistic regression
    on breastcancer data."""

    #Defining empty lists to save the accuracy values
    test_error = []
    train_error = []

    # Length of target list
    n = len(y)
    #Length of each fold
    i = int(n/k)
    #Calling the k_fold function to produce the folds
    test_folds = k_folds(n, k=k)

    #Looping over the test folds
    for i in test_folds:
        m = len(i)
        y_test = y[i]
        y_train = y[np.delete(np.arange(n), i)]
        x_test = x[i]
        x_train = x[np.delete(np.arange(n), i)]

        # Performs logistic regression on both the test and training set
        log_reg_code = LogReg(x_train, y_train)
        log_reg_code.SGD_logreg(epochs, mini_batches)
        pred_test = log_reg_code.predict(x_test)
        pred_train = log_reg_code.predict(x_train)

        test_error.append(accuracy(y_test, pred_test))
        train_error.append(accuracy(y_train, pred_train))

    test_accuracy = np.mean(test_error)
    train_accuracy = np.mean(train_error)

    return test_accuracy, train_accuracy
