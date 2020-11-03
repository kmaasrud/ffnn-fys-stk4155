# Import scikit regression tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

np.random.seed(2020) # set random seed for reproducibility
"""
The class is not done yet
"""
class LogReg:
    def __init__(self, X, y, predictor_names = None):
        self.y = y
        self.X = X

    def SGD(self, X_train, y_train, epochs, batch_size):

        train_data = tuple(zip(X_train, y_train))
        n = len(train_data)

        beta = np.random.randn(len(X[0]), 1)

        for _ in range(epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

"""
        for epoch in range(epochs):               # epoch
            for i in range(n_minibatches):          # minibatches
                random_index = np.random.randint(n_minibatches)

                p = 1/(1 + np.exp(-xi @ beta))
                gradient = -xi.T @ (yi - p)
                l = self.learning_rate(epoch*n_minibatches + i)
                beta = beta - l * gradient
                self.beta = beta

        self.beta = beta
"""
        return beta

    def predict(self, x, beta=None):
        if beta == None:
            beta = self.beta
        pred = np.round(1/(1 + np.exp(-x@beta))).ravel()
        return pred

    def learning_rate(self, t, t0=5, t1=50):
        return t0/(t+t1)
