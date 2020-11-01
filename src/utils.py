import numpy as np


# Different functions for use as activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """ Derivative of the sigmoid function. """
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return x if x > 0 else 0

def heaviside(x):
    return 1 if x >= 0 else 0

def leaky_ReLU(x):
    """ The Leaky ReLUfunction. """
    idx = np.where(x <= 0)
    x[idx] = 0.01 * x
    return x

def leaky_ReLU_derivative(x):
    """ Derivative of the Leaky ReLU function. """
    idx1 = np.where(x < 0)
    x[idx1] = 0.01

    idx2 = np.where(x > 0)
    x[idx2] = 1.0
    return x

def accuracy(y, y_pred):
    numerator=np.sum(y == y_pred)
    return numerator/len(y)
