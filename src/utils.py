import numpy as np


# Different functions for use as activation functions
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid_derivative(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """ Derivative of the sigmoid function. """
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x, derivative=False):
    if derivative:
        return heaviside(x)
    return x if x > 0 else 0

def heaviside(x):
    return 1 if x >= 0 else 0

def leaky_ReLU(x, derivative=False):
    """ The Leaky ReLUfunction. """
    if derivative:
        return leaky_ReLU_derivative(x)
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
    
def MSE(x, y):
    """The mean squared error function.
    The result is divided by 2 to make sure the derivative of the cost function is easily written as just (x - y)"""
    assert len(x) == len(y), "The arrays need to have the same length"
    for xval, yval in zip(x, y):
        s += (xval - yval)**2
    return s / (2 * len(x))