import numpy as np
from numba import jit
from sklearn import model_selection, preprocessing


# Different functions for use as activation functions
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid_derivative(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """ Derivative of the sigmoid function. """
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(x, derivative=False):
    if derivative:
        return np.heaviside(x, 1)
        
    if type(x) == np.ndarray:
        x[x<0] = 0
        return x
    return x if x > 0 else 0

def leaky_ReLU(x, derivative=False):
    """ The Leaky ReLUfunction. """
    if derivative:
        return leaky_ReLU_derivative(x)
    x[x<=0] = 0.01 * x[x<=0]
    return x

def leaky_ReLU_derivative(x):
    """ Derivative of the Leaky ReLU function. """
    x[x<0] = 0.01
    x[x>0] = 1.0
    return x


def quadratic_cost_function(x, y, derivative=False):
    """Quadratic cost function, divided in half to provide a simple derivative, which can optionally be returned."""
    if derivative:
        return x - y
    return 0.5 * (x - y)**2


def MSE(x, y):
    """The mean squared error function.
    The result is divided by 2 to make sure the derivative of the cost function is easily written as just (x - y)"""
    assert len(x) == len(y), "The arrays need to have the same length"
    for xval, yval in zip(x, y):
        s += (xval - yval)**2
    return s / (2 * len(x))

def accuracy(y, y_pred):
    y=np.ravel(y)
    y_pred=np.ravel(y_pred)
    numerator=np.sum(y == y_pred)
    return numerator/len(y)

@jit(nopython=True)
def design_matrix(x, y, d):
    """Function for setting up a design X-matrix with rows [1, x, y, x², y², xy, ...]
    Input: x and y mesh, argument d is the degree.
    """

    if len(x.shape) > 1:
    # reshape input to 1D arrays (easier to work with)
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((d+1)*(d+2)/2)	# number of elements in beta
    X = np.ones((N,p))

    for n in range(1,d+1):
        q = int((n)*(n+1)/2)
        for m in range(n+1):
            X[:,q+m] = x**(n-m)*y**m

    return X

def split_and_scale(X, y, test_size=0.2):
    """
    Function that splits the data in test and training data and scale the data.
    4/5 of the data is training data.
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False)."""
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False