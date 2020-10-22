"""
A module implementation of the stochastic gradient descent algorithm for a 
feedforward neural network. 
"""
import random
import numpy as np

from utils import sigmoid

np.random.seed(2020) # set random seed for reproducibility 


class FFNN():
    def __init__(self, X, y,
            n_layers,
            nodes,
            n_classes,
            epochs=10,
            batch_size=100,
            eta=0.1,
            problem_type="classification",
            activation="sigmoid"):
        """
        Initialize the feedforward neural network. 

        Parameters
        ----------
        X : ndarray
            x-data.
        y : ndarray
            y-data.
        n_layers : int
            Number of hidden layers.
        nodes : list
            Cointains the number of nodes in the hidden layers. E.g. if 
            n_nodes is [3, 2, 4] we have a three-layer network, with 3 neurons 
            in the first layer, 2 neurons in the second layer and 4 neurons in 
            the third layer. 
        n_classes : int
            Number of output classes
        epochs : int, optional
            The default is 10.
        batch_size : int, optional
            Size of mini-batches for the stochastic gradient descent. 
            The default is 100.
        eta : float, optional
            Learning rate. The default is 0.1.
        problem_type : str, optional
            Problemtype The default is "classification".
        activation : str, optional
            Activation functions: sigmoid, ReLU and leaky_ReLU. The default is "sigmoid".
        """
        
        # data
        self.X = X
        self.y = y
        
        # variables
        self.n_inputs, self.n_features = X.shape
        self.n_layers = n_layers
        self.nodes = nodes 
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.problem_type = problem_type
        self.activation = activation
              
        # weights and biases
        self.weights_and_biases()
        
    def weights_and_biases(self):
        """
        Function that initializes the weights and biases randomly, using a 
        Gaussian distribution with mean 0, and variance 1 over the square   
        root of the number of weights connecting to the same neuron. The first
        layer is assumed to be an input layer, and by convention, no 
        biases are set for those neurons.  (Since biases are only used in 
        computing the outputs from later layers.)
        """
        self.biases = [np.random.randn(y, 1) for y in self.nodes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.nodes[:-1], self.nodes[1:])]
    
    
    def feed_forward(self, X):
        """
        Function that returns the output of the network if ``x``is the input.
        """
        for b, w in zip(self.biases, self.weights):
            X = sigmoid(np.dot(w, X)+b)
        return X


    def SDG(self, X_train, y_train, epochs, batch_size, eta):
        """" 
        Function that trains the neural network using mini-batch stochastic 
        gradient descencent. 
        
        TODO:
            Add test_data as optional argument and evaluate the network against 
            the test data after each epoch if test data provided. 
        """
        
        train_data = tuple(zip(X_train, y_train))
        
        n = len(train_data)
        
        for _ in range(epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    
    def update_mini_batch(self, batch, eta):
        """
        Function that updates the networks's weights and biases by applying 
        gradient descent using backpropagation to a singel mini-batch. 
        """
        
        nabla_b = [np.empty(b.shape) for b in self.biases]
        nabla_w = [np.empty(w.shape) for w in self.weights]
        
        for X, y in batch:
            d_nabla_b, d_nabla_w = self.backpropagate(X, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, d_nabla_w)]

        self.weights = [w - (eta/len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        
    
    def backpropagate(self, X, y):
        
        nabla_b = [np.empty(b.shape) for b in self.biases]
        nabla_w = [np.empty(w.shape) for w in self.weights]
        
        # feedforward
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # backward pass
        cost_derivative = activations[-1] - y
        delta = cost_derivative * self.sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # l = 1 is the last layer of neurons, l = 2 is the
        # second-last layer ...
        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)