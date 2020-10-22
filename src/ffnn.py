"""
A module implementation of the stochastic gradient descent algorithm for a 
feedforward neural network. 
"""
import random
import numpy as np

from utils import sigmoid

np.random.seed(2020) # set random seed for reproducibility 


class FFNN:
    def __init__(self, X, y, n_layers, hidden_layers,
            # kwargs
            epochs=10,
            batch_size=100,
            eta=0.1,
            activation_function=sigmoid):
        """Initialize the feedforward neural network. 

        Arguments
        ----------
        X : ndarray
            Input data.
        y : ndarray
            y-data.
        hidden_layers : list
            Cointains the number of nodes in the hidden layers. E.g. if 
            n_nodes is [3, 2, 4] we have a five network (including IO), with 3 neurons 
            in the first hidden layer, 2 neurons in the second hiddenlayer and 4 neurons
            in the third hidden layer. 
        
        Keyword arguments
        -----------------
        epochs : int
            The default is 10.
        batch_size : int
            Size of mini-batches for the stochastic gradient descent. 
            The default is 100.
        eta : float
            Learning rate. The default is 0.1.
        activation : function
            Activation function.
            The default is utils.sigmoid.
        """
        
        # data
        self.X = X
        self.y = y
        
        # variables
        self.n_inputs, self.n_features = X.shape
        self.n_outputs, self.n_output_features = y.shape
        self.n_layers = len(hidden_layers) + 2
        self.nodes = [self.n_features] + hidden_layers + [self.n_output_features]
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.activation_function = activation_function
              
        # Initialize weights and biases with random values. Each weight-bias pair corresponds to a connection between
        # a layer and the neurons in the next layer.
        
        # Make vectors of biases that correspond to the size of each layer.
        # Each layer's bias-vector is an array on the form array([b1, b2, ...])
        self.biases = [np.random.randn(layer_size) for layer_size in self.nodes[1:]]
        # Make weight-vectors for each neuron in each layer.
        # Each layer's weight-matrix is an array on the form array([[w11, w12, ...], [w21, w22, ...], ...])
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.nodes[:-1], self.nodes[1:])]
        # TODO: Why divide by sqrt(x)? Shouldn't this just initialize randomly?
    
    
    def feed_forward(self):
        """Function that returns the output of the network if ``x``is the input."""
        X = self.X
        for b, w in zip(self.biases, self.weights):
            X = self.activation_function(np.dot(w, X)+b)
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