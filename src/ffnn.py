"""
A module implementation of the stochastic gradient descent algorithm for a 
feedforward neural network. 
"""
import random
import numpy as np

from utils import sigmoid, sigmoid_derivative, MSE

np.random.seed(2020) # set random seed for reproducibility 


class FFNN:
    def __init__(self, layers, epochs=10, batch_size=100, eta=0.1, activation_function=sigmoid):
        """Initialize the feedforward neural network. 

        Arguments
        ----------
        layers : list
            Cointains the number of nodes in the layers. E.g. if 
            layers is [3, 2, 4] we have a three layer network, with 3 neurons 
            in the first (input) layer, 2 neurons in the second (hidden) layer
            and 4 neurons in the third (output) layer. 
        
        Keyword arguments
        -----------------
        epochs : int
            Number of epochs. The default is 10.
        batch_size : int
            Size of mini-batches for the stochastic gradient descent. 
            The default is 100.
        eta : float
            Learning rate. The default is 0.1.
        activation_function : function
            Activation function. The default is utils.sigmoid.
        """
        
        # variables
        self.n_layers = len(layers)
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.activation_function = activation_function
              
        # Initialize weights and biases with random values. Each weight-bias pair corresponds to a connection between
        # a layer and the neurons in the next layer.
        
        # Make vectors of biases that correspond to the size of each layer.
        # Each layer's bias-vector is an array on the form array([b1, b2, ...])
        self.biases = [np.random.randn(layer_size) for layer_size in self.layers[1:]]
        # Make weight-vectors for each neuron in each layer.
        # Each layer's weight-matrix is an array on the form array([[w11, w12, ...], [w21, w22, ...], ...])
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
    
    
    def predict(self, X):
        """Function that returns the output of the network, given an input X."""
        return self.feed_forward(X)[-1]
    

    def feed_forward(self, X, include_weighted_inputs=False):
        """Function that feeds forward the input X and stores all the layers' activations in a list it returns.
        Optionally, it can also return the weighted inputs, which are useful in the backpropagation algorithm."""        
        activations = [X]
        if include_weighted_inputs: weighted_inputs = []
        for b, w in zip(self.biases,self.weights):
            weighted_input = np.dot(w, X) + b
            X = self.activation_function(weighted_input)
            activations.append(X)
            if include_weighted_inputs: weighted_inputs.append(weighted_input) 
            
        if include_weighted_inputs:
            return activations, weighted_inputs 
        return activations


    def SGD_train(self, X_train, y_train):
        """ Function that trains the neural network using mini-batch stochastic gradient descencent."""
        
        train_data = list(zip(X_train, y_train))
        
        n = len(train_data)
        
        # Quadruple for loop!!! There's probably some possible numba-improvements here, but that'll have to wait
        for _ in range(self.epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+self.batch_size] for k in range(0, n, self.batch_size)]

            for mini_batch in mini_batches:
                sum_nabla_b = [np.empty(b.shape) for b in self.biases]
                sum_nabla_w = [np.empty(w.shape) for w in self.weights]
                
                for X, y in mini_batch:
                    nabla_b, nabla_w = self.backpropagate(X, y)
                    sum_nabla_b = [snb + nb for snb, nb in zip(sum_nabla_b, nabla_b)]
                    sum_nabla_w = [snw + nw for snw, nw in zip(sum_nabla_w, nabla_w)]

                self.weights = [w - self.eta/len(mini_batch) * snw for w, snw in zip(self.weights, sum_nabla_w)]
                self.biases = [b - self.eta/len(mini_batch) * snb for b, snb in zip(self.biases, sum_nabla_b)]
        
    
    def backpropagate(self, X, y):
        nabla_b = [np.empty(b.shape) for b in self.biases]
        nabla_w = [np.empty(w.shape) for w in self.weights]
        
        activations, weighted_inputs = self.feed_forward(X, include_weighted_inputs=True)
        
        # This is a hard-coded derivative, assuming a cost function that reads 1/2 (x-y)^2 (quadratic cost, divided by 2)
        cost_derivative = activations[-1] - y
        
        # INFO1: * is the Hadamard product in numpy, doing element-wise multiplication. Beware that this is only supported by
        # Python 3.5 and above. Else, numpy.multiply might have to be used instead.
        delta = cost_derivative * self.activation_function(weighted_inputs[-1], derivative=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
         
        for l in range(2, self.n_layers):
            cost_derivative = np.dot(self.weights[-l+1].transpose(), delta) 
            # See INFO1 above.
            delta = cost_derivative * self.activation_function(weighted_inputs[-l], derivative=True)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return nabla_b, nabla_w