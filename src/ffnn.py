"""
A module implementation of the stochastic gradient descent algorithm for a 
feedforward neural network. 
"""
import random
import numpy as np
from numba import jit

from utils import sigmoid, sigmoid_derivative, MSE, quadratic_cost_function

np.random.seed(2020) # set random seed for reproducibility 


class FFNN:
    def __init__(self, layers, epochs=10, batch_size=100, eta=0.1, activation_function=sigmoid, cost_function=quadratic_cost_function):
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
        cost_function : function
            The cost function of the network. The default is utils.quadratic_cost_function.
        """
        
        # variables
        self.n_layers = len(layers)
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.activation_function = activation_function
        self.cost_function = cost_function
              
        # Initialize weights and biases with random values. Each weight-bias pair corresponds to a connection between
        # a layer and the neurons in the next layer.
        
        # Make vectors of biases that correspond to the size of each layer.
        # Each layer's bias-vector is an array on the form array([b1, b2, ...])
        self.biases = [np.random.randn(layer_size) for layer_size in self.layers[1:]]
        # Make weight-vectors for each neuron in each layer.
        # Each layer's weight-matrix is an array on the form array([[w11, w12, ...], [w21, w22, ...], ...])
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
    
    
    def predict(self, Xs):
        """Function that returns the output of the network, given inputs X."""
        y = []
        for X in Xs:
            tmp = self.feed_forward(X)[-1]

            # Single values get stored in a list, which I don't want. This is an ugly workaround
            if len(tmp) == 1: tmp = tmp[0]

            y.append(tmp)
        return np.array(y).T
    

    def feed_forward(self, X, include_weighted_inputs=False):
        """Function that feeds forward the input X and stores all the layers' activations in a list it returns.
        Optionally, it can also return the weighted inputs, which are useful in the backpropagation algorithm."""        
        activations = [X]
        if include_weighted_inputs: weighted_inputs = []
        for b, w in zip(self.biases,self.weights):
            weighted_input = w @ X + b
            X = self.activation_function(weighted_input)
            activations.append(X)
            if include_weighted_inputs: weighted_inputs.append(weighted_input) 
            
        if include_weighted_inputs: return activations, weighted_inputs 
        return activations


    def SGD_train(self, train_data):
        """Function that trains the neural network using mini-batch stochastic gradient descencent.
        
        Arguments
        ---------
        train_data : list
            List of coupled input-output pairs (as tuples)"""
        
        n = len(train_data)
        
        # Quadruple for loop!!! There's probably some possible numba-improvements here, but that'll have to wait
        for epoch in range(self.epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+self.batch_size] for k in range(0, n, self.batch_size)]

            for i, mini_batch in enumerate(mini_batches):
                # Pretty progress indicator
                print(f"\033[A\nTraining epoch {epoch+1} of {self.epochs}: {round((i+1)/len(mini_batches)*100,1)}%", end="")

                sum_nabla_b = [np.empty(b.shape) for b in self.biases]
                sum_nabla_w = [np.empty(w.shape) for w in self.weights]
                
                for X, y in mini_batch:
                    nabla_b, nabla_w = self.backpropagate(X, y)
                    sum_nabla_b = [snb + nb for snb, nb in zip(sum_nabla_b, nabla_b)]
                    sum_nabla_w = [snw + nw for snw, nw in zip(sum_nabla_w, nabla_w)]

                self.weights = [w - self.eta/len(mini_batch) * snw for w, snw in zip(self.weights, sum_nabla_w)]
                self.biases = [b - self.eta/len(mini_batch) * snb for b, snb in zip(self.biases, sum_nabla_b)]
                
            # Indicate done and begin next epoch
            print(" \033[32m✔ DONE\033[0m")
        
    
    def backpropagate(self, X, y):
        """Considers the single input-output pair X and y, and returns the gradient using the backpropagation algorithm."""
        nabla_b = [np.empty(b.shape) for b in self.biases]
        nabla_w = [np.empty(w.shape) for w in self.weights]
        
        activations, weighted_inputs = self.feed_forward(X, include_weighted_inputs=True)
        
        # INFO1: * is the Hadamard product in numpy, doing element-wise multiplication. Beware that this is only supported by
        # Python 3.5 and above. Else, numpy.multiply might have to be used instead.
        delta = self.cost_function(activations[-1], y, derivative=True) * self.activation_function(weighted_inputs[-1], derivative=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2].T)
         
        for l in range(2, self.n_layers):
            cost_derivative = np.dot(self.weights[-l+1].transpose(), delta) 
            # See INFO1 above.
            delta = cost_derivative * self.activation_function(weighted_inputs[-l], derivative=True)
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1].T)
            
        return nabla_b, nabla_w