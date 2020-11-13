"""
A module implementation of the stochastic gradient descent algorithm for a 
feedforward neural network. 
"""
import random
import numpy as np
from numba import jit

from utils import sigmoid, sigmoid_derivative, MSE, quadratic_cost_function, lookahead

np.random.seed(2020) # set random seed for reproducibility 


class FFNN:
    def __init__(self, layers,
                epochs=10, batch_size=100, eta=0.1,
                activation_function=sigmoid, cost_function=quadratic_cost_function, last_layer_activation_function=None):
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
        last_layer_activation_function : function
            Optional differing activation function for the last layer. The default is the same as activation_function.
        """
        
        # variables
        self.n_layers = len(layers)
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.last_layer_activation_function = last_layer_activation_function if last_layer_activation_function else activation_function
              
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
        ys = []
        for i, X in enumerate(Xs):
            # Pretty progress indicator
            print(f"\033[A\nPredicting new data: {round((i+1)/len(Xs)*100,1)}%", end="")
            y = self.feed_forward(X)[-1]

            # Single values also get stored in a list, which we don't want. This is an ugly workaround, but it works
            if len(y) == 1: y = y[0]

            ys.append(y)

        # Indicate done
        print(" \033[32m✔ DONE\033[0m")
        return np.array(ys).T
    

    def feed_forward(self, X, include_weighted_inputs=False):
        """Function that feeds forward the input X and stores all the layers' activations in a list it returns.
        Optionally, it can also return the weighted inputs, which are useful in the backpropagation algorithm."""        
        activations = [X]
        weighted_inputs = []

        for bw_tuple, has_more in lookahead(zip(self.biases ,self.weights)):
            f = self.activation_function if has_more else self.last_layer_activation_function
            b, w = bw_tuple
            weighted_input = w @ X + b
            X = f(weighted_input)
            activations.append(X)
            weighted_inputs.append(weighted_input) 
            
        if include_weighted_inputs: return activations, weighted_inputs 
        return activations


    def SGD_train(self, train_data, report_convergence=False):
        """Function that trains the neural network using mini-batch stochastic gradient descencent.
        
        Arguments
        ---------
        train_data : list
            List of coupled input-output pairs (as tuples)"""
        
        n = len(train_data)
        
        batch_MSEs = []
        # Quadruple for loop!!! There's probably some possible numba-improvements here, but that'll have to wait
        for epoch in range(self.epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+self.batch_size] for k in range(0, n, self.batch_size)]

            for i, mini_batch in enumerate(mini_batches):
                # Pretty progress indicator
                print(f"\033[A\nTraining epoch {epoch+1} of {self.epochs}: {round((i+1)/len(mini_batches)*100,1)}%\t(Activation function: {self.activation_function.__name__})", end="")

                sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
                sum_nabla_w = [np.zeros(w.shape) for w in self.weights]
                
                MSEs = []
                for X, y in mini_batch:
                    if report_convergence:
                        nabla_b, nabla_w, mse = self.backpropagate(X, y, report_convergence=True)
                        MSEs.append(mse)
                    else:
                        nabla_b, nabla_w = self.backpropagate(X, y)
                    sum_nabla_b = [snb + nb for snb, nb in zip(sum_nabla_b, nabla_b)]
                    sum_nabla_w = [snw + nw for snw, nw in zip(sum_nabla_w, nabla_w)]

                self.weights = [w - self.eta/len(mini_batch) * snw for w, snw in zip(self.weights, sum_nabla_w)]
                self.biases = [b - self.eta/len(mini_batch) * snb for b, snb in zip(self.biases, sum_nabla_b)]
                if report_convergence:
                    batch_MSEs.append(sum(MSEs)/len(MSEs))
                
            # Indicate done and begin next epoch
            print(" \033[32m✔ DONE\033[0m")
                
        if report_convergence:
            return MSEs
        
    
    def backpropagate(self, X, y, report_convergence=False):
        """Considers the single input-output pair X and y, and returns the gradient using the backpropagation algorithm."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activations, weighted_inputs = self.feed_forward(X, include_weighted_inputs=True)
        
        # INFO1: * is the Hadamard product in numpy, doing element-wise multiplication. Beware that this is only supported by
        # Python 3.5 and above. Else, numpy.multiply might have to be used instead.
        delta = self.cost_function(activations[-1], y, derivative=True) * self.last_layer_activation_function(weighted_inputs[-1], derivative=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2].T)
         
        for l in range(2, self.n_layers):
            cost_derivative = self.weights[-l+1].T @ delta 
            # See INFO1 above.
            delta = cost_derivative * self.activation_function(weighted_inputs[-l], derivative=True)
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1].T)
            
        if report_convergence:
            return nabla_b, nabla_w, sum((activations[-1] - y)**2)
        return nabla_b, nabla_w