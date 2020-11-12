import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from franke import franke
from utils import design_matrix, split_and_scale, ReLU, leaky_ReLU
from ffnn import FFNN

out_dir = "../../doc/assets/"

N = 100
deg = 8
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
X = design_matrix(x, y, deg)
Y = franke(x, y, noise_sigma=0.1, noise=True)
X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)

load_pickle = input("Load pickled networks (y/n)? ") == "y"
if load_pickle:
    with open("network.pickle", "rb") as f:
        nn_sigmoid, nn_relu, nn_leaky = pickle.load(f)
else:
    do_pickle = input("Pickle the neural networks (y/n)? ") == "y"
    layer_structure = [X.shape[1], 50, 1]
    training_data = list(zip(X_train, y_train))

    nn_sigmoid = FFNN(layer_structure, epochs=20)
    nn_sigmoid.SGD_train(training_data)

    nn_relu = FFNN(layer_structure, epochs=20, activation_function=ReLU)
    nn_relu.SGD_train(training_data)

    nn_leaky = FFNN(layer_structure, epochs=20, activation_function=leaky_ReLU)
    nn_leaky.SGD_train(training_data)

    if do_pickle:
        with open("network.pickle", "wb") as f:
            pickle.dump((nn_sigmoid, nn_relu, nn_leaky), f)


nn_sigmoid_predict = nn_sigmoid.predict(X_test)
nn_relu_predict = nn_relu.predict(X_test)
nn_leaky_predict = nn_leaky.predict(X_test)

fig = plt.figure()
ax = fig.gca(projection='3d')

def plot(X, y, ax, title):
    ax.plot_trisurf(X[:,1], X[:,2], y, linewidth=0, antialiased=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.savefig(os.path.join(out_dir, title))
    plt.cla()
    
plot(X_test, y_test, ax, "actual_franke_plot.png")
plot(X_test, nn_sigmoid_predict, ax, "nn_sigmoid_franke_plot.png")
plot(X_test, nn_relu_predict, ax, "nn_relu_franke_plot.png")
plot(X_test, nn_leaky_predict, ax, "nn_leaky_franke_plot.png")