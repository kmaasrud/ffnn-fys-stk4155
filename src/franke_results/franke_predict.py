import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from franke import franke
from utils import design_matrix, split_and_scale, ReLU
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

load_pickle = input("Load pickled network (y/n)? ") == "y"
if load_pickle:
    with open("network.pickle", "rb") as f:
        nn = pickle.load(f)
else:
    do_pickle = input("Pickle the neural network (y/n)? ") == "y"
    nn = FFNN([X.shape[1], 50, 1], epochs=20)
    nn.SGD_train(list(zip(X_train, y_train)))

    if do_pickle:
        with open("network.pickle", "wb") as f:
            pickle.dump(nn, f)


nn_predict = nn.predict(X_test)
print(nn_predict)

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
plot(X_test, nn_predict, ax, "nn_franke_plot.png")