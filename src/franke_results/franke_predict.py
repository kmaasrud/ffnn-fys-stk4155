import numpy as np
import matplotlib.pyplot as plt

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from franke import franke
from utils import design_matrix, split_and_scale
from ffnn import FFNN

out_dir = "../../doc/assets/"

N = 500
deg = 12
lmb = 2
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
X = design_matrix(x, y, deg)
print(X.shape)
Y = franke(x, y, noise_sigma=0.1, noise=True)
print(Y.shape)
X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)

nn = FFNN([X.shape[1], 100, 10, 1])
nn.SGD_train(X_train, y_train)

nn_predict = nn.predict(X_test)

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