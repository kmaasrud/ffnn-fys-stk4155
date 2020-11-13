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

layer_structure = [X.shape[1], 50, 1]
training_data = list(zip(X_train, y_train))

nn1 = FFNN(layer_structure, eta=0.1)
eta01_MSE = nn1.SGD_train(training_data, report_convergence=True)
nn2 = FFNN(layer_structure, eta=0.01)
eta001_MSE = nn2.SGD_train(training_data, report_convergence=True)
nn2 = FFNN(layer_structure, eta=0.001)
eta0001_MSE = nn2.SGD_train(training_data, report_convergence=True)

batches = list(range(len(eta01_MSE)))
plt.plot(batches, eta01_MSE, label=r"$\eta = 0.1$")
plt.plot(batches, eta001_MSE, label=r"$\eta = 0.01$")
plt.plot(batches, eta0001_MSE, label=r"$\eta = 0.001$")
plt.legend()
plt.show()