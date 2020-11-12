import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.datasets import mnist

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ffnn import FFNN
from utils import split_and_scale, ReLU, leaky_ReLU

out_dir = "../../doc/assets/"

with open("mnist_array_output.pickle", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

load_pickle = input("Load pickled networks (y/n)? ") == "y"
if load_pickle:
    with open("network.pickle", "rb") as f:
        nn_sigmoid, = pickle.load(f)
else:
    do_pickle = input("Pickle the neural networks (y/n)? ") == "y"
    layer_structure = [X_train.shape[1], 1000, y_train.shape[1]]
    training_data = list(zip(X_train, y_train))

    nn_sigmoid = FFNN(layer_structure, epochs=5)
    nn_sigmoid.SGD_train(training_data)

    if do_pickle:
        with open("network.pickle", "wb") as f:
            pickle.dump(nn_sigmoid, f)

# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#     plt.title(i)

# plt.show()