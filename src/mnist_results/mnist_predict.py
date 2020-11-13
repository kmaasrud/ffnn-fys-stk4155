import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ffnn import FFNN
from utils import split_and_scale, ReLU, leaky_ReLU

out_dir = "../../doc/assets/"
pickle_file = "network.pickle"

with open("mnist_array_output.pickle", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

load_pickle = input("Load pickled networks (y/n)? ") == "y"
do_plot = input("Plot the predicted data? (y/n)? ") == "y"
if load_pickle:
    with open(pickle_file, "rb") as f:
        nn_relu = pickle.load(f)
else:
    do_pickle = input("Pickle the neural networks (y/n)? ") == "y"
    layer_structure = [X_train.shape[1], 200, y_train.shape[1]]
    training_data = list(zip(X_train, y_train))

    nn_relu = FFNN(layer_structure, activation_function=ReLU)
    nn_relu.SGD_train(training_data)

    if do_pickle:
        with open(pickle_file, "wb") as f:
            pickle.dump(nn_relu, f)
            
if do_plot:
    y_predict_relu_vector_form = nn_relu.predict(X_test)

    # Reshape from vector y values into scalars
    y_predict_relu = np.zeros(y_predict_relu_vector_form.shape[1])
    for i, prediction in enumerate(y_predict_relu_vector_form.T):
        y_predict_relu[i] = np.argmax(prediction)
                
    with open("mnist_X_test_unraveled.pickle", "rb") as f:
        X_test_imgs = pickle.load(f)

    j = random.randint(0,9990)
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_test_imgs[j+i], cmap=plt.get_cmap('gray'))
        plt.title(y_predict_relu[j+i])
        plt.xticks([]); plt.yticks([])

    plt.savefig(os.path.join(out_dir, "nn_sigmoid_mnist_plot.png"))
