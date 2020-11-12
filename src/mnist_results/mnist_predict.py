import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ffnn import FFNN
from utils import split_and_scale, ReLU, leaky_ReLU

out_dir = "../../doc/assets/"

with open("mnist_scalar_output.pickle", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

load_pickle = input("Load pickled networks (y/n)? ") == "y"
do_plot = input("Plot the predicted data? (y/n)?") == "y"
if load_pickle:
    with open("network.pickle", "rb") as f:
        nn_sigmoid = pickle.load(f)
else:
    do_pickle = input("Pickle the neural networks (y/n)? ") == "y"
    layer_structure = [X_train.shape[1], 200, 50, y_train.shape[1]]
    training_data = list(zip(X_train, y_train))

    nn_sigmoid = FFNN(layer_structure, epochs=20)
    nn_sigmoid.SGD_train(training_data)

    if do_pickle:
        with open("network.pickle", "wb") as f:
            pickle.dump(nn_sigmoid, f)
            
if do_plot:
    y_predict_sigmoid= nn_sigmoid.predict(X_test)

    # Reshape from vector y values into scalars
    # y_predict_sigmoid = np.zeros(y_predict_sigmoid_vector_form.shape[1])
    # for i, prediction in enumerate(y_predict_sigmoid_vector_form.T):
    #     y_predict_sigmoid[i] = np.argmax(prediction)
                
    with open("mnist_X_test_unraveled.pickle", "rb") as f:
        X_test_imgs = pickle.load(f)

    for i in range(9):  
        plt.subplot(330 + 1 + i)
        plt.imshow(X_test_imgs[i], cmap=plt.get_cmap('gray'))
        plt.title(y_predict_sigmoid[i])

    plt.show()
