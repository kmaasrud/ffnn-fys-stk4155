from keras.datasets import mnist
import numpy as np
import pickle

(X_train_unraveled, y_train_unshaped), (X_test_unraveled, y_test_unshaped) = mnist.load_data()

# Reshape input data
X_train = np.zeros((X_train_unraveled.shape[0], 28 * 28)); X_test = np.zeros((X_test_unraveled.shape[0], 28 * 28))
for i, pic in enumerate(X_train_unraveled):
    X_train[i] = pic.ravel()
for i, pic in enumerate(X_test_unraveled):
    X_test[i] = pic.ravel()
    
# Reshape output data
y_train = np.zeros((y_train_unshaped.shape[0], 10)); y_test = np.zeros((y_test_unshaped.shape[0], 10))
for i, num in enumerate(y_train_unshaped):
    y_train[i] = np.array([int(bool(num - 1 == j)) for j in range(10)])
for i, num in enumerate(y_test_unshaped):
    y_test[i] = np.array([int(bool(num - 1 == j)) for j in range(10)])
    
with open("mnist_array_output.pickle", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
    
with open("mnist_scalar_output.pickle", "wb") as f:
    pickle.dump((X_train, X_test, y_train_unshaped, y_test_unshaped), f)