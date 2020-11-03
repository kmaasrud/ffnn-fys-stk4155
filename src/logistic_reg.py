# Import modules and neededfunctions
import numpy as np
from utils import sigmoid

# set random seed for reproducibility
np.random.seed(2020)

class LogReg:
    def __init__(self, X_train, y_train, predictor_names = None):
        self.y_train = y_train
        self.X_train = X_train

    def SGD_logreg(self, epochs, mini_batches):
        X_train = self.X_train
        y_train = self.y_train

        n = len(self.X_train)

        batch_size = int(n/mini_batches)

        beta = np.random.randn(len(self.X_train[0]), 1)
        train_data = [self.X_train, self.y_train]

        for epoch in range(epochs):
            for mini_batch in range(mini_batches):
                index=np.random.randint(mini_batches)

                start_calc=index*batch_size
                end_calc=index*batch_size+batch_size

                x_temp=self.X_train[start_calc:end_calc]
                y_temp=self.y_train[start_calc:end_calc]

                eksp=np.dot(x_temp,beta)
                sg = sigmoid(eksp)
                temp1=np.transpose(x_temp)
                grad = -np.dot(temp1,y_temp-sg)
                l = self.learning_rate(epoch*mini_batches + mini_batch)
                beta=beta-l*grad
                self.beta = beta
        self.beta=beta
        return beta

    def predict(self,X):
        eksp2=np.dot(X,self.beta)
        #Rounds the elements toward 1 or 0
        prediction = np.round(sigmoid(eksp2))

        #If I ravel, the accuracy decreases with about 30%. Remember to double check this
        #prediction=np.ravel(prediction)
        return prediction

    def learning_rate(self, t, t0=5, t1=50):
        return t0/(t+t1)
