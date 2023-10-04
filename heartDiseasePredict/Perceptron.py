import numpy as np
from math import exp, pow
import ActivationFunctions as af

class Perceptron:

    def __init__(self, eta, threshold, n_epochs):
        self.learnRate = eta
        self.threshold = threshold
        self.activationFunction = af.stepFunctionBinary
        self.weights = None
        self.theta = None
        self.n_ep = n_epochs

    def train(self, X, y):
        nSamples, nFeatures = X.shape

        self.weights = np.random.random(nFeatures)
        self.theta = 0

        print(f'Pesos iniciais: {self.weights}, theta:{self.theta}\n\n', end='=-'*20)

        sqError = 1 + self.threshold
        epochs = 1

        while (sqError > 0.2 or self.n_ep >= epochs):

            sqError = 0
            print(f'\n>> Época nº {epochs}')

            for idx in range(len(X)):
                net = np.dot(X.iloc[idx], self.weights) + self.theta
                y_Obt = self.activationFunction(net)

                error = y.iloc[idx] - y_Obt
                sqError += pow(error, 2)

                dE2 = 2 * error * -1

                self.weights += -self.learnRate * dE2 * X.iloc[idx]
                self.theta += -self.learnRate * dE2

            sqError /= nSamples
            epochs += 1

            print(f'>> SqError = {sqError}\n', end='#'*40)

        print('\n', end='=-'*20)
        print(f'\nPesos finais: \n{self.weights}\n theta:{self.theta}\n', end='=-'*20)


    def test(self, X):
        net = np.dot(X, self.weights) + self.theta
        y_Obt = self.activationFunction(net)
        return y_Obt
