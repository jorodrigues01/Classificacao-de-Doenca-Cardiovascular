import numpy as np
from math import exp, pow
import ActivationFunctions as af

class Perceptron:

    def __init__(self, eta=0.1, threshold=exp(-2)):
        self.learnRate = eta
        self.threshold = threshold
        self.activationFunction = None
        self.weights = None
        self.theta = None

    def train(self, X, y):
        nFeatures = len(X.columns)

        self.weights = np.random.random(nFeatures)
        self.theta = 0

        if [0, 1] not in y.unique():
            self.activationFunction = af.stepFunctionBipolar
        else:
            self.activationFunction = af.stepFunctionBinary


        sqError = 1 + self.threshold

        while(sqError < self.threshold):

            sqError = 0

            for idx, X_i in enumerate(X):
                net = np.dot(X_i, self.weights) + self.theta
                y_Obt = self.activationFunction(net)

                error = pow((y[idx]-y_Obt), 2)
                sqError = sqError + error

                devE2 =


    def test(self, X):
        net = np.dot(X, self.weights) + self.theta
        y_Obt = self.activationFunction(net)
