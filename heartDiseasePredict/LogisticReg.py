import numpy as np

class LogisticReg:
    def __init__(self, eta=0.01, n_epochs=1000):
      self.learnRate = eta
      self.n_epochs = n_epochs
      self.weights = None
      self.bias = None


    def train(self, X, y):
      n_samples, n_features = X.shape

      self.weights = np.random.rand(n_features)
      self.bias = np.random.rand()

      for _ in range(self.n_epochs):
        for id in range(n_samples):
          linear = np.dot(X.iloc[id], self.weights) + self.bias
          y_pred = self.sigmoid(linear)

          error = y_pred - y.iloc[id]

          dweights = (1/n_samples) * np.dot(2*X.iloc[id], error)
          dbias = (1/n_samples) * 2 * error

          self.weights += -self.learnRate * dweights
          self.bias += -self.learnRate * dbias


    def test(self, X):
      linear = np.dot(X, self.weights) + self.bias
      prediction = self.sigmoid(linear)

      return np.where(prediction > .5, 1, 0)


    def sigmoid(self, linear):
      return 1/(1 + np.exp(-linear))

