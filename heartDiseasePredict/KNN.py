import numpy as np
import pandas as pd
from collections import Counter

class KNN:
  def __init__(self, X, y, k=1):
    self.k = k
    self.NN = None
    self.distances = None
    self.X_train = X
    self.y_train = y


  def euclidian_dist(self, row_1, row_2, d):
    return np.absolute((pd.to_numeric(row_1.iloc[d]) - pd.to_numeric(row_2.iloc[d]))**2)


  def minkowski_dist(self, x_1, x_2):
    dims = len(x_1)
    distance = 0

    for d in range(dims):
      distance += self.euclidian_dist(x_1, x_2, d)

    return np.sqrt(distance)


  def test(self, X):
    predict = list()

    for id, test_point in X.iterrows():
      dists=list()

      dists = np.array([self.minkowski_dist(test_point, train_point) for id, train_point in self.X_train.iterrows()])

      self.distances = pd.DataFrame(data=dists, columns=['Distances'], index=self.y_train.index.array)

      self.NN = self.distances.sort_values(by='Distances')[:self.k]

      predicted_label = Counter(self.y_train[self.NN.index]).most_common(1)

      predict.append(predicted_label[0][0])

    return predict
