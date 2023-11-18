import numpy as np
import pandas as pd
from DTree.Node import Node


class DecisionTree():
    def __init__(self, min_split=2, max_depth=2):
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth


    def grow_decisionTree(self, X, y, curr_depth=0):
        num_samples, num_features = np.shape(X)
        n_classes = len(np.unique(y))

        if num_samples < self.min_split or curr_depth > self.max_depth or n_classes == 1:
          leaf_value = self.calculate_leaf_value(y)
          return Node(value=leaf_value)

        best_split = self.get_best_split(X, y, num_samples, num_features)

        if best_split["info_gain"] > 0:
            left_subtree = self.grow_decisionTree(best_split["frame_left"],
                                                best_split["y_left"], curr_depth+1)

            right_subtree = self.grow_decisionTree(best_split["frame_right"],
                                                 best_split["y_right"], curr_depth+1)

            return Node(best_split["feature_index"], best_split["threshold"],
                        left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)


    def get_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        max_info_gain = -1

        for feature_index in range(num_features):
            feature_values = X.iloc[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                ids_left, ids_right = self.split(X, feature_index, threshold)

                if len(ids_left) == 0 and len(ids_right) == 0:
                    continue

                frame_left = pd.DataFrame([X.iloc[id] for id in ids_left], index=ids_left)
                frame_right = pd.DataFrame([X.iloc[id] for id in ids_right], index=ids_right)
                left_y, right_y = y.iloc[ids_left], y.iloc[ids_right]

                curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["frame_left"] = frame_left
                    best_split["frame_right"] = frame_right
                    best_split["y_left"] = left_y
                    best_split["y_right"] = right_y
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain

        return best_split


    def split(self, X, feature_index, threshold):
        ids_left = np.array([id for id in range(len(X)) if X.iloc[id, feature_index] <= float(threshold)])
        ids_right = np.array([id for id in range(len(X)) if X.iloc[id, feature_index] > float(threshold)])
        return ids_left, ids_right


    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain


    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0

        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy


    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0

        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini


    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)


    def fit(self, X, y):
        self.root = self.grow_decisionTree(X, y)


    def predict(self, X):
        return np.array([self.make_prediction(X.iloc[idx], self.root) for idx in range(len(X))])


    def make_prediction(self, x, tree):
        if tree.value!=None:
            return tree.value

        feature_val = x.iloc[tree.feature_index]

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)

