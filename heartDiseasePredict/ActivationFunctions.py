import numpy as np

def stepFunctionBinary(x):
    return np.where(x >= 0, 1, 0)

def stepFunctionBipolar(x):
    return np.where(x >= 0, 1, -1)