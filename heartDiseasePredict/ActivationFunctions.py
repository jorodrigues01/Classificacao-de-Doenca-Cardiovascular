import numpy as np

def stepFunctionBinary(x):
    return (1 if x >= 0 else 0)

def stepFunctionBipolar(x):
    return (1 if x >= 0 else -1)