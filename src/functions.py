import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def binary_cross_entropy(y, s):
    """
    binary cross entropyを計算する.
    x > 0ではlog(1+e^x) = x + log(1+e^(-x)) で計算
    """
    if s > 0:
        L = y*np.log(1+np.exp(-s)) + (1-y)*(s+np.log(1+np.exp(-s)))
    else:
        L = y*np.log(1+np.exp(-s)) + (1-y)*np.log(1+np.exp(s))
    return L