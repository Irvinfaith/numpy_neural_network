# -*- coding: utf-8 -*-
"""
Created on 2020/12/22 11:37

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import numpy as np


def sigmoid(x, derive=False):
    if derive:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def relu(x, derive=False):
    if derive:
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)


def tanh(x, derive=False):
    if derive:
        return 1 - np.power(tanh(x), 2)
    return 2 * sigmoid(np.multiply(2, x)) - 1


def softmax(x, derive=False):
    if derive:
        pass
    return np.exp(x) * (1 / np.sum(np.exp(x), axis=1)).reshape(-1, 1)
