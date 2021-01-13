# -*- coding: utf-8 -*-
"""
Created on 2020/12/31 13:33

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import numpy as np


def mse(true_y, prediction_y, derive=False):
    if derive:
        return 2 / true_y.shape[0] * (prediction_y - true_y)
    else:
        return np.mean((true_y - prediction_y) ** 2)


def mae(true_y, prediction_y, derive=False):
    if derive:
        return 1 / true_y.shape[0] * (prediction_y - true_y)
    else:
        return np.mean(np.abs(true_y - prediction_y))


def cross_entropy_loss(true_y, prediction_y, derive=False, epsilon=1e-9):
    if derive:
        return prediction_y - true_y
    return -np.multiply(true_y, np.log(np.where(prediction_y < epsilon, epsilon, prediction_y))).sum() / \
           prediction_y.shape[0]
