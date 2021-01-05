# -*- coding: utf-8 -*-
"""
Created on 2020/12/31 13:33

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer


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
#
# def logloss(true_y, prediction_y, derive=False, epsilon=1e-9):
#     true_y = np.array(true_y)
#     prediction_y = np.array(prediction_y)
#     lb = LabelBinarizer()
#     lb.fit(true_y)
#     transformed_labels = lb.transform(true_y)
#     if transformed_labels.shape[1] == 1:
#         transformed_labels = np.append(1 - transformed_labels,
#                                        transformed_labels, axis=1)
#     return -np.multiply(transformed_labels, np.log(np.where(prediction_y < epsilon, epsilon, prediction_y))).sum() / prediction_y.shape[0]
