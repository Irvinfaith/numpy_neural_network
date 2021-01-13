# -*- coding: utf-8 -*-
"""
Created on 2021/1/12 18:22

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import pandas as pd
import numpy as np


def series_to_array(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Input type has to be `pandas.dataframe` or `numpy.ndarray`, your type is `{type(x)}`")


def array1d_to_onehot(y, num_classes):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    trans_y = np.zeros((y.shape[0], num_classes))
    for index, _ in enumerate(y):
        trans_y[index][_] += 1
    return trans_y
