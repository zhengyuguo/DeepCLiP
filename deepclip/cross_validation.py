#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from deepclip.train_test import train_and_test, train_and_test_aec
from deepclip.config import *
from sklearn.model_selection import StratifiedKFold

__all__ = ['cross_validation', 'cross_validation_aec']

def cross_validation(func, data_x, data_y, best_model_file = '/tmp/temp.hdf5'):
    skf = StratifiedKFold(n_splits = FOLD, shuffle = True)
    res_scores = list()
    for train_idx, val_idx in skf.split(data_x, data_y):
        model = func(input_shape = data_x.shape[1:])
        perf = train_and_test(model, (data_x[train_idx], data_y[train_idx]), (data_x[val_idx], data_y[val_idx]), best_model_file = best_model_file)
        res_scores.append(perf[0])
    return res_scores

def cross_validation_aec(func, data_x, data_y, best_model_file = '/tmp/temp_aec.hdf5', nofine = False):
    skf = StratifiedKFold(n_splits = FOLD, shuffle = True)
    res_scores = list()
    for train_idx, val_idx in skf.split(data_x, data_y):
        aec, encoder, pred_model = func(input_shape = data_x.shape[1:])
        perf = train_and_test_aec(aec, encoder, pred_model, (data_x[train_idx], data_y[train_idx]), (data_x[val_idx], data_y[val_idx]), best_model_file = best_model_file, nofine = nofine)
        res_scores.append(perf[0])
    return res_scores

