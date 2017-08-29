#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from deepclip.perf_eval import scores
from deepclip.train_test import train_model, model_predict, data_split, lab1_data
from deepclip.config import *
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_curve
from keras import backend as K

__all__ = ['cross_validation', 'cross_validation_aec']

def cross_validation(func, data_x, data_y, best_model_file = '/tmp/temp.hdf5'):
    skf = StratifiedKFold(n_splits = FOLD, shuffle = True)
    res_scores = list()
    for train_idx, val_idx in skf.split(data_x, data_y):
        model = func(input_shape = data_x.shape[1:])
        train, val = data_split(data_x, data_y)
        train_model(model, train, val, best_model_file = best_model_file, patience = PATIENCE)
        pred_y = model_predict(model, val[0], best_model_file)
        res_scores.append(scores(val[1], pred_y))
    return res_scores

def cross_validation_aec(func, data_x, data_y, best_model_file = '/tmp/temp_aec.hdf5', no_fine = False):
    skf = StratifiedKFold(n_splits = FOLD, shuffle = True)
    res_scores = list()
    for train_idx, val_idx in skf.split(data_x, data_y):
        aec, encoder, pred_model = func(input_shape = data_x.shape[1:])
        train, val = data_split(data_x, data_y)
        train_1 = lab1_data(*train)
        val_1   = lab1_data(*val)
        train_model(aec, (train_1, train_1), (val_1, val_1), patience = AEC_PRE_PATIENCE, best_model_file = best_model_file)
        if no_fine:
            encoder.trainable = False
        train_model(pred_model, train, val, patience = AEC_PRED_PATIENCE, best_model_file = best_model_file)
        pred_y = model_predict(pred_model, val[0], best_model_file)
        res_scores.append(scores(val[1], pred_y))
    return res_scores

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def aec_1d(y_true, y_pred):
    m1 = mean_squared_error(y_true, y_pred * y_true)
    m2 = mean_squared_error(y_true * 0, y_pred * (1-y_true))
    return m1 * 3 + m2
