#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepclip.data_helper import dataLoader
from deepclip.models import ideep_model, cnn_glob, cnn_auto
from deepclip.cross_validation import cross_validation, cross_validation_aec
from deepclip.train_test import train_and_test, train_and_test_aec, train_and_test_aec_2

__all__ = ['exec_cv', 'exec_train_test', 'exec_train_test_aec']

def exec_cv(infile, model_func, AEC = False):
    if AEC:
        data = dataLoader(infile, pad = 27 - 27)
        res = cross_validation_aec(model_func, data.x, data.y)
    else:
        data = dataLoader(infile, pad = 0)
        res = cross_validation(model_func, data.x, data.y)
    return res

def exec_train_test(train_file, test_file, model_func, AEC = False):
    if AEC:
        train_data = dataLoader(train_file, pad = 27 - 27)
        test_data = dataLoader(test_file, pad = 27 - 27)
        models = model_func(train_data.x.shape[1:]);
        res = train_and_test_aec(*models, (train_data.x, train_data.y), (test_data.x, test_data.y))
        #res = train_and_test_aec(*models, (train_data.x, train_data.y, train_data.x_n), (test_data.x, test_data.y))
    else:
        train_data = dataLoader(train_file, pad = 0)
        test_data = dataLoader(test_file, pad = 0)
        model = model_func(train_data.x.shape[1:]);
        res = train_and_test(model, (train_data.x, train_data.y), (test_data.x, test_data.y))
    return res

def exec_train_test_aec(train_file_aec, train_file, test_file, model_func, AEC = False):
    train_data_aec = dataLoader(train_file_aec, pad = 27 - 27)
    train_data = dataLoader(train_file, pad = 27 - 27)
    test_data = dataLoader(test_file, pad = 27 - 27)
    models = model_func(train_data.x.shape[1:]);
    res = train_and_test_aec_2(*models, (train_data_aec.x, train_data_aec.y, train_data_aec.x_n),(train_data.x, train_data.y, train_data.x_n), (test_data.x, test_data.y))
    return res
