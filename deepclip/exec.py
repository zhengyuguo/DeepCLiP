#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepclip.data_helper import dataLoader
from deepclip.models import ideep_model, cnn_glob, cnn_auto
from deepclip.cross_validation import cross_validation, cross_validation_aec
from deepclip.train_test import train_and_test, train_and_test_aec
import os
from sklearn.model_selection import StratifiedKFold
from deepclip.weblogo import logo_kmers_gen

__all__ = ['exec_cv', 'exec_train_test', 'exec_train_test_aec', 'exec_train_weblogo', 'exec_train_weblogo_aec']

def exec_cv(infile, model_func, AEC = False):
    if AEC:
        data = dataLoader(infile)
        res = cross_validation_aec(model_func, data.x, data.y)
    else:
        data = dataLoader(infile)
        res = cross_validation(model_func, data.x, data.y)
    return res

def exec_train_test(train_file, test_file, model_func, AEC = False):
    if AEC:
        train_data = dataLoader(train_file)
        test_data = dataLoader(test_file)
        models = model_func(train_data.x.shape[1:]);
        res = train_and_test_aec(*models, (train_data.x, train_data.y), (test_data.x, test_data.y))
        #res = train_and_test_aec(*models, (train_data.x, train_data.y, train_data.x_n), (test_data.x, test_data.y))
    else:
        train_data = dataLoader(train_file)
        test_data = dataLoader(test_file)
        model = model_func(train_data.x.shape[1:]);
        res = train_and_test(model, (train_data.x, train_data.y), (test_data.x, test_data.y))
    return res

def exec_train_test_aec(train_file_aec, train_file, test_file, model_func, AEC = False):
    train_data_aec = dataLoader(train_file_aec)
    train_data = dataLoader(train_file)
    test_data = dataLoader(test_file)
    models = model_func(train_data.x.shape[1:]);
    res = train_and_test_aec_2(*models, (train_data_aec.x, train_data_aec.y, train_data_aec.x_n),(train_data.x, train_data.y, train_data.x_n), (test_data.x, test_data.y))
    return res

def exec_train_weblogo(infile, model_func, output_dir, filter_size, cnn_layer_idx = 0):
    os.makedirs(output_dir, exist_ok = True)
    data = dataLoader(infile)
    model = model_func(data.x.shape[1:])

    skf = StratifiedKFold(n_splits = 5, shuffle = True)
    train_idx, test_idx = next(skf.split(data.x, data.y))
    train_data = (data.x[train_idx], data.y[train_idx])
    test_data = (data.x[test_idx], data.y[test_idx])
    logo_test_data = (data.x[test_idx], data.seqs[test_idx])

    train_and_test(model, train_data, test_data, "%s/model.hdf5" % output_dir)
    logo_kmers_gen("%s/model.hdf5" % output_dir, logo_test_data, filter_size, output_dir, cnn_layer_idx)

def exec_train_weblogo_aec(infile, model_func, output_dir, filter_size, cnn_layer_idx):
    os.makedirs(output_dir, exist_ok = True)
    data = dataLoader(infile)
    model = model_func(data.x.shape[1:])

    skf = StratifiedKFold(n_splits = 5, shuffle = True)
    train_idx, test_idx = next(skf.split(data.x, data.y))
    train_data = (data.x[train_idx], data.y[train_idx])
    test_data = (data.x[test_idx], data.y[test_idx])
    logo_test_data = (data.x[test_idx], data.seqs[test_idx])

    train_and_test_aec(*model, train_data, test_data, "%s/model.hdf5" % output_dir)
    logo_kmers_gen("%s/model.hdf5" % output_dir, logo_test_data, filter_size, output_dir, cnn_layer_idx)

