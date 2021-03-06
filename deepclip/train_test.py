#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepclip.perf_eval import perf_eval
from deepclip.config import *
from sklearn.model_selection import StratifiedKFold

__all__ = ['train_model', 'model_predict', 'data_split', 'lab1_data', 'train_and_test', 'train_and_test_aec']

def train_model(model, training_data, validation_data, patience, best_model_file = '/tmp/best_model.hdf5', loss = LOSS, optimizer = OPTIMIZER):

    model.compile(loss = loss, optimizer = optimizer, metrics=["accuracy"]) 
    earlystopper = EarlyStopping(monitor='val_loss', patience = patience, verbose=1)
    checkpoint = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(
            x               = training_data[0], 
            y               = training_data[1], 
            batch_size      = BATCH_SIZE,
            epochs          = EPOCHS, 
            verbose         = 1, 
            validation_data = validation_data, 
            callbacks       = [earlystopper, checkpoint])
    return

def model_predict(model, test_data, model_weight = '/tmp/best_model.hdf5'): 
    if model_weight:
        model.load_weights(model_weight)
    pred_y = model.predict(test_data)
    return pred_y

def data_split(x, y, splits = 5):
    skf = StratifiedKFold(n_splits = splits, shuffle = True)
    train_idx, val_idx = next(skf.split(x, y))
    train = x[train_idx], y[train_idx]
    val   = x[val_idx],   y[val_idx]
    return train, val

def lab1_data(x, y):
    idx = [i for i in range(len(y)) if y[i] == 1]
    return x[idx]

def train_and_test(model, training_data, test_data, best_model_file = '/tmp/temp_train_test.hdf5'):
    train, val = data_split(training_data[0], training_data[1])
    train_model(model, train, val, patience = PATIENCE, best_model_file = best_model_file)
    model.load_weights(best_model_file)
    pred_y = model_predict(model, test_data[0], best_model_file)
    return perf_eval(test_data[1], pred_y)

def train_and_test_aec(aec, encoder, pred_model, training_data, test_data, best_model_file = '/tmp/temp_train_test_aec.hdf5', no_fine = False):
    train, val = data_split(training_data[0], training_data[1])
    train_1 = lab1_data(*train)
    val_1   = lab1_data(*val)

    train_model(aec, (train_1, train_1), (val_1, val_1), patience = AEC_PRE_PATIENCE, best_model_file = best_model_file, loss = AEC_LOSS)

    aec.load_weights(best_model_file)

    pred_y = model_predict(aec, test_data[0], best_model_file)
    print(pred_y[0])
    print(test_data[0][0])

    #print(aec.layers[1])
    aec.layers[1].add_noise = False
    encoder.trainable = False
    train_model(pred_model, train, val, patience = AEC_PRED_PATIENCE, best_model_file = best_model_file) #, optimizer = AEC_OPTIMIZER)
    pred_model.load_weights(best_model_file)

    if not no_fine:
        encoder.trainable = True 
        train_model(pred_model, train, val, patience = AEC_PRED_PATIENCE, best_model_file = best_model_file, optimizer = AEC_OPTIMIZER)
        pred_model.load_weights(best_model_file)

    pred_y = model_predict(pred_model, test_data[0], best_model_file)
    return perf_eval(test_data[1], pred_y)

def train_aec(aec, training_data, best_model_file = '/tmp/temp_train_aec.hdf5'):
    train, val, n_train, n_val = data_split2(training_data[0], training_data[1], training_data[2])
    train_1 = lab1_data(*train)
    val_1   = lab1_data(*val)

    n_train_1 = lab1_data(*n_train)
    n_val_1   = lab1_data(*n_val)

    train_model(aec, (n_train_1, train_1), (n_val_1, val_1), patience = AEC_PRE_PATIENCE, best_model_file = best_model_file, loss = AEC_LOSS)
