#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D, GlobalAveragePooling1D
from keras.layers import Merge, Lambda
from keras import backend as K

__all__ = ['ideep_model', 'cnn_glob', 'cnn_auto']

def conv1d_bn(x, filters, kernel_size, padding = 'same', strides = 1):
    x = Convolution1D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=2, scale=False)(x)
    x = Activation('relu')(x)
    return x

def ideep_model(input_shape = (128, 4)):
    model = Sequential()
    model.add(Convolution1D(input_shape=input_shape, filters = 107, kernel_size = 7, padding = 'valid', activation="relu", strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(107, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def cnn_glob(input_shape = (128, 4)):
    model = Sequential()
    model.add(Convolution1D(input_shape=input_shape, filters = 128, kernel_size = 24, padding = 'valid', activation="relu", strides=1))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def cnn_auto(input_shape = (128, 4)):
    input_seq = Input(shape = input_shape)
    x = conv1d_bn(input_seq, 16, 10)
    x = MaxPooling1D(pool_size = 2, strides = 2)(x)
    encoder = Model(input_seq, x)
    encoded = x
    x = conv1d_bn(x, 32, 3)
    x = MaxPooling1D(pool_size = 2, strides = 2)(x)
    x = conv1d_bn(x, 4, 1) 

    x = conv1d_bn(x, 4, 1)
    x = UpSampling1D(2)(x)
    x = conv1d_bn(x, 32, 3)
    x = UpSampling1D(2)(x)
    x = conv1d_bn(x, 16, 10)
    x = Convolution1D(filters = 4, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
    x = Lambda(lambda x : x / K.expand_dims(K.sum(x,2), 2))(x)
    model = Model(input_seq, x)

    final = conv1d_bn(encoded, 4, 1)
    final = Flatten()(final)
    final = Dense(1, activation = 'sigmoid')(final)
    final = Model(input_seq, final)
    return model, encoder, final
