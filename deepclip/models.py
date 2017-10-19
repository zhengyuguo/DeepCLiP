#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D, GlobalMaxPooling1D, ZeroPadding1D, Cropping1D
from keras.layers import Merge, Lambda
from keras import backend as K
from keras.constraints import unit_norm
from keras import regularizers
from deepclip.noise import UniformNoise

__all__ = ['ideep_model', 'cnn_glob', 'cnn_auto', 'ideep_model_32', 'ideep_model_16']

def conv1d_bn(x, filters, kernel_size, padding = 'same', strides = 1, reg = None):
    if reg == None:
        x = Convolution1D(filters, kernel_size, strides=strides, padding=padding, use_bias = False)(x)
    else:
        x = Convolution1D(filters, kernel_size, strides=strides, padding=padding, use_bias = False, activity_regularizer = regularizers.l1(reg))(x)
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

def ideep_model_32(input_shape = (128, 4)):
    model = Sequential()
    model.add(Convolution1D(input_shape=input_shape, filters = 32, kernel_size = 7, padding = 'valid', activation="relu", strides=1))
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
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def cnn_auto(input_shape = (128, 4)):
    input_seq = Input(shape = input_shape)
    x = UniformNoise(rate = 0.1)(input_seq)
    x = ZeroPadding1D(padding = (0, 108 - input_shape[0]))(x) #(input_seq)
    x = conv1d_bn(x, 32, 10, padding = 'valid')
    x = conv1d_bn(x, 128, 10, padding = 'valid')
    x = MaxPooling1D(pool_size = 2, strides = 2)(x)
    x = conv1d_bn(x, 128, 7, padding = 'valid')
    x = MaxPooling1D(pool_size = 2, strides = 2)(x)
    x = conv1d_bn(x, 128, 7, padding = 'valid')
    x = MaxPooling1D(pool_size = 2, strides = 2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation = 'relu')(x)

    encoder = Model(input_seq, x)
    encoded = x

    x = Dropout(0.4)(x)
    x = Dense(384 * 2, activation = 'relu')(x)
    x = Reshape((24, 16 * 2))(x)

    x = conv1d_bn(x, 32, 7, padding = 'valid')
    x = UpSampling1D(2)(x)
    x = conv1d_bn(x, 32, 7, padding = 'valid')
    x = UpSampling1D(2)(x)
    x = conv1d_bn(x, 32, 7, padding = 'valid')
    x = UpSampling1D(2)(x)
    x = Convolution1D(filters = 4, kernel_size = 7, padding = 'valid', activation = 'softmax')(x)
    x = Cropping1D(cropping=(0, 102 - input_shape[0]))(x)
    model = Model(input_seq, x)


    final = encoded
    final = Dense(1, activation = 'sigmoid')(final)
    final = Model(input_seq, final)
    print(model.summary())
    print(final.summary())
    return model, encoder, final

