#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D, GlobalAveragePooling1D, ZeroPadding1D, Cropping1D
from keras.layers import Merge, Lambda
from keras import backend as K
from keras.constraints import unit_norm
from keras import regularizers
from deepclip.noise import UniformNoise

__all__ = ['ideep_model', 'cnn_glob', 'cnn_auto', 'ideep_model_32', 'ideep_model_16']

def conv1d_bn(x, filters, kernel_size, padding = 'same', strides = 1, reg = None):
    #if reg == None:
    #    x = Convolution1D(filters, kernel_size, strides=strides, padding=padding, use_bias = False, kernel_constraint = unit_norm(axis = [0, 1]))(x)
    #else:
    #    x = Convolution1D(filters, kernel_size, strides=strides, padding=padding, use_bias = False, kernel_constraint = unit_norm(axis = [0, 1]), activity_regularizer = regularizers.l1(reg))(x)
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

def ideep_model_16(input_shape = (128, 4)):
    model = Sequential()
    model.add(Convolution1D(input_shape=input_shape, filters = 16, kernel_size = 7, padding = 'valid', activation="relu", strides=1))
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

def cnn_auto_good(input_shape = (128, 4)):
    input_seq = Input(shape = input_shape)
    x = ZeroPadding1D(padding = (0, 104 - input_shape[0]))(input_seq)
    x = conv1d_bn(x, 16, 7)
    x = MaxPooling1D(pool_size = 3, strides = 2, padding = 'same')(x)
    #x = conv1d_bn(x, 16, 4)
    #x = MaxPooling1D(pool_size = 3, strides = 2, padding = 'same')(x)
    #x = conv1d_bn(x, 8, 4)
    #x = MaxPooling1D(pool_size = 3, strides = 2, padding = 'same')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(48, activation = 'relu')(x)

    encoder = Model(input_seq, x)
    encoded = x

    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(832, activation = 'relu')(x)
    x = Reshape((52, 16))(x)

    #x = conv1d_bn(x, 8, 4)
    #x = UpSampling1D(2)(x)
    #x = conv1d_bn(x, 16, 4)
    #x = UpSampling1D(2)(x)
    x = conv1d_bn(x, 16, 4)
    x = UpSampling1D(2)(x)
    x = Convolution1D(filters = 4, kernel_size = 7, padding = 'same', activation = 'softmax')(x)
    x = Cropping1D(cropping=(0, 104 - input_shape[0]))(x)
    model = Model(input_seq, x)


    final = Dropout(0.6)(encoded)
    #final = Flatten()(final)
    #final = Dense(48, activation = 'relu')(final)
    #final = Dropout(0.5)(final)
    final = Dense(1, activation = 'sigmoid')(final)
    final = Model(input_seq, final)
    print(model.summary())
    print(final.summary())
    return model, encoder, final

def cnn_auto(input_shape = (128, 4)):
    input_seq = Input(shape = input_shape)
    x = UniformNoise(rate = 0.2)(input_seq)
    x = ZeroPadding1D(padding = (0, 104 - input_shape[0]))(x) #(input_seq)
    x = conv1d_bn(x, 16, 7)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(48, activation = 'relu')(x)

    encoder = Model(input_seq, x)
    encoded = x

    x = Dropout(0.5)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(52 * 16, activation = 'relu')(x)
    x = Reshape((52, 16))(x)

    x = conv1d_bn(x, 16, 4)
    x = UpSampling1D(2)(x)
    x = Convolution1D(filters = 4, kernel_size = 7, padding = 'same', activation = 'softmax')(x)
    x = Cropping1D(cropping=(0, 104 - input_shape[0]))(x)
    model = Model(input_seq, x)


    final = Dropout(0.6)(encoded)
    final = Dense(1, activation = 'sigmoid')(final)
    final = Model(input_seq, final)
    print(model.summary())
    print(final.summary())
    return model, encoder, final

