#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import models
import keras.backend as K
import numpy as np
from deepclip.noise import UniformNoise
from deepclip.data_helper import dataLoader
import os

def get_feature(model, X_batch, index):
    inputs = [K.learning_phase()] + [model.input]
    _convout1_f = K.function(inputs, [model.layers[index].output])
    activations =  _convout1_f([0] + [X_batch])
    return activations[0]

def logo_kmers(filter_outs, filter_size, seqs, filename):
    with open(filename, 'w') as f:
        for i in range(filter_outs.shape[0]):
            for j in range(filter_outs.shape[1]):
                if filter_outs[i,j] > 0.0:
                    kmer = seqs[i][j:j+filter_size]
                    if len(kmer) <filter_size:
                        continue
                    print('>%d_%d' % (i,j), end = '\n', file = f)
                    print(kmer, end = '\n', file = f)

def logo_kmers_gen(model_file, data, filter_size, output_dir, cnn_layer_idx = 0):
    model = models.load_model(model_file, custom_objects = {'UniformNoise': UniformNoise})
    print(model.summary())
    print(model.layers[cnn_layer_idx])
    cnn_output = get_feature(model, data[0], cnn_layer_idx)
    os.makedirs(output_dir, exist_ok = True)
    for i in range(cnn_output.shape[2]):
        logo_kmers(cnn_output[:, :, i], filter_size, data[1], '%s/filter_%s.fa' % (output_dir, i))
