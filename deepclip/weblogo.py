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

def logo_kmers(filter_outs, filter_size, seqs, filename, maxpct_t = 0.7):
    all_outs = np.ravel(filter_outs)
    all_outs_mean = all_outs.mean()
    all_outs_norm = all_outs - all_outs_mean
    raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    with open(filename, 'w') as f:
        for i in range(filter_outs.shape[0]):
            for j in range(filter_outs.shape[1]):
                if filter_outs[i,j] > raw_t:
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
    
    with open('%s/filter_meme.txt' % output_dir, 'w') as meme:
        meme_header(meme, data[1])
        for i in range(5): #range(cnn_output.shape[2]):
            logo_kmers(cnn_output[:, :, i], filter_size, data[1], '%s/filter_%s.fa' % (output_dir, i))
            filter_pwm, nsites = make_filter_pwm('%s/filter_%s.fa' % (output_dir, i))
            print_pwm(meme, i, filter_pwm, nsites)

def make_filter_pwm(filter_fasta):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(filter_fasta):
        if line[0] == '>':
            continue

        seq = line.rstrip()
        nsites += 1
        if len(pwm_counts) == 0:
            for i in range(len(seq)):
                pwm_counts.append(np.array([1.0]*4))
               
        for i in range(len(seq)):
            try:
                pwm_counts[i][nts[seq[i]]] += 1
            except KeyError:
                pwm_counts[i] += np.array([0.25]*4)

    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])

    return np.array(pwm_freqs), nsites - 4

def print_pwm(f, filter_idx, filter_pwm, nsites):
    if nsites < 10:
        return

    print('MOTIF filter%d' % filter_idx, end = "\n", file = f)
    print('letter-probability matrix: alength= 4 w= %d nsites= %d' % (filter_pwm.shape[0], nsites), end = "\n", file = f) 

    for i in range(0, filter_pwm.shape[0]):
        print('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), end = "\n", file = f)
    print('', end = '\n', file = f)

def meme_header(f, seqs):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}

    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    print('MEME version 4', end = '\n', file = f)
    print('', end = '\n', file = f)
    print('ALPHABET= ACGU', end = '\n', file = f)
    print('', end = '\n', file = f)
    print('Background letter frequencies:', end = '\n', file = f)
    print('A %.4f C %.4f G %.4f U %.4f' % tuple(nt_freqs), end = '\n', file = f)
    print('', end = '\n', file = f)
