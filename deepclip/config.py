#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
from keras import optimizers

# Training Hyper-param
EPOCHS = 1#00
BATCH_SIZE = 100
OPTIMIZER = 'rmsprop'
LOSS = 'binary_crossentropy'
PATIENCE = 5

# CV Hyper-param
FOLD = 5

# AEC 
AEC_PRE_PATIENCE = 5
AEC_PRED_PATIENCE = 5

AEC_LOSS = 'mse' #'categorical_crossentropy'
AEC_OPTIMIZER = optimizers.RMSprop(lr = 0.0003)
#AEC_OPTIMIZER = OPTIMIZER
