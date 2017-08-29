#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Training Hyper-param
EPOCHS = 3 #100
BATCH_SIZE = 250 
OPTIMIZER = 'adadelta'
LOSS = 'binary_crossentropy'
PATIENCE = 5

# CV Hyper-param
FOLD = 5

# AEC 
AEC_PRE_PATIENCE = 1 #5
AEC_PRED_PATIENCE = 1 #20
