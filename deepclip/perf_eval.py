#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve

__all__ = ['scores', 'roc', 'perf_eval']

def scores(y, pred_y):
    auc = roc_auc_score(y, pred_y)
    acc = accuracy_score(y, np.round(pred_y))
    prec, recall, f1, _ = precision_recall_fscore_support(y, np.round(pred_y), average = 'macro')
    return [auc, acc, prec, recall, f1]

def roc(y, pred_y):
    return roc_curve(y, pred_y)

def perf_eval(y, pred_y):
    return scores(y, pred_y), roc(y, pred_y)
