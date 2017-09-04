#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepclip
import sys
import pickle

train_file = sys.argv[1]
test_file = sys.argv[2]
outfile = sys.argv[3]

scores, roc = deepclip.exec_train_test(train_file, test_file, deepclip.ideep_model_16)

with open(outfile + '.txt', 'w') as f:
    scores = [str(i) for i in scores]
    f.write('\t'.join(scores))

with open(outfile + '.pkl', 'wb') as f:
    pickle.dump(roc, f)
