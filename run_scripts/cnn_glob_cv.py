#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepclip
import sys

train_file = sys.argv[1]

outfile = sys.argv[3]

scores = deepclip.exec_cv(train_file, deepclip.cnn_glob)

with open(outfile + '.txt', 'w') as f:
    for i in scores:
        i = [str(j) for j in i]
        print('\t'.join(i) + '\n')
        f.write('\t'.join(i) + '\n')
