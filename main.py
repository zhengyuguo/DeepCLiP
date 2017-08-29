#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepclip

train_file = '/Users/zhengyuguo/Downloads/datasets/11_CLIPSEQ_ELAVL1_hg19/training_sample_0/folded.txt.gz'
test_file = '/Users/zhengyuguo/Downloads/datasets/11_CLIPSEQ_ELAVL1_hg19/test_sample_0/folded.txt.gz'
#res = deepclip.exec_train_test(train_file, test_file, deepclip.cnn_auto, AEC = True)
res = deepclip.exec_cv(train_file, deepclip.cnn_auto, AEC = True)
#res = deepclip.exec_cv(train_file, deepclip.cnn_glob)
print(res)
