#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepclip
import sys

train_file = sys.argv[1]
outdir = sys.argv[3]

deepclip.exec_train_weblogo_aec(train_file, deepclip.cnn_auto, outdir, 10, 5)

