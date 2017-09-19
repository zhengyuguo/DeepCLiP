#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepclip
import sys

train_file = sys.argv[1]
outdir = sys.argv[3]

deepclip.exec_train_weblogo(train_file, deepclip.ideep_model, outdir, 7, 0)
