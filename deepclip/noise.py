#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras.backend as K
from keras.engine import Layer
class UniformNoise(Layer):
    def __init__(self, rate, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        self.rate = rate
        self.add_noise = True

    def call(self, inputs, training=None):
        if 0 < self.rate < 1 and self.add_noise:
            def noised():
                shape = K.shape(inputs)[0:-1]
                nclass = K.shape(inputs)[-1]
                print(shape)
                print(nclass)
                one_hot = K.one_hot(K.cast(K.round(K.random_uniform(shape = shape, minval=-0.4999999, maxval=3.4999999)),'int32'), num_classes=nclass)
                random = K.random_binomial(shape = K.concatenate([shape, [1]]), p = self.rate)
                noised_inputs = random * one_hot  + (1 - random) * inputs
                return inputs * noised_inputs
            return K.in_train_phase(noised, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
