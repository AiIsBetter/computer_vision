#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer

class keras_exp_stride(Layer):

    def __init__(self,stride=None,
                 **kwargs):

        # We need these members for the config.
        self.stride = stride
        super(keras_exp_stride, self).__init__(**kwargs)

    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)
    def call(self, inputs, mask=None):
        batch, n_boxes = tf.shape(inputs)[0], tf.shape(inputs)[1]
        # inputs_exp = tf.exp(inputs*self._x)
        inputs_exp = tf.exp(inputs)*self.stride
        return inputs_exp

    def get_config(self):
        config = {
            'stride': self.stride,

        }
        base_config = super(keras_exp_stride, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
