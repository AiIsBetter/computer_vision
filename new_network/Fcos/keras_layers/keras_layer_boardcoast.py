#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division
import numpy as np
import tensorflow as tf
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

class keras_boardcoast(Layer):
    def __init__(self,n_classes=None,
                 **kwargs):

        # We need these members for the config.
        self.n_classes = n_classes
        super(keras_boardcoast, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(keras_boardcoast, self).build(input_shape)
    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)
    def call(self, inputs, mask=None):
        batch, n_boxes = tf.shape(inputs)[0], tf.shape(inputs)[1]
        inputs_boardcast = tf.broadcast_to(inputs, [batch, n_boxes, self.n_classes])
        return inputs_boardcast

    def get_config(self):
        config = {
            'n_classes': self.n_classes,

        }
        base_config = super(keras_boardcoast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
