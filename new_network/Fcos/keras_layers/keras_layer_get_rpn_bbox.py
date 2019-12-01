#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

class get_rpn_bbox(Layer):
    '''
    A Keras layer to decode the raw SSD prediction output.

    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.

    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 stride=None,
                 **kwargs):

        # We need these members for the config.
        self.stride = stride

        super(get_rpn_bbox, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(get_rpn_bbox, self).build(input_shape)
    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)
    def call(self, rpn_box_offset, mask=None):
        # rpn_box_offset = tf.Print(rpn_box_offset, [rpn_box_offset],
        #                     message='Debug message rpn_box_offset:',
        #                     first_n=10000, summarize=100000)
        batch, fm_height, fm_width = tf.shape(rpn_box_offset)[0], tf.shape(rpn_box_offset)[1], tf.shape(rpn_box_offset)[2]
        rpn_box_offset = tf.reshape(rpn_box_offset, [batch, -1, 4])

        y_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_height, tf.float32) - tf.constant(0.5),
                                                tf.cast(fm_height, tf.float32)],
                            Tout=[tf.float32])
        y_list = tf.broadcast_to(tf.reshape(y_list, [1, fm_height, 1, 1]), [1, fm_height, fm_width, 1])

        x_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_width, tf.float32) - tf.constant(0.5),
                                                tf.cast(fm_width, tf.float32)],
                            Tout=[tf.float32])
        x_list = tf.broadcast_to(tf.reshape(x_list, [1, 1, fm_width, 1]), [1, fm_height, fm_width, 1])

        xy_list = tf.concat([x_list, y_list], axis=3) * self.stride

        center = tf.reshape(tf.broadcast_to(xy_list, [batch, fm_height, fm_width, 2]),
                            [batch, -1, 2])

        xmin = tf.expand_dims(center[:, :, 0] - rpn_box_offset[:, :, 0], axis=2)
        ymin = tf.expand_dims(center[:, :, 1] - rpn_box_offset[:, :, 1], axis=2)
        xmax = tf.expand_dims(center[:, :, 0] + rpn_box_offset[:, :, 2], axis=2)
        ymax = tf.expand_dims(center[:, :, 1] + rpn_box_offset[:, :, 3], axis=2)
        all_boxes = tf.concat([xmin, ymin, xmax, ymax], axis=2)

        return all_boxes

    def get_config(self):
        config = {
            'stride': self.stride,
            # 'batch_size,': self.batch_size,
        }
        base_config = super(get_rpn_bbox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
