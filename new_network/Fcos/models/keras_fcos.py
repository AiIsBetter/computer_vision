#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Conv2DTranspose,Lambda,Dense
from keras.regularizers import l2
import keras.backend as K
import keras
from misc_utils import cfgs
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from keras_layers.keras_layer_get_rpn_bbox import get_rpn_bbox
from keras_layers.keras_layer_boardcoast import keras_boardcoast
from keras_layers.keras_layer_exp_stride import keras_exp_stride
from .resnet import ResnetBuilder

import tensorflow as tf

def FCOS(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            clip_boxes_boundary=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            anchor_stride_list = [8, 16, 32, 64, 128],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            # batch_size = 16
            ):
    # FCOS NET


    # n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)
    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    outputs= ResnetBuilder.build_resnet_50(x1,(img_channels, img_height, img_width), n_classes)
    # batch_size = tf.shape(x1)[0]
    # model.compile(loss="categorical_crossentropy", optimizer="sgd")
    C3, C4, C5 = outputs[1:]

    # upsample C5 to get P5 from the FPN paper
    P5 = Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced', kernel_initializer='he_normal')(C5)
    # 从上采样改进为deconv，参考DSSD
    P5_upsampled = Conv2DTranspose(256, (2, 2), strides=(2, 2),padding='same', kernel_initializer='he_normal',name='P5_upsampled')(P5)
    # P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5',kernel_initializer='he_normal')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced', kernel_initializer='he_normal')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    # P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4_upsampled = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                   name='P4_upsampled')(P4)
    P4 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer='he_normal')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced', kernel_initializer='he_normal')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer='he_normal')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6', kernel_initializer='he_normal')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7', kernel_initializer='he_normal')(P7)

    ### Build the subnet class predicts
    # class
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    subnet_class_scores = []
    subnet_class_prob = []
    rpn_cnt_scores= []
    # subnet_loc = []
    rpn_bbox = []
    output = [P3,P4,P5,P6,P7]
    for index,input in enumerate(output):
        subnet_tmp = input
        for i in range(4):
            subnet_tmp = keras.layers.Conv2D(
                filters=256,
                activation='relu',
                name='subnet_P{}_tmp{}'.format(index,i),
                kernel_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                bias_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                **options
            )(subnet_tmp)
        subnet_tmp1 = keras.layers.Conv2D(
            filters=n_classes,
            kernel_initializer=cfgs.FINAL_CONV_WEIGHTS_INITIALIZER,
            bias_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
            name='pyramid_class_P{}'.format(index),
            **options
        )(subnet_tmp)
        # rpn_box_scores = Lambda(lambda inputs: tf.reshape(inputs,
        #                                      [batch_size, -1, n_classes], name='pyramid_class_P{}_reshape'.format(index)))(subnet_tmp1)
        rpn_box_scores = Reshape((-1, n_classes), name='pyramid_class_P{}_reshape'.format(index))(subnet_tmp1)

        # rpn_box_scores = tf.reshape(subnet_tmp1,[batch_size, -1, n_classes], name='pyramid_class_P{}_reshape'.format(index))
        # rpn_box_probs2 = tf.nn.sigmoid(rpn_box_scores, name='pyramid_class_P{}_sigmoid'.format(index))
        # rpn_box_probs = Lambda(lambda inputs:tf.nn.sigmoid(inputs, name='pyramid_class_P{}_sigmoid'.format(index))(rpn_box_scores))
        rpn_box_probs = Activation('sigmoid', name='pyramid_class_P{}_sigmoid'.format(index))(rpn_box_scores)
        subnet_class_scores.append(rpn_box_scores)
        subnet_class_prob.append(rpn_box_probs)
        ##center ness
        subnet_tmp1 = keras.layers.Conv2D(
            filters=1,
            kernel_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
            bias_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
            name='pyramid_center_P{}'.format(index),
            **options
        )(subnet_tmp)
        # subnet_tmp = tf.reshape(subnet_tmp1, [batch_size, -1],
        #                             name='pyramid_center_P{}_reshape'.format(index))
        # subnet_tmp = Lambda(lambda inputs: tf.reshape(inputs,[batch_size, -1], name='pyramid_center_P{}_reshape'.format(index)))(subnet_tmp1)
        subnet_tmp = Reshape((-1, 1),  name='pyramid_center_P{}_reshape'.format(index))(subnet_tmp1)

        rpn_cnt_scores.append(subnet_tmp)
        # loc
        subnet_loc_tmp = input
        for i in range(4):
            subnet_loc_tmp = keras.layers.Conv2D(
                filters=256,
                activation='relu',
                name='subnet_loc_P{}_tmp{}'.format(index,i),
                kernel_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                bias_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                **options
            )(subnet_loc_tmp)
        subnet_loc_tmp = keras.layers.Conv2D(
            filters=4,
            kernel_initializer = cfgs.SUBNETS_WEIGHTS_INITIALIZER,
            bias_initializer = cfgs.SUBNETS_BIAS_INITIALIZER,
            name='pyramid_loc_P{}'.format(index),
            **options
        )(subnet_loc_tmp)

        # rpn_box_offset = Lambda(lambda inputs:  keras.backend.exp(inputs) * anchor_stride_list[index])(subnet_loc_tmp)

        rpn_box_offset = keras_exp_stride(stride = anchor_stride_list[index])(subnet_loc_tmp)
        rpn_box_offset = get_rpn_bbox(stride=anchor_stride_list[index])(rpn_box_offset)
        rpn_box_offset = Reshape((-1, 4),name='rpn_box_P{}_reshape'.format(index))(rpn_box_offset)
        # rpn_box_offset = Lambda(lambda inputs: tf.reshape(inputs,[batch_size, -1,4], name='rpn_box_P{}_reshape'.format(index)))(rpn_box_offset)
        rpn_bbox.append(rpn_box_offset)

    # subnet_class_scores = tf.concat(subnet_class_scores, axis=1)
    # subnet_class_scores = Lambda(lambda inputs:  tf.concat(inputs, axis=1))(subnet_class_scores)
    subnet_class_scores = Concatenate(axis=1)(subnet_class_scores)
    subnet_class_prob = Concatenate(axis=1)(subnet_class_prob)
    # subnet_class_prob = Lambda(lambda inputs: tf.concat(inputs, axis=1))(subnet_class_prob)
    # subenet_center = Concatenate(axis=1)(subenet_center)
    rpn_cnt_scores = Concatenate( axis=1)(rpn_cnt_scores)
    rpn_cnt_prob = Activation('sigmoid', name='rpn_cnt_prob_sigmoid')(rpn_cnt_scores)
    # rpn_cnt_prob = Lambda(lambda inputs:tf.expand_dims(inputs, axis=2))(rpn_cnt_prob)
    # 给每一类输出都分配相应的盒子中心概率值
    rpn_cnt_prob = keras_boardcoast(n_classes = n_classes)(rpn_cnt_prob)
    # rpn_cnt_scores = Lambda(lambda inputs: tf.expand_dims(inputs, axis=2))(rpn_cnt_scores)
    rpn_bbox = Concatenate(axis=1)(rpn_bbox)

    # rpn_bbox = Lambda(lambda inputs: tf.concat(inputs, axis=1))(rpn_bbox)

    # predictions ：batch size,total netural,(nclass_scores,nclass_prob,nclass_cnt_prob,xmin,ymin,xmax,ymax)
    predictions = Concatenate(axis=2, name='predictions')([subnet_class_scores,subnet_class_prob,rpn_cnt_scores,rpn_cnt_prob,rpn_bbox])


    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               clip_boxes_boundary=clip_boxes_boundary,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               clip_boxes_boundary=clip_boxes_boundary,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions_fast')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    return model
