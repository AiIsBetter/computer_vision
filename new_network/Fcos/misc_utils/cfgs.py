#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np


# ------------------------------------------------
VERSION = 'FCOS_Res50_20190418'
NET_NAME = 'resnet_v1_50'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')

SUMMARY_PATH = 'log_low/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'


TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
BATCH_SIZE = 8
EPSILON = 1e-5
MOMENTUM = 0.9
# LR = 5e-4 * NUM_GPU * BATCH_SIZE
LR = 1e-5

L2REGULARIZATION = 0.0005
MODEL_SAVE_PATH = 'ssd300_smallcls_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'
BOX_NUM = 2134
# -------------------------------------------- Data_preprocess_config
DATASET_NAME = '76_image_low'
CHANNELS = 3
MEAN_COLOR = [195, 199, 202]# The per-channel mean of the images in the dataset
SWAP_CHANNELS = [2, 1, 0]
IMG_SHORT_SIDE_LEN = 320
IMG_MAX_LENGTH = 320
CLASS_NUM = 7
CREATE_IMAGE_H5 = False
IMAGE_DIR =  'D:/Python/Project/PythonProject/KerasTest/Image/76low/gt320FCOS/'
TRAIN_LABEL_FILENAME = os.path.join(IMAGE_DIR, 'D:/Python/Project/PythonProject/KerasTest/Image/76low/gt320FCOS/labels_train_L_320.csv')
VAL_LABEL_FILENAME   = os.path.join(IMAGE_DIR, 'D:/Python/Project/PythonProject/KerasTest/Image/76low/gt320FCOS/labels_val_L_320.csv')
TEST_LABEL_FILENAME   = os.path.join(IMAGE_DIR, 'D:/Python/Project/PythonProject/KerasTest/Image/76low/gt320FCOS/labels_test_L_320.csv')
TRAIN_HDF_DATASET = 'dataset_low_train_320_L.h5'
VAL_HDF_DATASET = 'dataset_low_val_320_L.h5'
TEST_HDF_DATASET = 'dataset_low_test_320_L.h5'

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
FINAL_CONV_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-np.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = True
LEVLES = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # addjust the base anchor size for voc.
ANCHOR_STRIDE_LIST = [8, 16, 32, 64, 128]
ANCHOR_SCALE_FACTORS = [10., 10., 5.0, 5.0]
SET_WIN = np.asarray([0, 64, 128, 256, 512, 1e5]) * IMG_SHORT_SIDE_LEN / 800
CLIP_BOXES_BOUNDARY = True
# --------------------------------------------FPN config
SHARE_HEADS = True
ALPHA = 0.25
GAMMA = 2
NMS = True
NMS_IOU_THRESHOLD = 0.5
MAXIMUM_DETECTIONS = 400
FILTERED_SCORES = 0.1
SHOW_SCORE_THRSHOLD = 0.2
# --------------------------------------------display config
ONLY_DRAW_BOXES = -1




