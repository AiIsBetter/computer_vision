#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from keras import backend as K
from keras.models import load_model
import numpy as np
from keras_loss_function.keras_fcos_loss import FCOSLoss
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_get_rpn_bbox import get_rpn_bbox
from keras_layers.keras_layer_boardcoast import keras_boardcoast
from keras_layers.keras_layer_exp_stride import keras_exp_stride
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from ssd_encoder_decoder.ssd_output_decoder import decode_detections_fast
from misc_utils import draw_box_in_img,cfgs
import time
import cv2

model_path = 'inference_low.h5'
fcos_loss = FCOSLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
K.clear_session() # Clear previous models from memory.
model = load_model(model_path, custom_objects={
                                               'DecodeDetectionsFast': DecodeDetectionsFast,
                                               'get_rpn_bbox': get_rpn_bbox,
                                                'keras_boardcoast':keras_boardcoast,
                                                'keras_exp_stride': keras_exp_stride,
                                                # 'compute_loss':fcos_loss.compute_loss
                                                })

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=cfgs.IMG_SHORT_SIDE_LEN, width=cfgs.IMG_SHORT_SIDE_LEN)

if cfgs.CREATE_IMAGE_H5:
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
else:
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=cfgs.TEST_HDF_DATASET)
val_dataset.parse_csv(images_dir=cfgs.IMAGE_DIR,
                      labels_filename=cfgs.TEST_LABEL_FILENAME,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')
if cfgs.CREATE_IMAGE_H5:
    val_dataset.create_hdf5_dataset(file_path=cfgs.TEST_HDF_DATASET,
                                    resize=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN),
                                    variable_image_size=True,
                                    verbose=True)

val_dataset_size   = val_dataset.get_dataset_size()
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels, resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)
n_classes = 7
count = 0
for i in range(val_dataset_size):
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
    count +=1
    print(count)
    pos = batch_filenames[0].rfind('/')
    image_name = batch_filenames[0][pos + 1:len(batch_filenames[0])]

    time_start = time.time()
    y_pred = model.predict(batch_images)
    time_end = time.time()
    print('totally cost:', time_end - time_start)
    normalize_coords = True

    # y_pred_decoded = decode_detections_fast(y_pred,
    #                                    confidence_thresh=cfgs.FILTERED_SCORES,
    #                                    iou_threshold=cfgs.NMS_IOU_THRESHOLD,
    #                                    top_k=cfgs.MAXIMUM_DETECTIONS)

    y_pred_decoded_inv = apply_inverse_transforms(y_pred, batch_inverse_transforms)

    scores = np.ones(shape=[len(batch_original_labels[0]), ], dtype=np.float32) * cfgs.ONLY_DRAW_BOXES
    gt_img = draw_box_in_img.draw_boxes_with_label_and_scores(batch_images[0], batch_original_labels[0][:, -4:],
                                                              batch_original_labels[0][:, 0], scores)
    gt_img = cv2.resize(gt_img, dsize=(800, 600))
    cv2.namedWindow("Image")
    cv2.imshow('Image', gt_img)
    # cv2.waitKey()
    result_img = draw_box_in_img.draw_boxes_with_label_and_scores(batch_images[0], y_pred_decoded_inv[0][:, -4:],
                                                                  y_pred_decoded_inv[0][:, 0], y_pred_decoded_inv[0][:, 1])
    result_img = cv2.resize(result_img, dsize=(800, 600))
    cv2.namedWindow("Image1")
    cv2.imshow('Image1', result_img)
    cv2.waitKey()
