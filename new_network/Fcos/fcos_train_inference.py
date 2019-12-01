#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from keras.optimizers import Adam, SGD,Nadam
from keras import backend as K
from models.keras_fcos import FCOS
from keras_loss_function.keras_fcos_loss import FCOSLoss
from misc_utils import  cfgs
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

K.clear_session() # Clear previous models from memory.
model = FCOS(image_size=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN, cfgs.CHANNELS),
                n_classes=cfgs.CLASS_NUM,
                mode='inference_fast',
                l2_regularization=cfgs.L2REGULARIZATION,
                clip_boxes_boundary=cfgs.CLIP_BOXES_BOUNDARY,
                subtract_mean=cfgs.MEAN_COLOR,
                swap_channels=cfgs.SWAP_CHANNELS,
                # batch_size = batch_size,
                anchor_stride_list = cfgs.ANCHOR_STRIDE_LIST,
                confidence_thresh=cfgs.FILTERED_SCORES,
                iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                top_k=cfgs.MAXIMUM_DETECTIONS,
                nms_max_output_size=cfgs.MAXIMUM_DETECTIONS
                )

print("Model built.")

learning_rate = cfgs.LR
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
Nadam = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd = SGD(lr = learning_rate,momentum=0.9,decay = 0.0005)

fcos_loss = FCOSLoss(neg_pos_ratio=3, alpha=1.0,inference=True)
model_path = 'xxxxx.h5'
fine_tune = False
model.load_weights(model_path,by_name=True,skip_mismatch=True)
# model.compile(optimizer=adam,loss='categorical_crossentropy')
model.save('inference_low.h5')


