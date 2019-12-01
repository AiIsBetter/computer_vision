#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from keras.optimizers import Adam, SGD
from keras import backend as K
from math import ceil
from models.keras_fcos import FCOS
from keras_loss_function.keras_fcos_loss import FCOSLoss
from fcos_encoder_decoder.fcos_input_encoder import FCOSInputEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from misc_utils import cfgs,display_tensorboard
import datetime
# 1: Build the Keras model.
K.clear_session() # Clear previous models from memory.
model = FCOS(image_size=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN, cfgs.CHANNELS),
                n_classes=cfgs.CLASS_NUM,
                mode='training',
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
fcos_loss = FCOSLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=fcos_loss.compute_loss)
model.summary()
if cfgs.CREATE_IMAGE_H5:
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)
    test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
else:
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= cfgs.TRAIN_HDF_DATASET)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=cfgs.VAL_HDF_DATASET)
    test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=cfgs.TEST_HDF_DATASET)

train_dataset.parse_csv(images_dir=cfgs.IMAGE_DIR,
                        labels_filename=cfgs.VAL_LABEL_FILENAME,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=cfgs.IMAGE_DIR,
                      labels_filename=cfgs.VAL_LABEL_FILENAME,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')
test_dataset.parse_csv(images_dir=cfgs.IMAGE_DIR,
                      labels_filename=cfgs.TEST_LABEL_FILENAME,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')
if cfgs.CREATE_IMAGE_H5:
    train_dataset.create_hdf5_dataset(file_path=cfgs.VAL_HDF_DATASET,
                                      resize=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN),
                                      variable_image_size=True,
                                      verbose=True)
    val_dataset.create_hdf5_dataset(file_path=cfgs.VAL_HDF_DATASET,
                                    resize=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN),
                                    variable_image_size=True,
                                    verbose=True)
    test_dataset.create_hdf5_dataset(file_path=cfgs.TEST_HDF_DATASET,
                                      resize=(cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN),
                                      variable_image_size=True,
                                      verbose=True)

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 28, 0.5),
                                                            random_contrast=(0.5, 1.3, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                                                            random_scale=(0.99, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0,0,0))

predictor_sizes = [model.get_layer('P3').output_shape[1:3],
                   model.get_layer('P4').output_shape[1:3],
                   model.get_layer('P5').output_shape[1:3],
                   model.get_layer('P6').output_shape[1:3],
                   model.get_layer('P7').output_shape[1:3]]
# predictor_sizes = [[40,40],[20,20],[10,10],[5,5],[3,3]]


ssd_input_encoder = FCOSInputEncoder(img_height=cfgs.IMG_SHORT_SIDE_LEN,
                                    img_width=cfgs.IMG_SHORT_SIDE_LEN,
                                    n_classes=cfgs.CLASS_NUM,
                                    predictor_sizes=predictor_sizes)


train_generator = train_dataset.generate(batch_size=cfgs.BATCH_SIZE,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=cfgs.BATCH_SIZE,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)
visual_generator = test_dataset.generate(batch_size=cfgs.BATCH_SIZE,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'processed_labels'},
                                     keep_images_without_gt=False)


# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
print( cfgs.SUMMARY_PATH)
# custom callback
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
display_callback = display_tensorboard.Tensorboard_Keras_display(val_data = visual_generator,log_dir = cfgs.SUMMARY_PATH+now)

model_checkpoint = ModelCheckpoint(filepath=cfgs.MODEL_SAVE_PATH,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
csv_logger = CSVLogger(filename='ssd300_training_L_log.csv',
                       separator=',',
                       append=True)
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.000001)
callbacks = [model_checkpoint,csv_logger,early_stopping,reduce_learning_rate,display_callback]
initial_epoch   = 0
final_epoch     = 1000
steps_per_epoch = train_dataset_size/cfgs.BATCH_SIZE
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/cfgs.BATCH_SIZE),
                              initial_epoch=initial_epoch)

# plt.figure(figsize=(20,12))
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend(loc='upper right', prop={'size': 24})
# plt.show()
