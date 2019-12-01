from keras.optimizers import Adam, SGD,Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import math

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
import os

n_classes = 11 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
img_height = 320 # Height of the input images
img_width = 320 # Width of the input images
img_channels = 3 # Number of color channels of the input images
mean_color = [165, 166, 161] # The per-channel mean of the images in the dataset
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# TODO: Set the number of classes.
####################################这部分参数按照RefineDet论文里面给出的进行的设置####################################
scales =  [0.1, 0.2, 0.4, 0.8,0.69]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
aspect_ratios = [[0.5, 1, 2,3.0],
                 [0.5, 1, 2,3.0],
                 [0.5, 1, 2,3.0],
                 [0.5, 1, 2,3.0]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = False
steps = [8, 16, 32, 64] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
#######################################################################


clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
normalize_coords = True

# 1: Build the Keras model.
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

print("Model built.")

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model.
learning_rate = 0.001
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

batch_size = 8# Change the batch size if you like, or if you run into GPU memory issues.
sgd = SGD(lr = 0.0001,momentum=0.9,decay = 0.0005)
gt_max = 300
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0,n_class = n_classes+1,gt_max = gt_max)
model_path = 'your model path.h5'
fine_tune = False
epoch_num = 60
if fine_tune:
    model.load_weights(model_path)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
else:
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)
# train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= 'your image date save name.h5')
# val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='your image date save name.h5')
# TODO: Set the paths to your dataset here.
# Images
images_dir = 'your image path/'

# Ground truth
train_labels_filename = os.path.join(images_dir, 'your csv path/labels_train.csv')
val_labels_filename   = os.path.join(images_dir, 'your csv path/labels_val.csv')

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

train_dataset.create_hdf5_dataset(file_path='your image date save name.h5',
                                  resize=(img_height, img_width),
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='your image date save name.h5',
                                resize=(img_height, img_width),
                                variable_image_size=True,
                                verbose=True)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
# Set the image transformations for pre-processing and data augmentation options.
data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                            random_contrast=(0.5, 1.8, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                                                            random_scale=(0.5, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0,0,0))

#  Instantiate an encoder that can encode ground truth labels into the format needed by  the loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [
                   model.get_layer('conv4_3_arm_conf').output_shape[1:3],
                   model.get_layer('conv5_3_arm_conf').output_shape[1:3],
                   model.get_layer('fc7_arm_conf').output_shape[1:3],
                   model.get_layer('conv6_2_arm_conf').output_shape[1:3]]

# predictor_sizes=[(10,10),20,10,10]
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='your model save name-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=5,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
csv_logger = CSVLogger(filename='ssd300_training_log.csv',
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

callbacks = [model_checkpoint,csv_logger,early_stopping,reduce_learning_rate]
if fine_tune :
    initial_epoch = epoch_num
else:
    initial_epoch   = 0
final_epoch     = 1000
steps_per_epoch = train_dataset_size/batch_size
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)

plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})
plt.show()