from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
#from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast,decode_detections_fast_high

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import time
# Set the image size.
img_height = 320
img_width = 320
n_classes = 11
model_path = 'your model path.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0,n_class = n_classes+1)
K.clear_session() # Clear previous models from memory.
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

# 1: Set the generator for the predictions.
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# Images
images_dir = 'your image path/'
# Ground truth
val_labels_filename   = 'your csv path/labels_val.csv'

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
# val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='your image date save name.h5')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')
val_dataset.create_hdf5_dataset(file_path='dataset_high_trainval_320_11cls.h5',
                                resize=(img_height, img_width),
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


colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# classes = ['na',
#            'h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11',
#            'b1','b2','b3',
#            'j1','j2','j3','j4','j5','j6','j7','j8','j9','j10','j11',
#            'z',
#            'x']
plt.figure(figsize=(20,12))
current_axis = plt.gca()

def getBigCls(small_cls):
    class_dictionary = {0:0,
                        1: 1, 2: 1,
                        3: 2, 4: 2,
                        5: 3, 6: 3, 7: 3, 8: 3, 9: 3,
                        10: 4,
                        11: 5
                        }
    return class_dictionary[small_cls]
count = 0
# 2: Generate samples.
for i in range(val_dataset_size):
#for i in range(5):
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
    pos = batch_filenames[0].rfind('/')
    image_name = batch_filenames[0][pos + 1:len(batch_filenames[0])]

    count +=1
    print(count)
    # 3: Make predictions.
    time_start = time.time()
    y_pred = model.predict(batch_images)

    time_end = time.time()
    print('totally cost:', time_end - time_start)
    # 4: Decode the raw predictions in `y_pred`.
    normalize_coords = True
    y_pred_decoded = decode_detections_fast(y_pred,
                                       confidence_thresh=0.2,
                                       iou_threshold=0.2,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    # 5: Draw the predicted boxes onto the image

    box_pred_matched_array = []

    colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
    classes = ['na','h','yh','b', 'nong','csgj', 'lisj', 'lsj', 'qtj', 'fjxy','z','x']

    plt.figure(figsize=(20, 12))
    current_axis = plt.gca()

    #保存结果图片
    for box in batch_original_labels[0]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        current_axis.text(xmax, ymax, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

    for box in y_pred_decoded_inv[0]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    plt.imshow(batch_original_images[0])
    result_img_path_name = 'result_img/{}'.format(image_name)
    # plt.savefig(result_img_path_name)
    plt.show()
    plt.close()



