#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from keras.callbacks import Callback
import numpy as np
from keras import backend as K
import tensorflow as tf
import cv2

from misc_utils import cfgs,draw_box_in_img
from fcos_encoder_decoder.fcos_output_decoder import  decode_detections
from fcos_encoder_decoder.fcos_output_decoder import  decode_detections_fast

def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)

class Tensorboard_Keras_display(Callback):
    def __init__(self, log_dir='logs/', val_data=None):
          super(Tensorboard_Keras_display, self).__init__()
          self.seen = 0
          # self.feed_inputs_display = feed_inputs_display
          self.writer = tf.summary.FileWriter(log_dir)
          self.validation_data = val_data
    # A callback has access to its associated model through the class property self.model.
    def on_epoch_end(self,epoch,logs = None):
          self.seen += 1
          val = next(self.validation_data)
          gt_img = val[0]
          gt_label= val[1]
          disp_pred = K.get_session().run(self.model.output, feed_dict = {self.model.input : gt_img})
          y_pred_decoded = decode_detections_fast(disp_pred,
                                             confidence_thresh=cfgs.FILTERED_SCORES,
                                             iou_threshold=0.5,
                                             top_k=cfgs.MAXIMUM_DETECTIONS)

          scores = np.ones(shape=[len(gt_label[0]), ], dtype=np.float32) * cfgs.ONLY_DRAW_BOXES
          gt_img_result = draw_box_in_img.draw_boxes_with_label_and_scores(gt_img[0],
                                                                    gt_label[0][:, -4:],
                                                                    gt_label[0][:, 0], scores)
          gt_img_result = cv2.resize(gt_img_result, dsize=(800, 600))
          # cv2.namedWindow("Image")
          # cv2.imshow('Image', gt_img_result)
          # cv2.waitKey()
          if len(y_pred_decoded[0])<1:
              predict_img_result = gt_img[0]
          else:
              predict_img_result = draw_box_in_img.draw_boxes_with_label_and_scores(gt_img[0],
                                                                                    y_pred_decoded[0][:, -4:],
                                                                                    y_pred_decoded[0][:, 0],
                                                                                    y_pred_decoded[0][:, 1])
          predict_img_result = cv2.resize(predict_img_result, dsize=(800, 600))
          # cv2.namedWindow("predict_img_result")
          # cv2.imshow('predict_img_result', predict_img_result)
          # cv2.waitKey()

          # print(disp_pred)

          def make_image(tensor):
              """
              Convert an numpy representation image to Image protobuf.
              Copied from https://github.com/lanpa/tensorboard-pytorch/
              """
              from PIL import Image
              height, width, channel = tensor.shape
              image = Image.fromarray(tensor)
              import io
              output = io.BytesIO()
              image.save(output, format='PNG')
              image_string = output.getvalue()
              output.close()
              return tf.Summary.Image(height=height,
                                      width=width,
                                      colorspace=channel,
                                      encoded_image_string=image_string)

          gt_img_result = make_image(gt_img_result)
          predict_img_result = make_image(predict_img_result)
          summary = tf.Summary(value=[tf.Summary.Value(tag ='plot/gt_img', image=gt_img_result),
                                      tf.Summary.Value(tag='plot/result_img', image=predict_img_result)])
          # writer = tf.summary.FileWriter('./logs')
          self.writer.add_summary(summary, self.seen)
          self.writer.flush()

          return
