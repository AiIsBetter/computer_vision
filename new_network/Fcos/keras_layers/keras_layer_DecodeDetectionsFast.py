#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from misc_utils import cfgs
class DecodeDetectionsFast(Layer):
    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 clip_boxes_boundary=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''
        All default argument values follow the Caffe implementation.

        Arguments:
            confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
                positive class in order to be considered for the non-maximum suppression stage for the respective class.
                A lower value will result in a larger part of the selection process being done by the non-maximum suppression
                stage, while a larger value will result in a larger part of the selection process happening in the confidence
                thresholding stage.
            iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
                with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
                to the box score.
            top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
                non-maximum suppression stage.
            nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
                suppression.
            coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
                i.e. the format `(cx, cy, w, h)` (box center coordinates, width, and height). Other coordinate formats are
                currently not supported.
            normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
                and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
                relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
                Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
                coordinates. Requires `img_height` and `img_width` if set to `True`.
            img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
            img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        if coords != 'centroids':
            raise ValueError("The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.clip_boxes_boundary = clip_boxes_boundary
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_clip_boxes_boundary = tf.constant(self.clip_boxes_boundary, name='tf_clip_boxes_boundary')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetectionsFast, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetectionsFast, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        '''

        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################

        def clip_boxes_to_img_boundarie():
            xmin = y_pred[:, :, -4]
            ymin = y_pred[:, :, -3]
            xmax = y_pred[:, :, -2]
            ymax = y_pred[:, :, -1]
            img_h, img_w = self.img_width, self.img_width

            img_h, img_w = tf.cast(img_h, tf.float32), tf.cast(img_w, tf.float32)

            xmin = tf.maximum(tf.minimum(xmin, img_w - 1.), 0.)
            ymin = tf.maximum(tf.minimum(ymin, img_h - 1.), 0.)

            xmax = tf.maximum(tf.minimum(xmax, img_w - 1.), 0.)
            ymax = tf.maximum(tf.minimum(ymax, img_h - 1.), 0.)
            tf.expand_dims(xmin * self.tf_img_width, axis=-1)

            xmin = tf.expand_dims(xmin, axis=-1)
            ymin = tf.expand_dims(ymin, axis=-1)
            xmax = tf.expand_dims(xmax, axis=-1)
            ymax = tf.expand_dims(ymax, axis=-1)
            return xmin, ymin, xmax, ymax
        def ori_output_boxes_to_img():
            xmin = y_pred[:, :, -4]
            ymin = y_pred[:, :, -3]
            xmax = y_pred[:, :, -2]
            ymax = y_pred[:, :, -1]
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(self.tf_clip_boxes_boundary, clip_boxes_to_img_boundarie, ori_output_boxes_to_img)


        # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
        final_box_cls_prob = y_pred[:,:,cfgs.CLASS_NUM:2*cfgs.CLASS_NUM]*y_pred[:,:,2*cfgs.CLASS_NUM+1:-4]
        # y_pred = tf.concat(values=[final_box_cls_prob, xmin,ymin,xmax,ymax], axis=-1)
        class_ids = tf.expand_dims(tf.to_float(tf.argmax(final_box_cls_prob, axis=-1)+1), axis=-1)
        # Extract the confidences of the maximal classes.
        confidences = tf.reduce_max(final_box_cls_prob, axis=-1, keep_dims=True)
        y_pred = tf.concat(values=[class_ids, confidences, xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. Perform confidence thresholding, non-maximum suppression, and top-k filtering.
        #####################################################################################

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):

            # Keep only the non-background boxes.
            positive_boxes = tf.not_equal(batch_item[...,0], -1)
            predictions = tf.boolean_mask(tensor=batch_item,
                                          mask=positive_boxes)

            def perform_confidence_thresholding():
                # Apply confidence thresholding.
                threshold_met = predictions[:,1] > self.tf_confidence_thresh
                return tf.boolean_mask(tensor=predictions,
                                       mask=threshold_met)
            def no_positive_boxes():
                return tf.constant(value=0.0, shape=(1,6))

            # If there are any positive predictions, perform confidence thresholding.
            predictions_conf_thresh = tf.cond(tf.equal(tf.size(predictions), 0), no_positive_boxes, perform_confidence_thresholding)

            def perform_nms():
                scores = predictions_conf_thresh[...,1]

                # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(predictions_conf_thresh[...,-4], axis=-1)
                ymin = tf.expand_dims(predictions_conf_thresh[...,-3], axis=-1)
                xmax = tf.expand_dims(predictions_conf_thresh[...,-2], axis=-1)
                ymax = tf.expand_dims(predictions_conf_thresh[...,-1], axis=-1)
                boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=self.tf_nms_max_output_size,
                                                              iou_threshold=self.iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=predictions_conf_thresh,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima
            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1,6))

            # If any boxes made the threshold, perform NMS.
            predictions_nms = tf.cond(tf.equal(tf.size(predictions_conf_thresh), 0), no_confident_predictions, perform_nms)

            # Perform top-k filtering for this batch item or pad it in case there are
            # fewer than `self.top_k` boxes left at this point. Either way, produce a
            # tensor of length `self.top_k`. By the time we return the final results tensor
            # for the whole batch, all batch items must have the same number of predicted
            # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
            # predictions are left after the filtering process above, we pad the missing
            # predictions with zeros as dummy entries.
            def top_k():
                return tf.gather(params=predictions_nms,
                                 indices=tf.nn.top_k(predictions_nms[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)
            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=predictions_nms,
                                            paddings=[[0, self.tf_top_k - tf.shape(predictions_nms)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(predictions_nms)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return (batch_size, self.tf_top_k, 6) # Last axis: (class_ID, confidence, 4 box coordinates)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetectionsFast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
