#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05

from __future__ import division
import tensorflow as tf
from misc_utils import cfgs
from misc_utils import losses_fcos
class FCOSLoss:
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 inference = False):

        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha
        self.inference = inference
    def compute_loss(self, y_true, y_pred):



        cls_gt = y_true[:, :, 0]
        ctr_gt = y_true[:, :, 1]
        gt_boxes = y_true[:, :, 2:]

        rpn_cls_loss = losses_fcos.focal_loss(y_pred[:,:,cfgs.CLASS_NUM:2*cfgs.CLASS_NUM], cls_gt, alpha=cfgs.ALPHA, gamma=cfgs.GAMMA)
        rpn_bbox_loss = losses_fcos.iou_loss(y_pred[:,:,-4:], gt_boxes[:, :, :4], cls_gt)
        rpn_ctr_loss = losses_fcos.centerness_loss(y_pred[:,:,(-cfgs.CLASS_NUM-5)], ctr_gt, cls_gt)
        #
        #
        # rpn_bbox_loss = tf.Print(rpn_cls_loss, [rpn_bbox_loss],
        #                        message='Debug message rpn_bbox_loss:',
        #                        first_n=100000, summarize=100000000)
        # rpn_ctr_loss = tf.Print(rpn_ctr_loss, [rpn_ctr_loss],
        #                        message='Debug message rpn_ctr_loss:',
        #                        first_n=100000, summarize=100000000)
        total_loss = rpn_cls_loss + rpn_bbox_loss + rpn_ctr_loss




        return total_loss
