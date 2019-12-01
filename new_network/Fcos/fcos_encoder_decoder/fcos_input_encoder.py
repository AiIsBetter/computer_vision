#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.05
from __future__ import division
import numpy as np
from misc_utils import cfgs

class FCOSInputEncoder:
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,

                ):
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes # + 1 for the background class
        self.predictor_sizes = predictor_sizes
    def __call__(self, ground_truth_labels, diagnostics=False):

        batch_size = len(ground_truth_labels)

        def fcos_target(gt_boxes1, image_batch, fm_size_list,iii):

            gt_boxes = gt_boxes1[...,[1,2,3,4,0]]

            # 每个batch里面一张图单独处理
            gt_boxes = np.array(gt_boxes, np.int32)
            raw_height, raw_width = image_batch

            gt_boxes = np.concatenate([np.zeros((1, 5)), gt_boxes])
            b = np.nonzero(gt_boxes1)
            if b[0].shape[0] == 0:
                # gt_target = [0, 0, 0, 0, 0, 0] * 2134
                print(gt_boxes1)
                print(gt_boxes)
                return np.zeros(shape = [cfgs.BOX_NUM,6])
            gt_boxes_area = (np.abs(
                (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
            # gt_boxes按面积从小到大排序,
            gt_boxes = gt_boxes[np.argsort(gt_boxes_area)]
            boxes_cnt = len(gt_boxes)

            shift_x = np.arange(0, raw_width).reshape(-1, 1)
            shift_y = np.arange(0, raw_height).reshape(-1, 1)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            # 原图上的所有点和所有gtbox一次性生成所有lrtb
            off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
                     gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
            off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
                     gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
            off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
                      gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
            off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
                      gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

            center = ((np.minimum(off_l, off_r) * np.minimum(off_t, off_b)) / (
                np.maximum(off_l, off_r) * np.maximum(off_t, off_b) + cfgs.EPSILON))
            center = np.squeeze(np.sqrt(np.abs(center)))

            center[:, :, 0] = 0

            offset = np.concatenate([off_l, off_t, off_r, off_b], axis=3)
            cls = gt_boxes[:, 4]

            cls_res_list = []
            ctr_res_list = []
            gt_boxes_res_list = []

            for fm_i, stride in enumerate(cfgs.ANCHOR_STRIDE_LIST):
                fm_height = fm_size_list[fm_i][0]
                fm_width = fm_size_list[fm_i][1]

                shift_x = np.arange(0, fm_width)
                shift_y = np.arange(0, fm_height)
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                # 生成所有f map上面点坐标
                xy = np.vstack((shift_y.ravel(), shift_x.ravel())).transpose()
                # 所有坐标，从fmap上面转换到原图x*s 取出原图对应的lrtb值，shape(x,y,z)中y代表gtbox数量+1(background)
                off_xy = offset[xy[:, 0] * stride, xy[:, 1] * stride]
                # 找到对应的fmap中，每个点的lrtb中的最大值
                off_max_xy = off_xy.max(axis=2)
                # 生成所有featuremap上面对应点对应的gtbox，每个点对应所有gtbox
                off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

                is_in_boxes = (off_xy > 0).all(axis=2)
                is_in_layer = (off_max_xy <= cfgs.SET_WIN[fm_i+1]) & \
                              (off_max_xy >= cfgs.SET_WIN[fm_i])
                off_valid[xy[:, 0], xy[:, 1], :] = is_in_boxes & is_in_layer
                # 第一列为补充的背景列，置0弃用
                off_valid[:, :, 0] = 0
                # 每个点对应的最大gt,gt的顺序已经按面积从小到大排列，所以取出的gt已经相当于论文中按gt面积小的优先作为fmap的cls
                hit_gt_ind = np.argmax(off_valid, axis=2)

                # gt_boxes
                gt_boxes_res = np.zeros((fm_height, fm_width, 4))
                # 取出匹配上的gtbox的坐标
                gt_boxes_res[xy[:, 0], xy[:, 1]] = \
                    gt_boxes[hit_gt_ind[xy[:, 0], xy[:, 1]], :4]
                gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

                # cls
                cls_res = np.zeros((fm_height, fm_width))
                cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
                cls_res_list.append(cls_res.reshape(-1))

                # centerness
                center_res = np.zeros((fm_height, fm_width))
                center_res[xy[:, 0], xy[:, 1]] = center[
                    xy[:, 0] * stride, xy[:, 1] * stride,
                    hit_gt_ind[xy[:, 0], xy[:, 1]]]
                ctr_res_list.append(center_res.reshape(-1))

            cls_res_final = np.concatenate(cls_res_list, axis=0)[:, np.newaxis]
            ctr_res_final = np.concatenate(ctr_res_list, axis=0)[:, np.newaxis]
            gt_boxes_res_final = np.concatenate(gt_boxes_res_list, axis=0)
            return np.concatenate(
                [cls_res_final, ctr_res_final, gt_boxes_res_final], axis=1)
        def get_fcos_target_batch(gtboxes_batch, img_batch, fm_size_list):
            fcos_target_batch = []
            for i in range(batch_size):
                gt_target = fcos_target(gtboxes_batch[i], img_batch, fm_size_list,i)
                fcos_target_batch.append(gt_target)
            return np.array(fcos_target_batch, np.float32)

        img_batch = [self.img_height,self.img_width]
        fm_size_list = []
        for level in self.predictor_sizes:
            featuremap_height, featuremap_width = level[0],level[1]
            fm_size_list.append([featuremap_height, featuremap_width])

        fcos_target_batch = get_fcos_target_batch(ground_truth_labels, img_batch, fm_size_list)
        fcos_target_batch = np.reshape(fcos_target_batch, [batch_size, -1, 6])

        return  fcos_target_batch


class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
