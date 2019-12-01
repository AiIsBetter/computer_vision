'''
The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import tensorflow as tf
import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
class SSDLoss:
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 n_class = 12,
                 gt_max = 300):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
            gt_max(int) 每个imgae最大能接受的gt label数量.(max gt label num of one image)
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha
        self.n_class = n_class
        self.gt_num_max = gt_max
    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''

        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)

        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true1, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true1 (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12  (NEW Here)+5(gt label,xmin,ymin,xmax,ymax))`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        # arm total loss
        y_true = y_true1[:, :, :-5]

        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

        positives_arm = tf.reduce_sum(y_true[:, :, 1:self.n_class], axis=2,keepdims=True)
        y_true_arm = y_true[:,:,0:1]
        y_true_arm = tf.concat([y_true_arm, positives_arm], axis=-1)

        # 1: Compute the losses for class and box predictions for every box.
        classification_loss = tf.to_float(self.log_loss(y_true_arm[:,:,:], y_pred[:,:,0:2])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,2:6])) # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets.
        # Create masks for the positive and negative ground truth classes.
        negatives = y_true_arm[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = y_true_arm[:,:,1] # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)
        # n_positive = tf.Print(n_positive, [n_positive],
        #                        message='Debug message arm_n_positive:',
        #                        first_n=10000, summarize=100000)
        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)
        # Compute the classification loss for the negative default boxes (if there are any).
        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        # 这里是计算，最小的topk个负样本的loss值的坐标，方便后面取出，数量是self.neg_pos_ratio*正样本的数量
        # (Here is the coordinate of the loss value of the smallest topk negative sample, which is easy to take out later.total num is self.neg_pos_ratio*npositive )
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.
            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)
        class_loss_arm = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)
        # new 只计算正样本loss
        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).
        loc_loss_arm = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)
        total_loss_arm = (class_loss_arm + self.alpha * loc_loss_arm) / tf.maximum(1.0, n_positive)
        ########################odm loss###########################
        # # arm预测的loc 将其decode作为新的gtbox坐标(The loc predicted by arm takes its decode as a new gtbox coordinate )
        y_pred_decoded_raw = y_pred[:, :, 2:14]
        #(0-12)：gt(cx, cy ,w, h) prior(cx ,cy ,w, h) variance(0.1,0.1,0.2,0.2)
        # xmin, ymin, xmax, ymax decode
        hpref = tf.exp(y_pred_decoded_raw[:, :, 3:4] * y_pred_decoded_raw[:, :,  -1:])
        wpref = tf.exp(y_pred_decoded_raw[:, :, 2:3] * y_pred_decoded_raw[:, :, -2:-1])# exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)

        hpref = hpref * y_pred_decoded_raw[:, :,  -5:-4]
        wpref = wpref * y_pred_decoded_raw[:, :, -6:-5]

        cypref = y_pred_decoded_raw[:, :, 1:2] *y_pred_decoded_raw[:, :, -3:-2] * y_pred_decoded_raw[:, :, -5:-4]
        cxpref = y_pred_decoded_raw[:, :, 0:1] * y_pred_decoded_raw[:, :, -4:-3] * y_pred_decoded_raw[:, :, -6:-5]

        cypref = cypref + y_pred_decoded_raw[:, :,  -7:-6]
        cxpref = cxpref + y_pred_decoded_raw[:, :, -8:-7]


        xmin_a = (cxpref - wpref / 2.0)* 320  # Set xmin

        ymin_a =( cypref - hpref / 2.0 )* 320 # Set ymin

        xmax_a = (cxpref +wpref / 2.0 )* 320 # Set xmax
        ymax_a = (cypref + hpref / 2.0 )* 320 # Set ymax

        vol_anchors = (xmax_a - xmin_a) * (ymax_a - ymin_a)
        # 重新匹配所有gtbox和arm loc deode后的坐标，即将arm loc的坐标encode为gt，然后作为refine后的true给odm部分计算loss
        # Rematch all the coordinates after gtbox and arm loc deode, that is, the coordinate encode of arm loc is gt, and then calculate loss to the odm part as true after refine
        # gt_labels存放一个batch里面所有gtboxxe和class，(batch_size,anchors_id,(class,xmin, ymin, xmax, ymax))
        # Gt_labels holds all gtboxes and class in a batch
        gt_bboxes = y_true1[:, :, self.n_class + 12:]
        gt_labels = y_true1[:, :, self.n_class + 12:self.n_class + 13]
        #
        gt_num_max = self.gt_num_max

        # # 初始化各参数
        # Initialization parameters
        feat_labels = tf.cast(tf.zeros_like(y_true1[:,:,0:1]),tf.int32) # 存放默认框匹配的GTbox标签(Store refine anchor  matching gtbox tags )
        feat_scores = tf.zeros_like(y_true1[:,:,0:1])  # 存放默认框与匹配的GTbox的IOU（交并比）(Store refine anchor  matching gtbox iou )
        feat_matched = tf.cast(tf.zeros_like(y_true1[:,:,0:1]),tf.int32) #存放后续过滤等操作的标记样本,作为mask操作的判断依据(Store tag samples for subsequent filtering and other operations as the basis for judging mask operations )
        feat_gtnum = tf.cast(tf.zeros_like(y_true1[:,:,0:1]),tf.int32)
        feat_ymin = tf.zeros_like(y_true1[:,:,0:1])  # 存放默认框匹配到的GTbox的坐标信息(Store the coordinate information of the refine anchor matching gtbox)
        feat_xmin = tf.zeros_like(y_true1[:,:,0:1])
        feat_ymax = tf.zeros_like(y_true1[:,:,0:1])
        feat_xmax = tf.zeros_like(y_true1[:,:,0:1])

        def jaccard_with_anchors(label,bbox):  # 计算重叠度函数(cal iou)
            # 计算iou
            int_xmin = tf.maximum(label[:,:,0:1], bbox[:,:,0:1])
            int_ymin = tf.maximum(label[:,:,1:2], bbox[:,:,1:2])
            int_xmax = tf.minimum(label[:,:,2:3], bbox[:,:,2:3])
            int_ymax = tf.minimum(label[:,:,3:4], bbox[:,:,3:4])

            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w

            union_vol = vol_anchors - inter_vol + (label[:,:,2:3] - label[:,:,0:1]) * (label[:,:,3:4] - label[:,:,1:2])
            # iou scores
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard
        def condition(i, feat_labels, feat_scores,feat_gtnum, # 循环条件
                      feat_xmin, feat_ymin,
                      feat_xmax, feat_ymax):
            # 循环每个image内所有gt box(Loop through all gt box within each image)
            r = tf.less(tf.cast(i,dtype=tf.float32), gt_num_max)# tf.shape(labels)GTbox num，if i<=tf.shape(labels) return True
            return r

        def body(i, feat_labels, feat_scores, feat_gtnum, # 循环执行主体
                 feat_xmin, feat_ymin,feat_xmax, feat_ymax):
            """
            寻找每个GTbox与所有anchor的iou,根据每次iou的分数更新iou,大于上一步iou的就存入新的iou以及其它相应的标记和坐标值
            Find each gtbox with all anchor's iou, updates iou, greater than the previous iou's score based on each iou score save new iou and other
            corresponding marking and coordinate values
            """
            # Jaccard score.

            label = tf.concat([gt_bboxes[:,i:i+1,1:2], gt_bboxes[:,i:i+1,2:3],gt_bboxes[:,i:i+1,3:4],gt_bboxes[:,i:i+1,4:5]], axis=-1)
            bbox = tf.concat([xmin_a[:,:,0:1], ymin_a[:,:,0:1],xmax_a[:,:,0:1],ymax_a[:,:,0:1]], axis=-1)

            jaccard = jaccard_with_anchors(label,bbox)  # 计算每个batch的真实框与与arm decode生成的所有框的交并比(Calculate the intersection of the gt box of each batch with all boxes generated by the arm decode )

            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard,  feat_scores)  # 交并比是否比之前匹配的GTbox大(Intersection is larger than previous matching gtbox )
            mask1 = tf.equal(y_true_arm[:,:,0:1],1)
            mask1 = tf.logical_and(mask1,tf.greater_equal(y_pred[:,:,0:1],0.99))

            mask1 = tf.logical_not(mask1)
            mask = tf.logical_and(mask,mask1)
            imask = tf.cast(mask, tf.int32)  # 转型
            fmask = tf.cast(mask, tf.float32)  # dtype float32

            feat_labels = imask * tf.cast(gt_labels[:, i:i + 1, 0:1], tf.int32) + (1 - imask) * feat_labels  # 当imask为1时更新标签(1 - imask)即把交并比大的位置的mask变成0，其他位置变为1，变为0的位置更新标记值
            feat_gtnum = imask * tf.cast(i, tf.int32) + (1 - imask) * feat_gtnum  # When imask is 1, the update tag (1-imask) changes the mask of the intersection and larger position to 0, the other positions to 1, and the position to 0 to update the tag value

            feat_scores = tf.where(mask, jaccard, feat_scores)

            feat_xmin = fmask * label[:, :, 0:1] + (1 - fmask) * feat_xmin  # 当fmask为1.0时更新坐标信息(Update coordinate information when fmask is 1.0 )
            feat_ymin = fmask * label[:, :, 1:2] + (1 - fmask) * feat_ymin
            feat_xmax = fmask * label[:, :, 2:3] + (1 - fmask) * feat_xmax
            feat_ymax = fmask * label[:, :, 3:4] + (1 - fmask) * feat_ymax

            return [i + 1, feat_labels, feat_scores,feat_gtnum,
                    feat_xmin, feat_ymin, feat_xmax, feat_ymax]

        i = 0

        [i, feat_labels, feat_scores,feat_gtnum,feat_xmin, feat_ymin,
         feat_xmax, feat_ymax] = tf.while_loop(condition, body,  # tf.while_loop是一个循环函数condition是循环条件，body是循环体
                                               [i, feat_labels, feat_scores,  feat_gtnum,# 第三项是参数
                                                feat_xmin,feat_ymin,
                                                feat_xmax,feat_ymax])


        def condition2(i, feat_labels, feat_scores, feat_matched,feat_gtnum,feat_xmin, feat_ymin,feat_xmax, feat_ymax):

            r = tf.less(tf.cast(i, dtype=tf.float32),gt_num_max)  # tf.shape(labels)GTbox的个数，当i<=tf.shape(labels)是返回True

            return r
        def body2(i, feat_labels, feat_scores, feat_matched,feat_gtnum, # 循环执行主体
                 feat_xmin, feat_ymin,feat_xmax, feat_ymax):
            """这一步操作和上一步类似，不过是为每个gtbox匹配一个最大iou的anchor,同时标记这个anchor，方便第二步为每个anchor匹配一个gtbox
            This step is similar to the previous step, except that the anchor, that matches a maximum iou for each gtbox is tagged at the same
            time that the anchor, is convenient for the second step to match a gtbox for each anchor
            """
            #找寻每一个gtbox的最大iou anchor，然后标记成1保留，方便后续阈值过滤(Find the maximum iouanchor, for each gtbox and mark it as 1 reserved for subsequent threshold filtering )

            mask = tf.equal(feat_gtnum, i)
            # 取出feat_scores里面对应每个gtbox的iou(Take out the iou of each gtbox in the feat_scores )
            tmp = tf.where(mask,feat_scores,tf.zeros_like(feat_scores))
            # 计算每个gtbox的iou最大的anchor(Calculate the maximum anchor of the iou for each gtbox)
            max_score = tf.reduce_max(tf.reshape(tmp, shape=[1, -1]))
            # 将其坐标做成模板(Make its coordinates a template )
            mask = tf.equal(tmp, max_score)
            mask = tf.logical_and(mask, tf.greater(tmp, 0))

            mask = tf.logical_and(mask, tf.not_equal(feat_matched, 1))
            # 该模板在feat_matched里面标记为1(The template is marked 1 in the feat_matched )
            imask = tf.cast(mask, tf.int32)  # 转型

            feat_matched = imask * tf.cast(1, tf.int32) +  feat_matched  #

            return [i + 1, feat_labels, feat_scores,feat_matched,feat_gtnum,
                    feat_xmin, feat_ymin, feat_xmax, feat_ymax]

        i = 0
        [i, feat_labels, feat_scores,feat_matched,feat_gtnum,  feat_xmin, feat_ymin,feat_xmax, feat_ymax] = tf.while_loop(condition2, body2,  # tf.while_loop是一个循环函数condition是循环条件，body是循环体
                                               [i, feat_labels, feat_scores, feat_matched,feat_gtnum, feat_xmin,feat_ymin,feat_xmax,feat_ymax])

        mask =  tf.equal(feat_matched, 1)

        mask = tf.logical_or(mask,tf.greater_equal(feat_scores,0.5))
        feat_labels = tf.where(mask,feat_labels,tf.zeros_like(feat_labels))

        feat_xmin = tf.where(mask, feat_xmin, tf.zeros_like(feat_xmin))
        feat_ymin = tf.where(mask, feat_ymin, tf.zeros_like(feat_ymin))
        feat_xmax = tf.where(mask, feat_xmax, tf.zeros_like(feat_xmax))
        feat_ymax = tf.where(mask, feat_ymax, tf.zeros_like(feat_ymax))

        feat_matched = tf.where(mask, 2*tf.ones_like(feat_matched), feat_matched)

        # Transform to center / size. 转换回中心坐标以及宽高(Converted back to center coordinates and width and height )
        feat_cy = (feat_ymax + feat_ymin) / 2./320
        feat_cx = (feat_xmax + feat_xmin) / 2./320
        feat_h = (feat_ymax - feat_ymin)/320.
        feat_w = (feat_xmax - feat_xmin)/320.

        prior_scaling = [0.1,0.1,0.2,0.2]

        feat_cx = (feat_cx - cxpref) / (wpref * prior_scaling[0])
        feat_cy = (feat_cy - cypref) / (hpref * prior_scaling[1])  # refine框中心与匹配的真实框中心坐标偏差(Central coordinate deviation between refine anchor and matching gt Box )

        feat_w = tf.log(tf.maximum((feat_w) / (wpref),1e-15)) / prior_scaling[2]
        feat_h = tf.log(tf.maximum((feat_h) / (hpref),1e-15)) / prior_scaling[3]# 高和宽的偏差(Deviation of height and width )


        feat_labels1 = tf.cast(tf.one_hot(feat_labels, self.n_class, axis=-1),dtype=tf.int32)

        feat_labels_reshape = tf.reshape(feat_labels1, shape=[tf.shape(feat_labels1)[0], tf.shape(feat_labels1)[1], -1])
        # 生成新的y_true用来计算odm部分的loss(Generate a new y_true to calculate the loss of the odm section )
        y_refine = tf.concat([tf.cast(feat_labels_reshape,dtype=tf.float32),feat_cx,feat_cy,feat_w,feat_h,cxpref,cypref,wpref,hpref,y_pred[:,:,10:14]],axis = -1)

        # odm total loss
        classification_loss = tf.to_float(
            self.log_loss(y_refine[:, :, :-12], y_pred[:, :, 14:-12]))  # Output shape: (batch_size, n_boxes)

        localization_loss = tf.to_float(
            self.smooth_L1_loss(y_refine[:, :, self.n_class:self.n_class+4], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets.
        # Create masks for the positive and negative ground truth classes.
        negatives = y_refine[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_refine[:, :, 1:self.n_class], axis=-1))  # Tensor of shape (batch_size, n_boxes)

        mask = tf.equal(y_true_arm[:, :, 0], 1)
        mask = tf.logical_and(mask, tf.greater_equal(y_pred[:, :, 0], 0.99))
        # # 过滤负样本中iou>0.3的样本，这部分样本不计入loss(Filter samples with iou > 0. 3 in negative samples, which are not included in loss)
        mask1 = tf.not_equal(feat_matched[:,:,0],2)
        mask1 = tf.logical_and(mask1, tf.greater_equal(feat_scores[:,:,0], 0.3))
        mask = tf.logical_or(mask,mask1)

        # 将正负样本中满足mask条件的样本过滤掉，不计算loss回传更新参数(Filter out samples satisfying mask condition in positive and negative samples without calculating loss return update parameters)
        positives = tf.where(mask, tf.zeros_like(positives), positives)
        negatives = tf.where(mask, tf.zeros_like(negatives), negatives)

        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)
        # Compute the classification loss for the negative default boxes (if there are any).
        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                        dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min),
                                     n_neg_losses)
        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)  # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(
                                               neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(
                tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep,
                                           axis=-1)  # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)
        class_loss_odm = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).
        loc_loss_odm = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)
        # 4: Compute the total loss.
        total_loss_odm = (class_loss_odm + self.alpha * loc_loss_odm) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`

        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        # 将两部分loss相加(Add two parts of loss )
        total_loss = (total_loss_odm+total_loss_arm )* tf.to_float(batch_size)

        return total_loss
