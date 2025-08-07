import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional

# Import utilities
from utils.util_tf import make_anchors, wh2xy, xy2wh, box_iou


class ComputeLoss:
    """
    YOLOv8 Loss computation class for TensorFlow
    """
    
    def __init__(self, model, params=None):
        """
        Initialize loss computation
        
        Args:
            model: YOLOv8 model
            params: Loss parameters dictionary
        """
        self.model = model
        self.stride = model.stride
        self.nc = model.num_classes
        self.no = model.detection_head.num_outputs
        
        # Default parameters
        self.params = params or {
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        }
        
        # Task aligned assigner parameters
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9
        
        # DFL parameters
        self.dfl_ch = 16
        self.project = tf.range(self.dfl_ch, dtype=tf.float32)
        
        # BCE loss for classification
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        
    def __call__(self, outputs, targets):
        """
        Compute loss
        
        Args:
            outputs: Model outputs (list of tensors from detection heads)
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
        
        Returns:
            Total loss value
        """
        device_type = outputs[0].dtype
        
        # Reshape and concatenate outputs
        batch_size = tf.shape(outputs[0])[0]
        output_cat = []
        
        for i, xi in enumerate(outputs):
            # Reshape from [batch, height, width, channels] to [batch, channels, -1]
            b, h, w, c = xi.shape
            xi = tf.reshape(xi, [b, c, h * w])
            output_cat.append(xi)
        
        # Concatenate along anchor dimension
        output = tf.concat(output_cat, axis=2)  # [batch, no, total_anchors]
        output = tf.transpose(output, [0, 2, 1])  # [batch, total_anchors, no]
        
        # Split predictions
        pred_output = output[..., :4 * self.dfl_ch]
        pred_scores = output[..., 4 * self.dfl_ch:]
        
        # Generate anchors
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, 0.5)
        
        # Prepare targets
        gt_labels, gt_bboxes, mask_gt = self._prepare_targets(targets, batch_size)
        
        # Process predictions
        pred_bboxes = self._decode_bboxes(pred_output, anchor_points)
        
        # Assign targets
        target_bboxes, target_scores, fg_mask = self._assign(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, anchor_points * stride_tensor
        )
        
        # Scale targets
        target_bboxes = target_bboxes / stride_tensor
        target_scores_sum = tf.reduce_sum(target_scores)
        
        # Classification loss
        loss_cls = self._classification_loss(pred_scores, target_scores, target_scores_sum)
        
        # Box and DFL losses
        loss_box, loss_dfl = self._bbox_loss(
            pred_output, pred_bboxes, target_bboxes, target_scores, 
            fg_mask, target_scores_sum, anchor_points
        )
        
        # Apply loss weights and return total
        loss_cls *= self.params['cls']
        loss_box *= self.params['box'] 
        loss_dfl *= self.params['dfl']
        
        total_loss = loss_cls + loss_box + loss_dfl
        
        return total_loss
    
    def _prepare_targets(self, targets, batch_size):
        """
        Prepare ground truth targets
        
        Args:
            targets: Raw targets [batch_idx, class, x, y, w, h]
            batch_size: Batch size
        
        Returns:
            gt_labels: Ground truth labels
            gt_bboxes: Ground truth bboxes
            mask_gt: Valid mask
        """
        if tf.shape(targets)[0] == 0:
            gt_labels = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            gt_bboxes = tf.zeros([batch_size, 0, 4], dtype=tf.float32)
            mask_gt = tf.zeros([batch_size, 0, 1], dtype=tf.bool)
            return gt_labels, gt_bboxes, mask_gt
        
        # Get batch indices and counts
        batch_idx = tf.cast(targets[:, 0], tf.int32)
        _, _, counts = tf.unique_with_counts(batch_idx)
        max_boxes = tf.reduce_max(counts)
        
        # Initialize tensors
        gt_labels = tf.zeros([batch_size, max_boxes, 1], dtype=tf.float32)
        gt_bboxes = tf.zeros([batch_size, max_boxes, 4], dtype=tf.float32)
        mask_gt = tf.zeros([batch_size, max_boxes, 1], dtype=tf.bool)
        
        # Fill in ground truth for each batch
        for i in range(batch_size):
            mask = batch_idx == i
            num_gt = tf.reduce_sum(tf.cast(mask, tf.int32))
            
            if num_gt > 0:
                gt_targets = tf.boolean_mask(targets, mask)
                indices = tf.stack([
                    tf.fill([num_gt], i),
                    tf.range(num_gt)
                ], axis=1)
                
                # Update labels
                gt_labels = tf.tensor_scatter_nd_update(
                    gt_labels,
                    indices,
                    tf.expand_dims(gt_targets[:, 1], axis=-1)
                )
                
                # Update bboxes (convert from xywh to xyxy)
                gt_boxes_xyxy = wh2xy(gt_targets[:, 2:6])
                gt_bboxes = tf.tensor_scatter_nd_update(
                    gt_bboxes,
                    indices,
                    gt_boxes_xyxy
                )
                
                # Update mask
                mask_gt = tf.tensor_scatter_nd_update(
                    mask_gt,
                    indices,
                    tf.ones([num_gt, 1], dtype=tf.bool)
                )
        
        return gt_labels, gt_bboxes, mask_gt
    
    def _decode_bboxes(self, pred_output, anchor_points):
        """
        Decode predicted bounding boxes using DFL
        
        Args:
            pred_output: Raw box predictions
            anchor_points: Anchor points
        
        Returns:
            Decoded bounding boxes
        """
        b, a, c = pred_output.shape
        
        # Reshape for DFL processing
        pred_dist = tf.reshape(pred_output, [b, a, 4, self.dfl_ch])
        pred_dist = tf.nn.softmax(pred_dist, axis=-1)
        
        # Apply DFL (Distribution Focal Loss) - weighted sum
        pred_dist = tf.reduce_sum(
            pred_dist * tf.reshape(self.project, [1, 1, 1, self.dfl_ch]),
            axis=-1
        )
        
        # Split into lt (left-top) and rb (right-bottom)
        lt, rb = tf.split(pred_dist, 2, axis=-1)
        
        # Decode boxes: x1y1 = anchor - lt, x2y2 = anchor + rb
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        
        # Concatenate to get full boxes
        pred_bboxes = tf.concat([x1y1, x2y2], axis=-1)
        
        return pred_bboxes
    
    def _assign(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, anchors):
        """
        Task-aligned assignment of ground truth to predictions
        
        Args:
            pred_scores: Predicted class scores
            pred_bboxes: Predicted bounding boxes
            gt_labels: Ground truth labels
            gt_bboxes: Ground truth boxes
            mask_gt: Valid ground truth mask
            anchors: Anchor points
        
        Returns:
            target_bboxes: Assigned target boxes
            target_scores: Assigned target scores
            fg_mask: Foreground mask
        """
        batch_size = tf.shape(pred_scores)[0]
        num_anchors = tf.shape(pred_scores)[1]
        num_classes = tf.shape(pred_scores)[2]
        num_gt = tf.shape(gt_labels)[1]
        
        if num_gt == 0:
            return (
                tf.zeros_like(pred_bboxes),
                tf.zeros_like(pred_scores),
                tf.zeros([batch_size, num_anchors], dtype=tf.bool)
            )
        
        # Compute IoU between predictions and ground truth
        # Shape: [batch, num_gt, num_anchors]
        overlaps = self._batch_iou(gt_bboxes, pred_bboxes)
        
        # Get alignment metric
        # Extract scores for ground truth classes
        batch_ind = tf.range(batch_size)[:, None, None]
        gt_ind = tf.range(num_gt)[None, :, None]
        anchor_ind = tf.range(num_anchors)[None, None, :]
        class_ind = tf.cast(gt_labels, tf.int32)
        
        # Get predicted scores for GT classes
        pred_scores_for_gt = tf.gather_nd(
            pred_scores,
            tf.stack([
                tf.broadcast_to(batch_ind, [batch_size, num_gt, num_anchors]),
                tf.broadcast_to(anchor_ind, [batch_size, num_gt, num_anchors]),
                tf.broadcast_to(class_ind, [batch_size, num_gt, num_anchors])
            ], axis=-1)
        )
        
        # Compute alignment metric
        align_metric = tf.pow(pred_scores_for_gt, self.alpha) * tf.pow(overlaps, self.beta)
        
        # Check if anchors are inside ground truth boxes
        is_in_gts = self._check_anchors_in_gts(anchors, gt_bboxes, mask_gt)
        align_metric = align_metric * tf.cast(is_in_gts, tf.float32)
        
        # Select top-k anchors for each ground truth
        top_k_metric, top_k_idx = tf.nn.top_k(align_metric, k=self.top_k, sorted=False)
        
        # Create assignment mask
        mask_topk = tf.reduce_sum(
            tf.one_hot(top_k_idx, num_anchors, dtype=tf.float32),
            axis=2
        )
        
        # Handle multiple assignments
        mask_pos = mask_topk * tf.cast(is_in_gts, tf.float32) * tf.cast(mask_gt, tf.float32)
        fg_mask = tf.reduce_sum(mask_pos, axis=1) > 0
        
        # Get target indices
        target_gt_idx = tf.argmax(mask_pos, axis=1, output_type=tf.int32)
        
        # Gather target boxes and labels
        batch_ind = tf.range(batch_size)[:, None]
        batch_ind = tf.broadcast_to(batch_ind, [batch_size, num_anchors])
        
        indices = tf.stack([batch_ind, target_gt_idx], axis=-1)
        target_bboxes = tf.gather_nd(gt_bboxes, indices)
        target_labels = tf.gather_nd(gt_labels, indices)
        
        # Create target scores (one-hot encoded)
        target_scores = tf.one_hot(
            tf.cast(target_labels[..., 0], tf.int32),
            num_classes,
            dtype=tf.float32
        )
        
        # Apply foreground mask
        target_scores = target_scores * tf.cast(fg_mask[..., None], tf.float32)
        
        # Normalize scores
        align_metric_sum = tf.reduce_sum(align_metric, axis=2)
        align_metric_sum = tf.where(
            align_metric_sum > 0,
            align_metric_sum,
            tf.ones_like(align_metric_sum)
        )
        norm_factor = tf.gather_nd(align_metric_sum, indices)
        target_scores = target_scores * norm_factor[..., None]
        
        return target_bboxes, target_scores, fg_mask
    
    def _batch_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes for entire batch
        
        Args:
            boxes1: [batch, N, 4]
            boxes2: [batch, M, 4]
        
        Returns:
            IoU tensor [batch, N, M]
        """
        # Expand dimensions for broadcasting
        boxes1 = tf.expand_dims(boxes1, axis=2)  # [batch, N, 1, 4]
        boxes2 = tf.expand_dims(boxes2, axis=1)  # [batch, 1, M, 4]
        
        # Compute intersection
        x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
        
        # Compute areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Compute union
        union = area1 + area2 - intersection
        
        # Compute IoU
        iou = intersection / (union + self.eps)
        
        return iou
    
    def _check_anchors_in_gts(self, anchors, gt_bboxes, mask_gt):
        """
        Check if anchors are inside ground truth boxes
        
        Args:
            anchors: Anchor points [num_anchors, 2]
            gt_bboxes: Ground truth boxes [batch, num_gt, 4]
            mask_gt: Valid GT mask [batch, num_gt, 1]
        
        Returns:
            Boolean mask [batch, num_gt, num_anchors]
        """
        batch_size = tf.shape(gt_bboxes)[0]
        num_gt = tf.shape(gt_bboxes)[1]
        num_anchors = tf.shape(anchors)[0]
        
        # Expand dimensions for broadcasting
        anchors = tf.reshape(anchors, [1, 1, num_anchors, 2])
        gt_bboxes = tf.reshape(gt_bboxes, [batch_size, num_gt, 1, 4])
        
        # Check if anchors are inside boxes
        x_inside = tf.logical_and(
            anchors[..., 0] >= gt_bboxes[..., 0],
            anchors[..., 0] <= gt_bboxes[..., 2]
        )
        y_inside = tf.logical_and(
            anchors[..., 1] >= gt_bboxes[..., 1],
            anchors[..., 1] <= gt_bboxes[..., 3]
        )
        
        is_inside = tf.logical_and(x_inside, y_inside)
        
        # Apply valid mask
        mask_gt = tf.reshape(mask_gt, [batch_size, num_gt, 1])
        is_inside = tf.logical_and(is_inside, mask_gt)
        
        return is_inside
    
    def _classification_loss(self, pred_scores, target_scores, target_scores_sum):
        """
        Compute classification loss
        
        Args:
            pred_scores: Predicted scores
            target_scores: Target scores
            target_scores_sum: Sum of target scores for normalization
        
        Returns:
            Classification loss
        """
        # Binary cross entropy loss
        loss = tf.keras.losses.binary_crossentropy(
            target_scores,
            pred_scores,
            from_logits=True
        )
        
        # Sum and normalize
        loss = tf.reduce_sum(loss) / (target_scores_sum + self.eps)
        
        return loss
    
    def _bbox_loss(self, pred_dist, pred_bboxes, target_bboxes, target_scores,
                   fg_mask, target_scores_sum, anchor_points):
        """
        Compute bounding box regression and DFL losses
        
        Args:
            pred_dist: Predicted distributions for DFL
            pred_bboxes: Predicted boxes
            target_bboxes: Target boxes
            target_scores: Target scores
            fg_mask: Foreground mask
            target_scores_sum: Sum for normalization
            anchor_points: Anchor points
        
        Returns:
            Box loss and DFL loss
        """
        if tf.reduce_sum(tf.cast(fg_mask, tf.float32)) == 0:
            return tf.constant(0.0), tf.constant(0.0)
        
        # Get foreground predictions and targets
        fg_pred_bboxes = tf.boolean_mask(pred_bboxes, fg_mask)
        fg_target_bboxes = tf.boolean_mask(target_bboxes, fg_mask)
        
        # Weight for each box (sum of class scores)
        weight = tf.reduce_sum(target_scores, axis=-1)
        fg_weight = tf.boolean_mask(weight, fg_mask)
        fg_weight = tf.expand_dims(fg_weight, axis=-1)
        
        # IoU loss
        iou = self._ciou(fg_pred_bboxes, fg_target_bboxes)
        loss_box = tf.reduce_sum((1.0 - iou) * fg_weight) / (target_scores_sum + self.eps)
        
        # DFL loss
        fg_pred_dist = tf.boolean_mask(pred_dist, fg_mask)
        fg_anchor_points = tf.boolean_mask(
            tf.broadcast_to(anchor_points, [tf.shape(pred_dist)[0], tf.shape(anchor_points)[0], 2]),
            fg_mask
        )
        
        # Convert target boxes to distances from anchors
        target_lt = fg_anchor_points - fg_target_bboxes[..., :2]
        target_rb = fg_target_bboxes[..., 2:] - fg_anchor_points
        target_dist = tf.concat([target_lt, target_rb], axis=-1)
        target_dist = tf.clip_by_value(target_dist, 0, self.dfl_ch - 1.01)
        
        # Compute DFL loss
        loss_dfl = self._df_loss(fg_pred_dist, target_dist, fg_weight)
        loss_dfl = loss_dfl / (target_scores_sum + self.eps)
        
        return loss_box, loss_dfl
    
    def _ciou(self, boxes1, boxes2):
        """
        Compute Complete IoU
        
        Args:
            boxes1: First set of boxes
            boxes2: Second set of boxes
        
        Returns:
            CIoU values
        """
        # Basic IoU
        x1_min, y1_min, x1_max, y1_max = tf.split(boxes1, 4, axis=-1)
        x2_min, y2_min, x2_max, y2_max = tf.split(boxes2, 4, axis=-1)
        
        # Intersection
        inter_xmin = tf.maximum(x1_min, x2_min)
        inter_ymin = tf.maximum(y1_min, y2_min)
        inter_xmax = tf.minimum(x1_max, x2_max)
        inter_ymax = tf.minimum(y1_max, y2_max)
        
        inter_area = tf.maximum(0.0, inter_xmax - inter_xmin) * \
                    tf.maximum(0.0, inter_ymax - inter_ymin)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union + self.eps)
        
        # Center distance
        cx1 = (x1_min + x1_max) / 2
        cy1 = (y1_min + y1_max) / 2
        cx2 = (x2_min + x2_max) / 2
        cy2 = (y2_min + y2_max) / 2
        
        center_dist = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)
        
        # Enclosing box
        enc_xmin = tf.minimum(x1_min, x2_min)
        enc_ymin = tf.minimum(y1_min, y2_min)
        enc_xmax = tf.maximum(x1_max, x2_max)
        enc_ymax = tf.maximum(y1_max, y2_max)
        
        enc_diag = tf.square(enc_xmax - enc_xmin) + tf.square(enc_ymax - enc_ymin)
        
        # Aspect ratio
        w1 = x1_max - x1_min
        h1 = y1_max - y1_min
        w2 = x2_max - x2_min
        h2 = y2_max - y2_min
        
        v = (4 / (math.pi ** 2)) * tf.square(
            tf.atan(w2 / (h2 + self.eps)) - tf.atan(w1 / (h1 + self.eps))
        )
        
        alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - (center_dist / (enc_diag + self.eps) + alpha * v)
        
        return ciou
    
    def _df_loss(self, pred_dist, target_dist, weight):
        """
        Distribution Focal Loss
        
        Args:
            pred_dist: Predicted distribution [N, 4*dfl_ch]
            target_dist: Target distribution [N, 4]
            weight: Loss weight [N, 1]
        
        Returns:
            DFL loss
        """
        # Reshape predictions
        pred_dist = tf.reshape(pred_dist, [-1, 4, self.dfl_ch])
        
        # Get target left and right indices
        target_left = tf.cast(tf.floor(target_dist), tf.int32)
        target_right = target_left + 1
        
        # Get weights for left and right
        weight_left = tf.cast(target_right, tf.float32) - target_dist
        weight_right = 1 - weight_left
        
        # Clip indices
        target_left = tf.clip_by_value(target_left, 0, self.dfl_ch - 1)
        target_right = tf.clip_by_value(target_right, 0, self.dfl_ch - 1)
        
        # Compute cross entropy for left and right
        loss_left = tf.keras.losses.sparse_categorical_crossentropy(
            target_left,
            pred_dist,
            from_logits=True
        )
        loss_right = tf.keras.losses.sparse_categorical_crossentropy(
            target_right,
            pred_dist,
            from_logits=True
        )
        
        # Weighted combination
        loss = weight_left * loss_left + weight_right * loss_right
        
        # Apply weight and sum
        loss = tf.reduce_mean(loss, axis=-1, keepdims=True)
        loss = tf.reduce_sum(loss * weight)
        
        return loss


# Export
__all__ = ['ComputeLoss']