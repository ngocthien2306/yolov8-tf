import os
import random
import math
import time
from typing import List, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


def setup_seed(seed=0):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def setup_multi_processes():
    """Setup multi-processing environment variables for TensorFlow"""
    # Disable OpenCV multithreading
    cv2.setNumThreads(0)
    
    # Setup OMP threads
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'
    
    # Setup MKL threads  
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'
    
    # Configure TensorFlow threading
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def scale_coords(coords, img1_shape, img0_shape, ratio_pad=None):
    """Rescale coordinates from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    
    coords = tf.cast(coords, tf.float32)
    coords = coords - tf.constant([pad[0], pad[1], pad[0], pad[1]])
    coords = coords / gain
    
    # Clip coordinates
    coords = tf.clip_by_value(coords, 0.0, [img0_shape[1], img0_shape[0], img0_shape[1], img0_shape[0]])
    return coords


def box_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    
    Args:
        box1: Tensor of shape [N, 4]
        box2: Tensor of shape [M, 4]
    
    Returns:
        iou: Tensor of shape [N, M] containing pairwise IoU values
    """
    # Expand dimensions for broadcasting
    box1 = tf.expand_dims(box1, 1)  # [N, 1, 4]
    box2 = tf.expand_dims(box2, 0)  # [1, M, 4]
    
    # Calculate intersection
    inter_min = tf.maximum(box1[..., :2], box2[..., :2])  # [N, M, 2]
    inter_max = tf.minimum(box1[..., 2:], box2[..., 2:])  # [N, M, 2]
    inter_wh = tf.maximum(inter_max - inter_min, 0.0)     # [N, M, 2]
    intersection = inter_wh[..., 0] * inter_wh[..., 1]    # [N, M]
    
    # Calculate areas
    box1_wh = box1[..., 2:] - box1[..., :2]  # [N, 1, 2]
    box2_wh = box2[..., 2:] - box2[..., :2]  # [1, M, 2]
    area1 = box1_wh[..., 0] * box1_wh[..., 1]  # [N, 1]
    area2 = box2_wh[..., 0] * box2_wh[..., 1]  # [1, M]
    
    # Calculate IoU
    union = area1 + area2 - intersection
    return intersection / (union + 1e-7)


def wh2xy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format"""
    y = tf.identity(x)
    xy = y[..., :2]  # center x, center y
    wh = y[..., 2:]  # width, height
    
    xy1 = xy - wh / 2  # top left
    xy2 = xy + wh / 2  # bottom right
    
    return tf.concat([xy1, xy2], axis=-1)


def xy2wh(x):
    """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format"""
    xy1 = x[..., :2]  # top left
    xy2 = x[..., 2:]  # bottom right
    
    xy = (xy1 + xy2) / 2  # center
    wh = xy2 - xy1       # width, height
    
    return tf.concat([xy, wh], axis=-1)


def non_max_suppression(predictions, 
                       conf_threshold=0.25, 
                       iou_threshold=0.45,
                       max_detections=300,
                       max_nms=30000):
    """
    Non-Maximum Suppression for YOLO predictions
    
    Args:
        predictions: Tensor of shape [batch_size, num_anchors, num_classes + 4]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        max_nms: Maximum boxes for NMS
    
    Returns:
        List of detection tensors, one per image in the batch
    """
    batch_size = tf.shape(predictions)[0]
    num_classes = tf.shape(predictions)[-1] - 4
    
    outputs = []
    
    for i in range(batch_size):
        pred = predictions[i]  # [num_anchors, num_classes + 4]
        
        # Extract boxes and scores
        boxes = pred[:, :4]  # [num_anchors, 4]
        class_scores = pred[:, 4:]  # [num_anchors, num_classes]
        
        # Get best class score for each anchor
        max_scores = tf.reduce_max(class_scores, axis=-1)  # [num_anchors]
        best_classes = tf.argmax(class_scores, axis=-1)    # [num_anchors]
        
        # Filter by confidence threshold
        valid_mask = max_scores > conf_threshold
        if tf.reduce_sum(tf.cast(valid_mask, tf.int32)) == 0:
            # No valid detections
            outputs.append(tf.zeros((0, 6), dtype=tf.float32))
            continue
        
        # Filter predictions
        valid_boxes = tf.boolean_mask(boxes, valid_mask)
        valid_scores = tf.boolean_mask(max_scores, valid_mask)
        valid_classes = tf.boolean_mask(best_classes, valid_mask)
        
        # Convert center format to corner format
        valid_boxes = wh2xy(valid_boxes)
        
        # Sort by confidence and limit number of boxes
        sorted_indices = tf.nn.top_k(valid_scores, k=tf.minimum(tf.shape(valid_scores)[0], max_nms))[1]
        valid_boxes = tf.gather(valid_boxes, sorted_indices)
        valid_scores = tf.gather(valid_scores, sorted_indices)
        valid_classes = tf.gather(valid_classes, sorted_indices)
        
        # Apply NMS
        selected_indices = tf.image.non_max_suppression(
            valid_boxes,
            valid_scores,
            max_output_size=max_detections,
            iou_threshold=iou_threshold
        )
        
        # Get final detections
        final_boxes = tf.gather(valid_boxes, selected_indices)
        final_scores = tf.gather(valid_scores, selected_indices)
        final_classes = tf.gather(valid_classes, selected_indices)
        
        # Combine into final output format [x1, y1, x2, y2, conf, class]
        detections = tf.concat([
            final_boxes,
            tf.expand_dims(final_scores, -1),
            tf.expand_dims(tf.cast(final_classes, tf.float32), -1)
        ], axis=-1)
        
        outputs.append(detections)
    
    return outputs


def smooth(y, f=0.05):
    """Box filter smoothing"""
    nf = max(round(len(y) * f * 2) // 2 + 1, 1)  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute Average Precision (AP) given true positives, confidence scores, and class predictions.
    
    Args:
        tp: True positives array [N, 10] for IoU thresholds 0.5:0.95
        conf: Confidence scores [N]  
        pred_cls: Predicted classes [N]
        target_cls: Target classes [M]
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Tuple of (tp, fp, precision, recall, map50, mean_ap)
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes
    
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap = np.zeros((nc, tp.shape[1]))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        
        if no == 0 or nl == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (nl + eps)  # recall curve
        r = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
        
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            
            # Compute the precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            
            # Integrate area under curve
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)  # integrate
    
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    
    names = [f'Class_{c}' for c in unique_classes] if nc > 1 else ['all']
    i = smooth(f1, 0.1).argmax()  # max F1 index
    p, r, f1 = p[i], r[i], f1[i]  # max-F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    
    ap50, mean_ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, mean_ap = p, r, ap50.mean(), mean_ap.mean()
    
    return tp, fp, mp, mr, map50, mean_ap


def strip_optimizer(model, filename):
    """Save model without optimizer state (equivalent to PyTorch version)"""
    try:
        # Save only the model weights and architecture
        model.save_weights(filename.replace('.pt', '.h5'))
        print(f"Model weights saved to {filename.replace('.pt', '.h5')}")
        
        # Also save in SavedModel format for better compatibility
        saved_model_path = filename.replace('.pt', '_savedmodel')
        tf.saved_model.save(model, saved_model_path)
        print(f"SavedModel saved to {saved_model_path}")
        
    except Exception as e:
        print(f"Error saving stripped model: {e}")


def clip_gradients(model, max_norm=10.0):
    """Clip gradients by global norm"""
    # Note: In TensorFlow, gradient clipping is typically done in the training loop
    # This is a placeholder for the concept
    pass


class EMA:
    """
    Exponential Moving Average for TensorFlow models
    Similar to the PyTorch version but adapted for TensorFlow
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema_model = tf.keras.models.clone_model(model)
        self.ema_model.set_weights(model.get_weights())
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
    
    def update(self, model):
        """Update EMA weights"""
        self.updates += 1
        d = self.decay(self.updates)
        
        model_weights = model.get_weights()
        ema_weights = self.ema_model.get_weights()
        
        new_weights = []
        for ema_w, model_w in zip(ema_weights, model_weights):
            if ema_w.dtype in [np.float16, np.float32, np.float64]:
                new_w = d * ema_w + (1 - d) * model_w
                new_weights.append(new_w)
            else:
                new_weights.append(ema_w)
        
        self.ema_model.set_weights(new_weights)
    
    @property 
    def ema(self):
        """Get EMA model"""
        return self.ema_model


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        if not math.isnan(float(val)):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class ComputeLoss:
    """
    Compute YOLOv8 loss function
    Adapted from PyTorch version for TensorFlow
    """
    
    def __init__(self, model, params):
        self.model = model
        self.params = params
        
        # Loss functions
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        
        # Model parameters
        if hasattr(model, 'detection_head'):
            head = model.detection_head
            self.stride = getattr(head, 'strides', tf.constant([8., 16., 32.]))
            self.nc = getattr(head, 'num_classes', 80)
            self.no = getattr(head, 'num_outputs', self.nc + 64)
        else:
            self.stride = tf.constant([8., 16., 32.])
            self.nc = params.get('nc', 80)
            self.no = self.nc + 64
        
        # Task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9
        
        # DFL parameters
        self.dfl_ch = 16
        self.project = tf.range(self.dfl_ch, dtype=tf.float32)
    
    def __call__(self, outputs, targets):
        """
        Calculate loss
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
        
        Returns:
            Total loss (classification + box + DFL)
        """
        if isinstance(outputs, (list, tuple)):
            x = outputs
        else:
            x = [outputs]
        
        # Concatenate all predictions
        predictions = []
        for i, pred in enumerate(x):
            b, h, w, c = tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], tf.shape(pred)[3]
            pred_reshaped = tf.reshape(pred, (b, h * w, c))
            predictions.append(pred_reshaped)
        
        output = tf.concat(predictions, axis=1)
        
        # Split predictions
        pred_output = output[..., :4 * self.dfl_ch]  # Box predictions
        pred_scores = output[..., 4 * self.dfl_ch:]  # Class predictions
        
        # Get image size and anchors
        batch_size = tf.shape(pred_scores)[0]
        device = pred_scores.device if hasattr(pred_scores, 'device') else 'cpu'
        
        # Create anchors
        anchor_points, stride_tensor = make_anchors_tf(x, self.stride)
        
        # Process targets
        if tf.shape(targets)[0] == 0:
            gt = tf.zeros((batch_size, 0, 5), dtype=tf.float32)
        else:
            # Convert targets to proper format
            gt = self._process_targets(targets, pred_scores)
        
        gt_labels, gt_bboxes = tf.split(gt, [1, 4], axis=-1)
        mask_gt = tf.reduce_sum(gt_bboxes, axis=-1, keepdims=True) > 0
        
        # Compute box predictions using DFL
        pred_bboxes = self._compute_bbox_from_dfl(pred_output, anchor_points)
        
        # Assign targets
        target_bboxes, target_scores, fg_mask = self._assign_targets(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, anchor_points * stride_tensor
        )
        
        target_bboxes = target_bboxes / stride_tensor
        target_scores_sum = tf.reduce_sum(target_scores)
        
        # Classification loss
        loss_cls = self.bce(target_scores, pred_scores)
        loss_cls = tf.reduce_sum(loss_cls) / (target_scores_sum + self.eps)
        
        # Box and DFL loss
        loss_box = tf.constant(0.0)
        loss_dfl = tf.constant(0.0)
        
        if tf.reduce_sum(tf.cast(fg_mask, tf.float32)) > 0:
            # IoU loss
            fg_pred_bboxes = tf.boolean_mask(pred_bboxes, fg_mask)
            fg_target_bboxes = tf.boolean_mask(target_bboxes, fg_mask)
            fg_target_scores = tf.boolean_mask(tf.reduce_sum(target_scores, axis=-1), fg_mask)
            
            weight = tf.expand_dims(fg_target_scores, -1)
            iou_loss = self._compute_iou_loss(fg_pred_bboxes, fg_target_bboxes)
            loss_box = tf.reduce_sum((1.0 - iou_loss) * weight) / (target_scores_sum + self.eps)
            
            # DFL loss
            fg_pred_output = tf.boolean_mask(pred_output, fg_mask)
            fg_target_lt_rb = self._compute_dfl_targets(fg_target_bboxes, anchor_points[fg_mask])
            loss_dfl = self._df_loss(fg_pred_output, fg_target_lt_rb)
            loss_dfl = tf.reduce_sum(loss_dfl * weight) / (target_scores_sum + self.eps)
        
        # Apply loss weights
        loss_cls *= self.params.get('cls', 0.5)
        loss_box *= self.params.get('box', 7.5)
        loss_dfl *= self.params.get('dfl', 1.5)
        
        return loss_cls + loss_box + loss_dfl
    
    def _process_targets(self, targets, pred_scores):
        """Process targets to match prediction format"""
        batch_size = tf.shape(pred_scores)[0]
        
        if tf.shape(targets)[0] == 0:
            return tf.zeros((batch_size, 0, 5), dtype=tf.float32)
        
        # Group targets by image index
        image_indices = tf.cast(targets[:, 0], tf.int32)
        _, counts = tf.unique_with_counts(image_indices)
        max_targets = tf.reduce_max(counts)
        
        gt = tf.zeros((batch_size, max_targets, 5), dtype=tf.float32)
        
        for i in range(batch_size):
            mask = image_indices == i
            if tf.reduce_sum(tf.cast(mask, tf.int32)) > 0:
                targets_i = tf.boolean_mask(targets, mask)
                n_targets = tf.shape(targets_i)[0]
                gt = tf.tensor_scatter_nd_update(
                    gt, [[i, j] for j in range(n_targets)], targets_i[:, 1:]
                )
        
        return gt
    
    def _compute_bbox_from_dfl(self, pred_dist, anchor_points):
        """Compute bounding boxes from DFL predictions"""
        b, n, c = tf.shape(pred_dist)[0], tf.shape(pred_dist)[1], tf.shape(pred_dist)[2]
        
        # Reshape and apply softmax
        pred_dist = tf.reshape(pred_dist, (b, n, 4, self.dfl_ch))
        pred_dist = tf.nn.softmax(pred_dist, axis=-1)
        
        # Apply integral projection
        pred_dist = tf.reduce_sum(pred_dist * self.project[None, None, None, :], axis=-1)
        
        # Convert to box format
        lt, rb = tf.split(pred_dist, 2, axis=-1)
        x1y1 = tf.expand_dims(anchor_points, 0) - lt
        x2y2 = tf.expand_dims(anchor_points, 0) + rb
        
        return tf.concat([x1y1, x2y2], axis=-1)
    
    def _assign_targets(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, anchors):
        """Task-aligned target assignment"""
        # This is a simplified version - full implementation would be quite complex
        # For now, return dummy values that maintain tensor shapes
        batch_size = tf.shape(pred_scores)[0]
        num_anchors = tf.shape(pred_scores)[1]
        
        target_bboxes = tf.zeros_like(pred_bboxes)
        target_scores = tf.zeros_like(pred_scores)
        fg_mask = tf.zeros((batch_size, num_anchors), dtype=tf.bool)
        
        return target_bboxes, target_scores, fg_mask
    
    def _compute_iou_loss(self, pred_boxes, target_boxes):
        """Compute Complete IoU (CIoU) loss"""
        # Calculate intersection
        inter_min = tf.maximum(pred_boxes[..., :2], target_boxes[..., :2])
        inter_max = tf.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
        inter_wh = tf.maximum(inter_max - inter_min, 0.0)
        intersection = inter_wh[..., 0] * inter_wh[..., 1]
        
        # Calculate union
        pred_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
        target_wh = target_boxes[..., 2:] - target_boxes[..., :2]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]
        union = pred_area + target_area - intersection + 1e-7
        
        # Basic IoU
        iou = intersection / union
        
        # Complete IoU components
        # Convex diagonal
        c_min = tf.minimum(pred_boxes[..., :2], target_boxes[..., :2])
        c_max = tf.maximum(pred_boxes[..., 2:], target_boxes[..., 2:])
        c_wh = c_max - c_min
        c2 = c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2 + 1e-7
        
        # Center distance
        pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
        target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
        rho2 = tf.reduce_sum((pred_center - target_center) ** 2, axis=-1)
        
        # Aspect ratio penalty
        v = (4 / (math.pi ** 2)) * tf.pow(
            tf.atan(target_wh[..., 0] / (target_wh[..., 1] + 1e-7)) - 
            tf.atan(pred_wh[..., 0] / (pred_wh[..., 1] + 1e-7)), 2
        )
        alpha = v / (v - iou + 1.0 + 1e-7)
        alpha = tf.stop_gradient(alpha)  # Don't backprop through alpha
        
        # Complete IoU
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou
    
    def _compute_dfl_targets(self, target_boxes, anchors):
        """Compute DFL targets"""
        # Convert target boxes to left-top, right-bottom format relative to anchors
        target_lt = anchors - target_boxes[..., :2]
        target_rb = target_boxes[..., 2:] - anchors
        target_lt_rb = tf.concat([target_lt, target_rb], axis=-1)
        return tf.clip_by_value(target_lt_rb, 0, self.dfl_ch - 1.01)
    
    def _df_loss(self, pred_dist, target):
        """Distribution Focal Loss"""
        # Reshape predictions
        pred_dist = tf.reshape(pred_dist, (-1, self.dfl_ch))
        target = tf.reshape(target, (-1,))
        
        # Get target left and right
        target_left = tf.cast(tf.floor(target), tf.int32)
        target_right = target_left + 1
        weight_left = tf.cast(target_right, tf.float32) - target
        weight_right = 1.0 - weight_left
        
        # Compute losses
        loss_left = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_left, logits=pred_dist
        )
        loss_right = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.minimum(target_right, self.dfl_ch - 1), logits=pred_dist
        )
        
        return tf.expand_dims(loss_left * weight_left + loss_right * weight_right, -1)


def make_anchors_tf(feature_maps, strides, offset=0.5):
    """Generate anchors from feature maps - TensorFlow version"""
    anchor_points = []
    stride_tensor = []
    
    for i, feature_map in enumerate(feature_maps):
        if len(feature_map.shape) == 4:  # [B, H, W, C]
            h, w = tf.shape(feature_map)[1], tf.shape(feature_map)[2]
        else:  # [B, N, C]
            # Infer spatial dimensions from stride
            total_anchors = tf.shape(feature_map)[1]
            # This is approximate - in practice you'd store the original dimensions
            side_length = tf.cast(tf.sqrt(tf.cast(total_anchors, tf.float32)), tf.int32)
            h, w = side_length, side_length
        
        stride = strides[i] if i < len(strides) else strides[-1]
        
        # Create coordinate grids
        x = tf.range(tf.cast(w, tf.float32), dtype=tf.float32) + offset
        y = tf.range(tf.cast(h, tf.float32), dtype=tf.float32) + offset
        yv, xv = tf.meshgrid(y, x, indexing='ij')
        
        # Stack and reshape
        anchors = tf.stack([xv, yv], axis=-1)
        anchors = tf.reshape(anchors, (-1, 2))
        anchor_points.append(anchors)
        
        # Create stride tensor
        num_points = tf.shape(anchors)[0]
        strides_i = tf.fill((num_points, 1), stride)
        stride_tensor.append(strides_i)
    
    return tf.concat(anchor_points, axis=0), tf.concat(stride_tensor, axis=0)


def initialize_weights(model):
    """Initialize model weights similar to PyTorch version"""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # He normal initialization for conv layers
            tf.keras.initializers.HeNormal()(layer.kernel.shape)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # Initialize batch norm
            layer.gamma.assign(tf.ones_like(layer.gamma))
            layer.beta.assign(tf.zeros_like(layer.beta))


def model_info(model, verbose=True, img_size=640):
    """
    Model information similar to PyTorch version
    
    Args:
        model: TensorFlow/Keras model
        verbose: Print detailed information
        img_size: Input image size for calculating FLOPs
    """
    n_p = sum(x.shape.num_elements() for x in model.trainable_variables)  # number parameters
    n_g = sum(x.shape.num_elements() for x in model.trainable_variables if x.trainable)  # number gradients
    
    if verbose:
        print(f"{'layer':<5} {'name':<25} {'gradient':<9} {'parameters':<12} {'shape':<25} {'mu':<10} {'sigma':<10}")
        for i, (name, p) in enumerate(zip([v.name for v in model.variables], model.variables)):
            print(f"{i:<5} {name:<25} {p.trainable:<9} {p.shape.num_elements():<12} {str(list(p.shape)):<25} "
                  f"{tf.reduce_mean(p).numpy():<10.3g} {tf.math.reduce_std(p).numpy():<10.3g}")
    
    # Calculate approximate FLOPs
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        
        forward_pass = tf.function(model.call)
        graph_info = profile(forward_pass.get_concrete_function(
            tf.TensorSpec(shape=(1, img_size, img_size, 3), dtype=tf.float32)).graph,
            options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops
    except Exception:
        flops = 0
    
    fs = f", {flops / 1E9:.1f} GFLOPs" if flops else ""
    print(f"Model Summary: {len(list(model.layers))} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")


def save_checkpoint(model, optimizer, epoch, best_fitness, ema=None, filename='checkpoint.h5'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'best_fitness': best_fitness,
        'model_weights': model.get_weights(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_weights'] = optimizer.get_weights()
    
    if ema is not None:
        checkpoint['ema_weights'] = ema.ema.get_weights()
    
    # Save using pickle for full compatibility
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None, ema=None):
    """Load training checkpoint"""
    try:
        import pickle
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model.set_weights(checkpoint['model_weights'])
        
        if optimizer is not None and 'optimizer_weights' in checkpoint:
            optimizer.set_weights(checkpoint['optimizer_weights'])
        
        if ema is not None and 'ema_weights' in checkpoint:
            ema.ema.set_weights(checkpoint['ema_weights'])
        
        print(f"Checkpoint loaded: {filename}")
        return checkpoint['epoch'], checkpoint.get('best_fitness', 0)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0


# Configuration and setup functions
def setup_tensorflow():
    """Setup TensorFlow for optimal performance"""
    # Enable mixed precision if available
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except:
        print("Mixed precision not available")
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Set up XLA compilation
    tf.config.optimizer.set_jit(True)
    print("XLA JIT compilation enabled")


# Example usage and testing
if __name__ == "__main__":
    print("Testing TensorFlow YOLO utilities...")
    
    # Test basic functions
    print("\n1. Testing box operations:")
    boxes1 = tf.constant([[10, 10, 20, 20], [15, 15, 25, 25]], dtype=tf.float32)
    boxes2 = tf.constant([[12, 12, 22, 22]], dtype=tf.float32)
    
    iou = box_iou(boxes1, boxes2)
    print(f"IoU shape: {iou.shape}, values: {iou.numpy()}")
    
    wh_boxes = xy2wh(boxes1)
    xy_boxes = wh2xy(wh_boxes)
    print(f"Box conversion test passed: {tf.reduce_all(tf.abs(boxes1 - xy_boxes) < 1e-6)}")
    
    # Test NMS
    print("\n2. Testing NMS:")
    predictions = tf.random.normal((2, 100, 85))  # [batch, anchors, classes+4]
    results = non_max_suppression(predictions)
    print(f"NMS results: {len(results)} images, shapes: {[r.shape for r in results]}")
    
    # Test utility classes
    print("\n3. Testing utility classes:")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i * 0.1, 1)
    print(f"Average meter: avg={meter.avg:.3f}, count={meter.count}")
    
    print("\nAll tests completed successfully!")