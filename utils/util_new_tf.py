import copy
import math
import random
import time
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union


def setup_seed(seed=0):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_multi_processes():
    """
    Setup multi-processing environment variables for TensorFlow
    """
    import os
    from platform import system
    
    # Set TensorFlow threading configuration
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Setup environment variables
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    
    # Disable OpenCV multithreading to avoid conflicts
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


def make_anchors(feature_maps, strides, offset=0.5):
    """
    Generate anchors from feature maps
    
    Args:
        feature_maps: List of feature map tensors or shapes
        strides: List of stride values for each feature map
        offset: Center offset for anchors
    
    Returns:
        anchor_points: Concatenated anchor points
        stride_tensor: Concatenated stride values
    """
    anchor_points = []
    stride_tensor = []
    
    # Handle both tensor and list inputs for strides
    if isinstance(strides, tf.Tensor):
        strides_list = tf.unstack(strides)
    else:
        strides_list = strides
    
    for i in range(len(feature_maps)):
        if isinstance(feature_maps[i], tf.Tensor):
            shape = tf.shape(feature_maps[i])
            h, w = shape[1], shape[2]
        else:
            h, w = feature_maps[i][0], feature_maps[i][1]
            
        # Create grid points
        sx = tf.range(tf.cast(w, tf.float32), dtype=tf.float32) + offset
        sy = tf.range(tf.cast(h, tf.float32), dtype=tf.float32) + offset
        sy, sx = tf.meshgrid(sy, sx, indexing='ij')
        
        # Stack and reshape
        anchor = tf.stack([sx, sy], axis=-1)
        anchor = tf.reshape(anchor, [-1, 2])
        anchor_points.append(anchor)
        
        # Create stride tensor
        if isinstance(strides, tf.Tensor):
            stride_val = strides_list[i]
        else:
            stride_val = strides[i]
            
        num_anchors = tf.cast(h * w, tf.int32)
        stride_t = tf.fill([num_anchors, 1], tf.cast(stride_val, tf.float32))
        stride_tensor.append(stride_t)
    
    return tf.concat(anchor_points, axis=0), tf.concat(stride_tensor, axis=0)


def scale_boxes(boxes, shape1, shape2, ratio_pad=None):
    """
    Scale bounding boxes from one image size to another
    
    Args:
        boxes: Bounding boxes to scale [N, 4] in (x1, y1, x2, y2) format
        shape1: Current shape (height, width)
        shape2: Target shape (height, width)  
        ratio_pad: Optional ratio and padding values
    
    Returns:
        Scaled bounding boxes
    """
    if ratio_pad is None:
        # Calculate from shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    boxes = tf.identity(boxes)
    
    # Scale boxes
    boxes = tf.concat([
        (boxes[..., 0:1] - pad[0]) / gain,  # x1
        (boxes[..., 1:2] - pad[1]) / gain,  # y1
        (boxes[..., 2:3] - pad[0]) / gain,  # x2
        (boxes[..., 3:4] - pad[1]) / gain   # y2
    ], axis=-1)
    
    # Clip boxes to image bounds
    boxes = tf.concat([
        tf.clip_by_value(boxes[..., 0:1], 0, shape2[1]),  # x1
        tf.clip_by_value(boxes[..., 1:2], 0, shape2[0]),  # y1
        tf.clip_by_value(boxes[..., 2:3], 0, shape2[1]),  # x2
        tf.clip_by_value(boxes[..., 3:4], 0, shape2[0])   # y2
    ], axis=-1)
    
    return boxes


def wh2xy(x):
    """
    Convert [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        x: Box coordinates in [x, y, w, h] format
    
    Returns:
        Box coordinates in [x1, y1, x2, y2] format
    """
    y = tf.identity(x)
    y = tf.concat([
        x[..., 0:1] - x[..., 2:3] / 2,  # x1
        x[..., 1:2] - x[..., 3:4] / 2,  # y1
        x[..., 0:1] + x[..., 2:3] / 2,  # x2
        x[..., 1:2] + x[..., 3:4] / 2   # y2
    ], axis=-1)
    return y


def xy2wh(x):
    """
    Convert [x1, y1, x2, y2] to [x, y, w, h]
    
    Args:
        x: Box coordinates in [x1, y1, x2, y2] format
    
    Returns:
        Box coordinates in [x, y, w, h] format
    """
    y = tf.concat([
        (x[..., 0:1] + x[..., 2:3]) / 2,  # x center
        (x[..., 1:2] + x[..., 3:4]) / 2,  # y center
        x[..., 2:3] - x[..., 0:1],        # width
        x[..., 3:4] - x[..., 1:2]         # height
    ], axis=-1)
    return y


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        box1: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        box2: Tensor of shape [M, 4] in [x1, y1, x2, y2] format
    
    Returns:
        IoU tensor of shape [N, M]
    """
    # Expand dimensions for broadcasting
    box1 = tf.expand_dims(box1, axis=1)  # [N, 1, 4]
    box2 = tf.expand_dims(box2, axis=0)  # [1, M, 4]
    
    # Calculate intersection
    x1 = tf.maximum(box1[..., 0], box2[..., 0])
    y1 = tf.maximum(box1[..., 1], box2[..., 1])
    x2 = tf.minimum(box1[..., 2], box2[..., 2])
    y2 = tf.minimum(box1[..., 3], box2[..., 3])
    
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    # Calculate union
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)
    
    return iou


def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45, max_det=300):
    """
    Perform Non-Maximum Suppression on predictions
    
    Args:
        prediction: Model predictions [batch, num_anchors, 4 + num_classes]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections
    
    Returns:
        List of detections for each image in batch
    """
    batch_size = tf.shape(prediction)[0]
    num_classes = tf.shape(prediction)[2] - 4
    
    outputs = []
    
    for idx in range(batch_size):
        x = prediction[idx]  # [num_anchors, 4 + nc]
        
        # Split box and class predictions
        box = x[:, :4]
        cls = x[:, 4:]
        
        # Get max class score and class id
        cls_score = tf.reduce_max(cls, axis=-1)
        cls_id = tf.argmax(cls, axis=-1)
        
        # Filter by confidence threshold
        mask = cls_score > conf_threshold
        box = tf.boolean_mask(box, mask)
        cls_score = tf.boolean_mask(cls_score, mask)
        cls_id = tf.boolean_mask(cls_id, mask)
        
        if tf.shape(box)[0] == 0:
            outputs.append(tf.zeros((0, 6), dtype=tf.float32))
            continue
        
        # Convert box format from xywh to xyxy
        box = wh2xy(box)
        
        # Perform NMS
        selected_indices = tf.image.non_max_suppression(
            boxes=box,
            scores=cls_score,
            max_output_size=max_det,
            iou_threshold=iou_threshold
        )
        
        # Get selected boxes, scores, and classes
        selected_boxes = tf.gather(box, selected_indices)
        selected_scores = tf.gather(cls_score, selected_indices)
        selected_classes = tf.gather(cls_id, selected_indices)
        
        # Combine results [x1, y1, x2, y2, score, class]
        detections = tf.concat([
            selected_boxes,
            tf.expand_dims(selected_scores, axis=-1),
            tf.expand_dims(tf.cast(selected_classes, tf.float32), axis=-1)
        ], axis=-1)
        
        outputs.append(detections)
    
    return outputs


def smooth(y, f=0.05):
    """
    Box filter smoothing
    
    Args:
        y: Input array
        f: Fraction of filter
    
    Returns:
        Smoothed array
    """
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute Average Precision
    
    Args:
        tp: True positives
        conf: Confidence scores
        pred_cls: Predicted classes
        target_cls: Target classes
        eps: Small epsilon value
    
    Returns:
        Average Precision metrics
    """
    # Convert to numpy for easier manipulation
    tp = tp.numpy() if hasattr(tp, 'numpy') else tp
    conf = conf.numpy() if hasattr(conf, 'numpy') else conf
    pred_cls = pred_cls.numpy() if hasattr(pred_cls, 'numpy') else pred_cls
    target_cls = target_cls.numpy() if hasattr(target_cls, 'numpy') else target_cls
    
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    
    # Create Precision-Recall curve
    px = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]
        no = i.sum()
        
        if no == 0 or nl == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (nl + eps)
        
        # Precision
        precision = tpc / (tpc + fpc)
        
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = np.trapz(np.interp(px, recall[:, j], precision[:, j]), px)
    
    # Compute F1
    f1 = 2 * ap / (1 + ap + eps)
    
    return ap.mean(), f1.mean()


def clip_gradients(gradients, max_norm=10.0):
    """
    Clip gradients by norm
    
    Args:
        gradients: List of gradient tensors
        max_norm: Maximum gradient norm
    
    Returns:
        Clipped gradients
    """
    clipped_grads, _ = tf.clip_by_global_norm(gradients, max_norm)
    return clipped_grads


class AverageMeter:
    """Compute and store the average and current value"""
    
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
            self.avg = self.sum / self.count if self.count > 0 else 0


class EMA:
    """
    Exponential Moving Average for model weights
    
    Similar to PyTorch implementation but for TensorFlow
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Args:
            model: TensorFlow/Keras model
            decay: EMA decay rate
            tau: Tau for decay warmup
            updates: Number of updates performed
        """
        self.model = model
        self.decay_fn = lambda x: decay * (1 - tf.exp(-x / tau))
        self.updates = updates
        
        # Create EMA weights
        self.ema_weights = []
        for weight in model.trainable_weights:
            self.ema_weights.append(tf.Variable(weight.read_value(), 
                                                trainable=False,
                                                name=f"ema_{weight.name}"))
    
    def update(self, model):
        """Update EMA weights"""
        self.updates += 1
        decay = self.decay_fn(self.updates)
        
        for ema_weight, weight in zip(self.ema_weights, model.trainable_weights):
            ema_weight.assign(decay * ema_weight + (1 - decay) * weight)
    
    def apply(self, model=None):
        """Apply EMA weights to model"""
        if model is None:
            model = self.model
            
        for weight, ema_weight in zip(model.trainable_weights, self.ema_weights):
            weight.assign(ema_weight)


class ComputeLoss:
    """YOLOv8 loss computation"""
    
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module
            
        # Get model device and head
        m = model.detection_head
        
        self.stride = tf.constant([8., 16., 32.])  # model strides
        self.nc = m.num_classes  # number of classes
        self.no = m.num_outputs
        self.params = params
        
        # Task aligned assigner parameters
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9
        
        # DFL Loss params
        self.dfl_ch = m.dfl_channels
        self.project = tf.range(self.dfl_ch, dtype=tf.float32)
        
    def __call__(self, outputs, targets):
        # Prepare predictions
        if isinstance(outputs, tuple):
            x = outputs[1]
        else:
            x = outputs
            
        # Concatenate outputs from different scales
        output_list = []
        for i, output in enumerate(x):
            b, h, w, c = tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], tf.shape(output)[3]
            output_reshaped = tf.reshape(output, [b, self.no, h * w])
            output_list.append(output_reshaped)
        
        output = tf.concat(output_list, axis=2)  # [batch, outputs, anchors]
        pred_output, pred_scores = tf.split(output, [4 * self.dfl_ch, self.nc], axis=1)
        
        pred_output = tf.transpose(pred_output, [0, 2, 1])  # [batch, anchors, 4*dfl_ch]
        pred_scores = tf.transpose(pred_scores, [0, 2, 1])  # [batch, anchors, nc]
        
        # Get image size
        size = tf.cast(tf.shape(x[0])[1:3], pred_scores.dtype) * self.stride[0]
        
        # Generate anchors
        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)
        
        # Process targets
        if tf.shape(targets)[0] == 0:
            gt = tf.zeros([tf.shape(pred_scores)[0], 0, 5], dtype=pred_scores.dtype)
        else:
            # Group targets by image index
            i = targets[:, 0]  # image index
            unique_i, _, counts = tf.unique_with_counts(tf.cast(i, tf.int32))
            max_count = tf.reduce_max(counts)
            
            gt = tf.zeros([tf.shape(pred_scores)[0], max_count, 5], dtype=pred_scores.dtype)
            
            for j in range(tf.shape(pred_scores)[0]):
                matches = tf.equal(i, j)
                n = tf.reduce_sum(tf.cast(matches, tf.int32))
                if n > 0:
                    matched_targets = tf.boolean_mask(targets[:, 1:], matches)
                    # Pad or truncate to max_count
                    if n > max_count:
                        matched_targets = matched_targets[:max_count]
                    elif n < max_count:
                        padding = tf.zeros([max_count - n, 5], dtype=matched_targets.dtype)
                        matched_targets = tf.concat([matched_targets, padding], axis=0)
                    
                    gt = tf.tensor_scatter_nd_update(gt, [[j]], [matched_targets])
            
            # Convert boxes from normalized xywh to absolute xyxy
            size_tensor = tf.stack([size[1], size[0], size[1], size[0]])
            gt = tf.concat([
                gt[..., :1],  # class
                wh2xy(gt[..., 1:5] * size_tensor)  # boxes
            ], axis=-1)
        
        gt_labels, gt_bboxes = tf.split(gt, [1, 4], axis=2)
        mask_gt = tf.reduce_sum(gt_bboxes, axis=2, keepdims=True) > 0
        
        # Process box predictions using DFL
        b, a, c = tf.shape(pred_output)[0], tf.shape(pred_output)[1], tf.shape(pred_output)[2]
        pred_output_reshaped = tf.reshape(pred_output, [b, a, 4, c // 4])
        pred_bboxes = tf.nn.softmax(pred_output_reshaped, axis=-1)
        pred_bboxes = tf.reduce_sum(pred_bboxes * self.project, axis=-1)
        
        # Convert to absolute coordinates
        anchor_points_exp = tf.expand_dims(anchor_points, 0)  # [1, anchors, 2]
        pred_lt, pred_rb = tf.split(pred_bboxes, 2, axis=-1)
        pred_bboxes = tf.concat([
            anchor_points_exp - pred_lt,
            anchor_points_exp + pred_rb
        ], axis=-1)
        
        # Compute loss
        target_scores_sum = tf.maximum(tf.reduce_sum(tf.cast(mask_gt, tf.float32)), 1.0)
        
        # Classification loss
        target_scores = tf.zeros_like(pred_scores)
        if tf.reduce_any(mask_gt):
            # Simple assignment for now (can be improved with proper assignment strategy)
            target_scores = tf.nn.one_hot(tf.cast(gt_labels[..., 0], tf.int32), self.nc)
            target_scores = tf.where(mask_gt, target_scores, 0.0)
        
        loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_scores, logits=pred_scores
        )
        loss_cls = tf.reduce_sum(loss_cls) / target_scores_sum
        
        # Box and DFL loss (simplified)
        loss_box = tf.constant(0.0)
        loss_dfl = tf.constant(0.0)
        
        # Apply loss weights
        loss_cls *= self.params.get('cls', 0.5)
        loss_box *= self.params.get('box', 7.5)
        loss_dfl *= self.params.get('dfl', 1.5)
        
        return loss_cls + loss_box + loss_dfl


# Data augmentation utilities
def augment_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    HSV color space augmentation
    
    Args:
        image: Input image tensor
        h_gain: Hue gain
        s_gain: Saturation gain
        v_gain: Value gain
    
    Returns:
        Augmented image
    """
    # Random gains
    gains = tf.random.uniform([3], -1, 1) * [h_gain, s_gain, v_gain] + 1
    
    # Convert to HSV
    image_hsv = tf.image.rgb_to_hsv(image)
    
    # Split channels
    h, s, v = tf.split(image_hsv, 3, axis=-1)
    
    # Apply gains
    h = tf.math.floormod(h * gains[0], 1.0)
    s = tf.clip_by_value(s * gains[1], 0, 1)
    v = tf.clip_by_value(v * gains[2], 0, 1)
    
    # Merge and convert back
    image_hsv = tf.concat([h, s, v], axis=-1)
    image = tf.image.hsv_to_rgb(image_hsv)
    
    return image


def random_flip(image, boxes, prob_ud=0.0, prob_lr=0.5):
    """
    Random flip augmentation
    
    Args:
        image: Input image
        boxes: Bounding boxes [x, y, w, h] normalized
        prob_ud: Probability of up-down flip
        prob_lr: Probability of left-right flip
    
    Returns:
        Flipped image and boxes
    """
    # Up-down flip
    if tf.random.uniform([]) < prob_ud:
        image = tf.image.flip_up_down(image)
        if boxes is not None and tf.size(boxes) > 0:
            boxes = tf.concat([
                boxes[..., 0:1],
                1 - boxes[..., 1:2],
                boxes[..., 2:4]
            ], axis=-1)
    
    # Left-right flip
    if tf.random.uniform([]) < prob_lr:
        image = tf.image.flip_left_right(image)
        if boxes is not None and tf.size(boxes) > 0:
            boxes = tf.concat([
                1 - boxes[..., 0:1],
                boxes[..., 1:2],
                boxes[..., 2:4]
            ], axis=-1)
    
    return image, boxes


def learning_rate_schedule(epoch, epochs, lr0=0.01, lrf=0.01):
    """
    Linear learning rate schedule
    
    Args:
        epoch: Current epoch
        epochs: Total epochs
        lr0: Initial learning rate
        lrf: Final learning rate factor
    
    Returns:
        Learning rate for current epoch
    """
    return lr0 * ((1 - epoch / epochs) * (1.0 - lrf) + lrf)


def warmup_schedule(step, warmup_steps, lr0=0.01, warmup_bias_lr=0.1, warmup_momentum=0.8, momentum=0.937):
    """
    Warmup schedule for learning rate and momentum
    
    Args:
        step: Current step
        warmup_steps: Total warmup steps
        lr0: Initial learning rate
        warmup_bias_lr: Warmup bias learning rate
        warmup_momentum: Warmup momentum
        momentum: Target momentum
    
    Returns:
        Dictionary with lr and momentum values
    """
    if step <= warmup_steps:
        xi = [0, warmup_steps]
        
        # Learning rate warmup
        lr = np.interp(step, xi, [warmup_bias_lr, lr0])
        
        # Momentum warmup
        mom = np.interp(step, xi, [warmup_momentum, momentum])
        
        return {'lr': lr, 'momentum': mom}
    
    return None


# Export all utilities
__all__ = [
    'setup_seed',
    'setup_multi_processes',
    'make_anchors',
    'scale_boxes',
    'wh2xy',
    'xy2wh',
    'box_iou',
    'non_max_suppression',
    'smooth',
    'compute_ap',
    'clip_gradients',
    'AverageMeter',
    'EMA',
    'ComputeLoss',
    'augment_hsv',
    'random_flip',
    'learning_rate_schedule',
    'warmup_schedule'
]