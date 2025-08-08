import math
import random
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union


def setup_seed(seed=0):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def preprocess_image(image, input_size=320, augment=False):
    """
    Preprocess image for YOLOv8
    
    Args:
        image: Input image tensor
        input_size: Target size for the image
        augment: Whether to apply augmentation
    
    Returns:
        Preprocessed image tensor
    """
    # Resize image
    image = tf.image.resize(image, [input_size, input_size])
    
    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image


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
    num_anchors = tf.shape(prediction)[1]
    nc = tf.shape(prediction)[2] - 4  # number of classes
    
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


def scale_boxes(boxes, shape1, shape2, ratio_pad=None):
    """
    Scale bounding boxes from one image size to another
    
    Args:
        boxes: Bounding boxes to scale
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
    boxes = tf.concat([
        (boxes[..., 0:1] - pad[0]) / gain,
        (boxes[..., 1:2] - pad[1]) / gain,
        (boxes[..., 2:3] - pad[0]) / gain,
        (boxes[..., 3:4] - pad[1]) / gain
    ], axis=-1)
    
    # Clip boxes
    boxes = tf.concat([
        tf.clip_by_value(boxes[..., 0:1], 0, shape2[1]),
        tf.clip_by_value(boxes[..., 1:2], 0, shape2[0]),
        tf.clip_by_value(boxes[..., 2:3], 0, shape2[1]),
        tf.clip_by_value(boxes[..., 3:4], 0, shape2[0])
    ], axis=-1)
    
    return boxes


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
    
    def restore(self, model=None):
        """Restore original weights"""
        if model is None:
            model = self.model
            
        # This would need to store original weights first
        pass


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


def mixup(image1, boxes1, image2, boxes2, alpha=32.0):
    """
    MixUp augmentation
    
    Args:
        image1: First image
        boxes1: First image boxes
        image2: Second image
        boxes2: Second image boxes
        alpha: Beta distribution parameter
    
    Returns:
        Mixed image and concatenated boxes
    """
    # Sample mixing ratio
    ratio = np.random.beta(alpha, alpha)
    
    # Mix images
    mixed_image = image1 * ratio + image2 * (1 - ratio)
    
    # Concatenate boxes
    mixed_boxes = tf.concat([boxes1, boxes2], axis=0)
    
    return mixed_image, mixed_boxes


# Export all utilities
__all__ = [
    'setup_seed',
    'make_anchors',
    'preprocess_image',
    'wh2xy',
    'xy2wh',
    'box_iou',
    'non_max_suppression',
    'scale_boxes',
    'AverageMeter',
    'EMA',
    'clip_gradients',
    'compute_ap',
    'smooth',
    'learning_rate_schedule',
    'warmup_schedule',
    'augment_hsv',
    'random_flip',
    'mixup'
]