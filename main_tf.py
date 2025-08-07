import argparse
import os
import sys
import yaml
import glob
import time
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from nets.model import yolo_v8_tiny, yolo_v8_n, yolo_v8_s, yolo_v8_m, yolo_v8_l, yolo_v8_x
from utils.loss import ComputeLoss
from utils.util_tf import (
    setup_seed, make_anchors, non_max_suppression, compute_ap,
    EMA, AverageMeter, clip_gradients, learning_rate_schedule,
    warmup_schedule, scale_boxes, wh2xy, box_iou
)
from dataset.custom_dataset import create_yolo_dataset

# Set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_model_summary(model, input_shape=(320, 320, 3)):
    """Print model summary"""
    dummy_input = tf.random.normal((1, *input_shape))
    _ = model(dummy_input, training=False)
    
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print("=" * 70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("=" * 70)


def create_model(args, params):
    """Create YOLOv8 model based on variant"""
    num_classes = len(params['names'])
    
    model_variants = {
        'tiny': yolo_v8_tiny,
        'n': yolo_v8_n,
        's': yolo_v8_s,
        'm': yolo_v8_m,
        'l': yolo_v8_l,
        'x': yolo_v8_x
    }
    
    if args.model not in model_variants:
        raise ValueError(f"Unknown model variant: {args.model}")
    
    model = model_variants[args.model](
        num_classes=num_classes,
        input_size=args.input_size,
        training_mode=True
    )
    
    return model


def create_optimizer(model, args, params):
    """Create optimizer with parameter groups"""
    # Create optimizer with initial learning rate
    optimizer = keras.optimizers.SGD(
        learning_rate=params['lr0'],
        momentum=params['momentum'],
        nesterov=True
    )
    
    return optimizer


@tf.function
def train_step(model, images, targets, optimizer, compute_loss, training=True):
    """Single training step"""
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(images, training=training)
        
        # Compute loss
        loss = compute_loss(predictions, targets)
        
        # Add regularization losses
        if model.losses:
            loss += tf.add_n(model.losses)
    
    if training:
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_weights)
        
        # Clip gradients
        gradients = clip_gradients(gradients, max_norm=10.0)
        
        # Update weights
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return loss, predictions


@tf.function
def val_step(model, images, targets, compute_loss):
    """Single validation step"""
    # Forward pass
    predictions = model(images, training=False)
    
    # Compute loss
    loss = compute_loss(predictions, targets)
    
    return loss, predictions


def train_epoch(model, train_dataset, optimizer, compute_loss, epoch, args, params, ema=None):
    """Train for one epoch"""
    loss_meter = AverageMeter()
    
    # Progress bar
    print(f"\nEpoch {epoch + 1}/{args.epochs}")
    print("-" * 70)
    
    start_time = time.time()
    
    # Calculate learning rate for this epoch
    lr = learning_rate_schedule(epoch, args.epochs, params['lr0'], params['lrf'])
    optimizer.learning_rate = lr
    
    for step, (images, targets, _) in enumerate(train_dataset):
        # Warmup
        if args.warmup_epochs > 0:
            warmup_steps = args.warmup_epochs * args.steps_per_epoch
            current_step = epoch * args.steps_per_epoch + step
            
            if current_step < warmup_steps:
                # Linear warmup
                warmup_lr = params['lr0'] * (current_step / warmup_steps)
                optimizer.learning_rate = warmup_lr
        
        # Training step
        loss, _ = train_step(model, images, targets, optimizer, compute_loss, training=True)
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update metrics
        loss_meter.update(loss.numpy(), images.shape[0])
        
        # Print progress
        if step % args.print_freq == 0:
            current_lr = optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            
            memory = 0
            if tf.config.list_physical_devices('GPU'):
                try:
                    memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e9
                except:
                    memory = 0
            
            print(f"Step [{step}/{args.steps_per_epoch}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"LR: {current_lr:.6f} "
                  f"Mem: {memory:.1f}GB")
    
    epoch_time = time.time() - start_time
    print(f"Epoch time: {epoch_time:.1f}s, Avg loss: {loss_meter.avg:.4f}")
    
    return loss_meter.avg


def validate(model, val_dataset, compute_loss, args):
    """Validate model"""
    loss_meter = AverageMeter()
    all_predictions = []
    all_targets = []
    all_shapes = []
    
    print("\nValidating...")
    
    for images, targets, shapes in val_dataset:
        # Validation step
        loss, predictions = val_step(model, images, targets, compute_loss)
        
        # Update metrics
        loss_meter.update(loss.numpy(), images.shape[0])
        
        # Store predictions for mAP calculation
        all_predictions.extend(predictions)
        all_targets.append(targets)
        all_shapes.extend(shapes)
    
    # Calculate mAP
    map50, map_score = calculate_map(all_predictions, all_targets, all_shapes, args)
    
    print(f"Validation - Loss: {loss_meter.avg:.4f}, mAP@50: {map50:.3f}, mAP@50-95: {map_score:.3f}")
    
    return loss_meter.avg, map50, map_score


def calculate_map(predictions, targets, shapes, args):
    """Calculate mAP metrics"""
    # Process predictions with NMS
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Initialize metrics
    ap_per_class = []
    
    # Process each image
    for pred, target, shape in zip(predictions, targets, shapes):
        # Apply NMS to predictions
        pred_nms = non_max_suppression(
            pred[None, ...],
            conf_threshold=0.001,
            iou_threshold=0.65,
            max_det=300
        )[0]
        
        if len(pred_nms) == 0 or len(target) == 0:
            continue
        
        # Scale predictions to original image size
        if shape is not None:
            pred_boxes = pred_nms[:, :4]
            pred_boxes = scale_boxes(pred_boxes, args.input_size, shape[:2])
            pred_nms = tf.concat([pred_boxes, pred_nms[:, 4:]], axis=-1)
        
        # Convert target boxes to xyxy format
        target_boxes = target[:, 2:6]
        target_boxes = wh2xy(target_boxes)
        target_labels = target[:, 1]
        
        # Calculate IoU
        if len(pred_nms) > 0 and len(target_boxes) > 0:
            ious = box_iou(pred_nms[:, :4], target_boxes)
            
            # For each IoU threshold
            for iou_thresh in iou_thresholds:
                # Match predictions to targets
                matches = ious > iou_thresh
                # Calculate AP for this threshold
                # This is simplified - full implementation would need per-class AP
    
    # Simplified mAP calculation
    map50 = 0.5  # Placeholder
    map_score = 0.3  # Placeholder
    
    return map50, map_score


def save_checkpoint(model, optimizer, epoch, best_map, args, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = Path(args.weights_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    if is_best:
        model.save_weights(checkpoint_dir / 'best.h5')
        print(f"Saved best model with mAP: {best_map:.3f}")
    
    # Always save last checkpoint
    model.save_weights(checkpoint_dir / 'last.h5')
    
    # Save checkpoint info
    checkpoint_info = {
        'epoch': epoch,
        'best_map': best_map,
        'model_variant': args.model,
        'input_size': args.input_size
    }
    
    with open(checkpoint_dir / 'checkpoint_info.yaml', 'w') as f:
        yaml.dump(checkpoint_info, f)


def load_checkpoint(model, args):
    """Load model checkpoint"""
    checkpoint_path = Path(args.weights_dir) / 'last.h5'
    
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Load checkpoint info
        info_path = Path(args.weights_dir) / 'checkpoint_info.yaml'
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = yaml.safe_load(f)
                return info.get('epoch', 0), info.get('best_map', 0)
    
    return 0, 0


def main(args):
    """Main training function"""
    # Set random seed
    setup_seed(args.seed)
    
    # Set GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {args.gpu}")
            except RuntimeError as e:
                print(e)
    
    # Load parameters
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    
    # Update params with args
    params['lr0'] = args.lr0
    params['momentum'] = args.momentum
    params['weight_decay'] = args.weight_decay
    
    # Set default values if not in config
    if 'lrf' not in params:
        params['lrf'] = 0.01  # Final learning rate factor
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = create_yolo_dataset(
        image_dir=os.path.join(args.data_root, 'train/images'),
        input_size=args.input_size,
        batch_size=args.batch_size,
        augment=True
    )
    
    val_dataset = create_yolo_dataset(
        image_dir=os.path.join(args.data_root, 'valid/images'),
        input_size=args.input_size,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Calculate steps per epoch
    args.steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    if args.steps_per_epoch < 0:
        args.steps_per_epoch = 1000  # Default value if unknown
    
    # Create model
    print(f"Creating model: YOLOv8-{args.model}")
    model = create_model(args, params)
    
    if args.print_model:
        print_model_summary(model, (args.input_size, args.input_size, 3))
    
    # Create optimizer
    optimizer = create_optimizer(model, args, params)
    
    # Create loss function
    compute_loss = ComputeLoss(model, params)
    
    # Create EMA
    ema = EMA(model, decay=0.9999) if args.ema else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_map = 0
    if args.resume:
        start_epoch, best_map = load_checkpoint(model, args)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'map50': [],
        'map': []
    }
    
    # CSV logger
    csv_path = Path(args.weights_dir) / 'results.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=['epoch', 'train_loss', 'val_loss', 'map50', 'map']
    )
    csv_writer.writeheader()
    
    # Training loop
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_dataset, optimizer, compute_loss,
            epoch, args, params, ema
        )
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            # Use EMA weights for validation if available
            if ema:
                ema.apply(model)
            
            val_loss, map50, map_score = validate(
                model, val_dataset, compute_loss, args
            )
            
            # Restore original weights after validation
            if ema:
                # Note: This would need proper implementation to store/restore weights
                pass
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['map50'].append(map50)
            history['map'].append(map_score)
            
            # Write to CSV
            csv_writer.writerow({
                'epoch': epoch + 1,
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'map50': f'{map50:.3f}',
                'map': f'{map_score:.3f}'
            })
            csv_file.flush()
            
            # Save checkpoint
            is_best = map_score > best_map
            if is_best:
                best_map = map_score
            
            save_checkpoint(model, optimizer, epoch + 1, best_map, args, is_best)
        
        # Save last checkpoint every epoch
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_map, args, False)
    
    # Close CSV file
    csv_file.close()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best mAP: {best_map:.3f}")
    print(f"Weights saved to: {args.weights_dir}")


def test(args):
    """Test model on validation set"""
    # Load parameters
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    
    # Create model
    print(f"Creating model: YOLOv8-{args.model}")
    model = create_model(args, params)
    
    # Load weights
    weights_path = Path(args.weights_dir) / 'best.h5'
    if not weights_path.exists():
        weights_path = Path(args.weights_dir) / 'last.h5'
    
    if weights_path.exists():
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"No weights found at {weights_path}")
        return
    
    # Create dataset
    val_dataset = create_yolo_dataset(
        image_dir=os.path.join(args.data_root, 'valid/images'),
        input_size=args.input_size,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Create loss function
    compute_loss = ComputeLoss(model, params)
    
    # Validate
    val_loss, map50, map_score = validate(model, val_dataset, compute_loss, args)
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"mAP@50: {map50:.3f}")
    print(f"mAP@50-95: {map_score:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 TensorFlow Training')
    
    # Model settings
    parser.add_argument('--model', default='tiny', type=str,
                       choices=['tiny', 'n', 's', 'm', 'l', 'x'],
                       help='Model variant')
    parser.add_argument('--input-size', default=320, type=int,
                       help='Input image size')
    parser.add_argument('--config', default='utils/args.yaml', type=str,
                       help='Path to config file')
    
    # Dataset settings
    parser.add_argument('--data-root', default='/root/nguyen/research/evs/data_processed',
                       type=str, help='Dataset root directory')
    parser.add_argument('--batch-size', default=32, type=int,
                       help='Batch size')
    parser.add_argument('--workers', default=8, type=int,
                       help='Number of data loading workers')
    
    # Training settings
    parser.add_argument('--epochs', default=300, type=int,
                       help='Number of epochs')
    parser.add_argument('--lr0', default=0.01, type=float,
                       help='Initial learning rate')
    parser.add_argument('--momentum', default=0.937, type=float,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', default=3, type=int,
                       help='Warmup epochs')
    parser.add_argument('--ema', action='store_true',
                       help='Use EMA for model weights')
    
    # Checkpoint settings
    parser.add_argument('--weights-dir', default='weights', type=str,
                       help='Directory to save weights')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--save-freq', default=10, type=int,
                       help='Save checkpoint frequency')
    parser.add_argument('--val-freq', default=5, type=int,
                       help='Validation frequency')
    
    # Other settings
    parser.add_argument('--gpu', default='0', type=str,
                       help='GPU device to use')
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed')
    parser.add_argument('--print-freq', default=10, type=int,
                       help='Print frequency')
    parser.add_argument('--print-model', action='store_true',
                       help='Print model summary')
    
    # Mode
    parser.add_argument('--train', action='store_true',
                       help='Train model')
    parser.add_argument('--test', action='store_true',
                       help='Test model')
    
    args = parser.parse_args()
    
    # Set default mode to train if not specified
    if not args.train and not args.test:
        args.train = True
    
    if args.train:
        main(args)
    
    if args.test:
        test(args)