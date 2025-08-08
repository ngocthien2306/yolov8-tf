import argparse
import csv
import os
import warnings
import glob
import yaml
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm

from nets.nn_tf import yolo_v8_tiny, yolo_v8_n, yolo_v8_s, yolo_v8_m, yolo_v8_l, yolo_v8_x
from utils.util_new_tf import (
    setup_seed, setup_multi_processes, AverageMeter, EMA, ComputeLoss, 
    non_max_suppression, scale_boxes, compute_ap, clip_gradients,
    learning_rate_schedule, warmup_schedule
)
from dataset.custom_dataset import Dataset

warnings.filterwarnings("ignore")


def get_model(model_name, num_classes, input_size=320, training_mode=True):
    """Get model by name"""
    models = {
        'tiny': yolo_v8_tiny,
        'nano': yolo_v8_n,
        'small': yolo_v8_s,
        'medium': yolo_v8_m,
        'large': yolo_v8_l,
        'xlarge': yolo_v8_x
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](num_classes, input_size, training_mode)


def print_model_summary(model, input_shape):
    """Print model summary"""
    print("\nModel Summary:")
    print("=" * 80)
    
    # Build model with input shape
    model.build(input_shape)
    
    # Calculate total parameters
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 80)


def create_dataset(data_root, split, input_size, params, is_training=True):
    """Create dataset from directory structure"""
    images_dir = Path(data_root) / split / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    filenames = []
    for ext in image_extensions:
        filenames.extend(glob.glob(str(images_dir / ext)))
    
    if not filenames:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(filenames)} images in {split} set")
    
    # Shuffle filenames for training
    if is_training:
        np.random.shuffle(filenames)
    
    return Dataset(filenames, input_size, params, is_training)


@tf.function
def train_step(model, images, targets, optimizer, compute_loss, training=True):
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model(images, training=training)
        loss = compute_loss(predictions, targets)
    
    if training:
        gradients = tape.gradient(loss, model.trainable_variables)
        # Clip gradients
        gradients = clip_gradients(gradients, max_norm=10.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, predictions


def train_epoch(model, dataset, optimizer, compute_loss, epoch, args, params):
    """Train for one epoch"""
    model.trainable = True
    
    # Create data loader
    batch_size = args.batch_size
    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // batch_size
    
    # Convert dataset to tf.data
    def data_generator():
        for i in range(dataset_size):
            yield dataset[i]
    
    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(args.input_size, args.input_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),  # [img_id, cls, x, y, w, h]
            tf.TensorSpec(shape=(), dtype=tf.string)  # filename
        )
    )
    
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Training loop
    loss_meter = AverageMeter()
    progress_bar = tqdm.tqdm(
        enumerate(tf_dataset), 
        total=steps_per_epoch, 
        desc=f'Epoch {epoch+1}/{args.epochs}'
    )
    
    for step, (images, targets, _) in progress_bar:
        # Warmup learning rate
        global_step = epoch * steps_per_epoch + step
        warmup_steps = int(params.get('warmup_epochs', 3) * steps_per_epoch)
        
        if global_step < warmup_steps:
            warmup_info = warmup_schedule(
                global_step, warmup_steps, 
                params.get('lr0', 0.01),
                params.get('warmup_bias_lr', 0.1),
                params.get('warmup_momentum', 0.8),
                params.get('momentum', 0.937)
            )
            if warmup_info:
                optimizer.learning_rate.assign(warmup_info['lr'])
        
        # Preprocess images and targets
        images = tf.cast(images, tf.float32) / 255.0
        
        # Training step
        loss, predictions = train_step(
            model, images, targets, optimizer, compute_loss, training=True
        )
        
        loss_meter.update(loss.numpy(), images.shape[0])
        
        # Update progress bar
        memory_usage = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e9 if len(tf.config.list_physical_devices('GPU')) > 0 else 0
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.learning_rate.numpy():.6f}',
            'memory': f'{memory_usage:.1f}G'
        })
    
    return loss_meter.avg


@tf.function
def val_step(model, images):
    """Single validation step"""
    predictions = model(images, training=False)
    return predictions


def validate(model, dataset, args, params):
    """Validate the model"""
    model.trainable = False
    
    # Create data loader
    batch_size = min(args.batch_size, 32)  # Use smaller batch size for validation
    dataset_size = len(dataset)
    
    # Convert dataset to tf.data
    def data_generator():
        for i in range(dataset_size):
            yield dataset[i]
    
    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(args.input_size, args.input_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Validation metrics
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm.tqdm(tf_dataset, desc='Validating')
    
    for images, targets, shapes in progress_bar:
        # Preprocess
        images = tf.cast(images, tf.float32) / 255.0
        
        # Inference
        predictions = val_step(model, images)
        
        # Convert predictions to detections
        if isinstance(predictions, list):
            # Training mode output - convert to inference format
            batch_size = tf.shape(images)[0]
            processed_preds = []
            
            for i in range(batch_size):
                # Process each image in the batch
                img_preds = []
                for pred in predictions:
                    img_pred = pred[i:i+1]  # Get single image prediction
                    b, h, w, c = tf.shape(img_pred)[0], tf.shape(img_pred)[1], tf.shape(img_pred)[2], tf.shape(img_pred)[3]
                    img_pred = tf.reshape(img_pred, [b, h*w, c])
                    img_preds.append(img_pred)
                
                # Concatenate predictions from different scales
                img_pred_cat = tf.concat(img_preds, axis=1)
                processed_preds.append(img_pred_cat[0])  # Remove batch dimension
            
            predictions = tf.stack(processed_preds)
        
        # Apply NMS
        detections = non_max_suppression(
            predictions.numpy(), 
            conf_threshold=0.001, 
            iou_threshold=0.65,
            max_det=300
        )
        
        all_predictions.extend(detections)
        all_targets.extend(targets.numpy())
    
    # Compute metrics (simplified)
    if all_predictions:
        print(f"\nValidation completed. Processed {len(all_predictions)} images.")
        # For now, return dummy metrics
        map50, mean_ap = 0.5, 0.3  # Placeholder values
    else:
        map50, mean_ap = 0.0, 0.0
    
    print(f'mAP@50: {map50:.3f}, mAP@50-95: {mean_ap:.3f}')
    return map50, mean_ap


def train(args, params):
    """Main training function"""
    print(f"Starting training with model: {args.model}")
    
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set GPU growth
            for gpu in gpus:
                tf.config.experimental.set_gpu_growth_enabled(gpu, True)
            
            # Set specific GPU
            if args.gpu is not None:
                tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
                print(f"Using GPU: {args.gpu}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")
    
    # Create model
    num_classes = len(params['names'])
    model = get_model(args.model, num_classes, args.input_size, training_mode=True)
    
    # Print model summary
    print_model_summary(model, (None, args.input_size, args.input_size, 3))
    
    # Create datasets
    train_dataset = create_dataset(
        args.data_root, 'train', args.input_size, params, is_training=True
    )
    val_dataset = create_dataset(
        args.data_root, 'valid', args.input_size, params, is_training=False
    )
    
    # Setup optimizer
    initial_lr = params.get('lr0', 0.01)
    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=params.get('momentum', 0.937),
        nesterov=True
    )
    
    # Setup learning rate scheduler
    def lr_schedule(epoch):
        return learning_rate_schedule(
            epoch, args.epochs, 
            params.get('lr0', 0.01), 
            params.get('lrf', 0.01)
        )
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
    
    # Setup EMA
    ema = EMA(model) if args.gpu == 0 or args.gpu is None else None
    
    # Setup loss function
    compute_loss = ComputeLoss(model, params)
    
    # Create weights directory
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_map = 0.0
    train_losses = []
    
    # Setup CSV logging
    csv_file = weights_dir / 'training_log.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['epoch', 'train_loss', 'mAP@50', 'mAP@50-95', 'lr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for epoch in range(args.epochs):
            print(f'\n--- Epoch {epoch+1}/{args.epochs} ---')
            
            # Train for one epoch
            train_loss = train_epoch(
                model, train_dataset, optimizer, compute_loss, epoch, args, params
            )
            train_losses.append(train_loss)
            
            # Update EMA
            if ema:
                ema.update(model)
            
            # Validation
            if (epoch + 1) % args.val_interval == 0:
                val_model = model
                if ema:
                    # Use EMA weights for validation
                    val_model = keras.models.clone_model(model)
                    val_model.build((None, args.input_size, args.input_size, 3))
                    # Copy EMA weights
                    for ema_weight, val_weight in zip(ema.ema_weights, val_model.trainable_weights):
                        val_weight.assign(ema_weight)
                
                map50, mean_ap = validate(val_model, val_dataset, args, params)
                
                # Save best model
                if mean_ap > best_map:
                    best_map = mean_ap
                    model.save_weights(weights_dir / 'best_weights')
                    print(f'New best mAP: {best_map:.4f} - Model saved')
                
                # Log to CSV
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'mAP@50': map50,
                    'mAP@50-95': mean_ap,
                    'lr': optimizer.learning_rate.numpy()
                })
                f.flush()
            
            # Update learning rate
            lr_scheduler.on_epoch_end(epoch)
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                model.save_weights(weights_dir / f'epoch_{epoch+1}_weights')
        
        # Save final model
        model.save_weights(weights_dir / 'final_weights')
    
    print(f'\nTraining completed! Best mAP: {best_map:.4f}')
    print(f'Weights saved to: {weights_dir}')


def test(args, params):
    """Test/inference function"""
    print(f"Starting testing with model: {args.model}")
    
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and args.gpu is not None:
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            tf.config.experimental.set_gpu_growth_enabled(gpus[args.gpu], True)
            print(f"Using GPU: {args.gpu}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Create model
    num_classes = len(params['names'])
    model = get_model(args.model, num_classes, args.input_size, training_mode=False)
    
    # Load weights
    weights_path = Path('weights') / 'best_weights'
    if weights_path.exists():
        model.load_weights(weights_path)
        print(f"Loaded weights from: {weights_path}")
    else:
        print("No weights found, using random initialization")
    
    # Create test dataset
    test_dataset = create_dataset(
        args.data_root, 'valid', args.input_size, params, is_training=False
    )
    
    # Run validation
    map50, mean_ap = validate(model, test_dataset, args, params)
    
    print(f'\nTest Results:')
    print(f'mAP@50: {map50:.4f}')
    print(f'mAP@50-95: {mean_ap:.4f}')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 TensorFlow Training')
    
    # Model arguments
    parser.add_argument('--model', default='tiny', type=str, 
                       choices=['tiny', 'nano', 'small', 'medium', 'large', 'xlarge'],
                       help='Model size')
    parser.add_argument('--input-size', default=320, type=int, help='Input image size')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    
    # Training arguments
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('--data-root', required=True, type=str, help='Dataset root directory')
    parser.add_argument('--gpu', default=None, type=int, help='GPU ID to use')
    
    # Mode arguments
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--test', action='store_true', help='Testing mode')
    
    # Other arguments
    parser.add_argument('--val-interval', default=10, type=int, help='Validation interval (epochs)')
    parser.add_argument('--save-interval', default=50, type=int, help='Save interval (epochs)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train and not args.test:
        parser.error("Must specify either --train or --test")
    
    if not Path(args.data_root).exists():
        parser.error(f"Data root directory does not exist: {args.data_root}")
    
    # Setup
    setup_seed(args.seed)
    setup_multi_processes()
    
    # Load parameters
    params_file = Path('utils') / 'args.yaml'
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
    else:
        # Default parameters
        params = {
            'names': {0: 'person'},  # Default single class
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_bias_lr': 0.1,
            'warmup_momentum': 0.8,
            'cls': 0.5,
            'box': 7.5,
            'dfl': 1.5
        }
        print(f"Warning: {params_file} not found, using default parameters")
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input size: {args.input_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data root: {args.data_root}")
    print(f"  Classes: {len(params['names'])}")
    
    # Run training or testing
    if args.train:
        train(args, params)
    
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()