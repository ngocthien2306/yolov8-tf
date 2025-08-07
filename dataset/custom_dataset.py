import os
import random
import math
import glob
from typing import Tuple, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


class YOLOv8Dataset:
    """TensorFlow Dataset for YOLOv8 training and validation"""
    
    def __init__(self, 
                 filenames: List[str], 
                 input_size: int = 320, 
                 params: dict = None, 
                 augment: bool = True,
                 batch_size: int = 32):
        self.params = params or self._get_default_params()
        self.input_size = input_size
        self.augment = augment
        self.mosaic = augment
        self.batch_size = batch_size
        
        # Load and cache labels
        print("Loading dataset labels...")
        cache = self._load_labels(filenames)
        if not cache:
            raise ValueError("No valid images found in the dataset")
            
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float32)
        self.filenames = list(cache.keys())
        self.n = len(shapes)
        
        print(f"Loaded {self.n} images with labels")
    
    def _get_default_params(self):
        """Default augmentation parameters"""
        return {
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
            'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
            'shear': 0.0, 'flip_ud': 0.0, 'flip_lr': 0.5,
            'mosaic': 1.0, 'mix_up': 0.0
        }
    
    def create_dataset(self, shuffle: bool = True):
        """Create TensorFlow dataset"""
        # Create dataset from indices
        indices = list(range(self.n))
        if shuffle:
            random.shuffle(indices)
        
        dataset = tf.data.Dataset.from_tensor_slices(indices)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, self.n))
        
        # Map the loading function
        dataset = dataset.map(
            lambda idx: tf.py_function(
                func=self._load_sample,
                inp=[idx],
                Tout=[tf.float32, tf.float32, tf.int32]  # image, targets, shapes
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch the dataset
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_sample(self, index):
        """Load a single sample (called by tf.py_function)"""
        index = int(index.numpy())
        
        if self.mosaic and random.random() < self.params['mosaic']:
            # Load mosaic
            image, labels = self._load_mosaic(index)
            shapes = np.array([0, 0], dtype=np.int32)  # Placeholder for mosaic
        else:
            # Load single image
            image, shape = self._load_image(index)
            h, w = image.shape[:2]
            
            # Resize image
            image, ratio, pad = self._resize_image(image, self.input_size, self.augment)
            shapes = np.array([h, w], dtype=np.int32)
            
            # Process labels
            labels = self.labels[index].copy() if len(self.labels[index]) > 0 else np.zeros((0, 5))
            if labels.size > 0:
                labels[:, 1:] = self._wh2xy(labels[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            
            if self.augment:
                image, labels = self._random_perspective(image, labels)
        
        # Convert labels to proper format
        nl = len(labels)
        if nl > 0:
            labels[:, 1:5] = self._xy2wh(labels[:, 1:5], image.shape[1], image.shape[0])
        
        # Apply additional augmentations
        if self.augment:
            image = self._augment_hsv(image)
            
            # Flip augmentations
            if random.random() < self.params['flip_ud']:
                image = np.flipud(image)
                if nl > 0:
                    labels[:, 2] = 1 - labels[:, 2]
            
            if random.random() < self.params['flip_lr']:
                image = np.fliplr(image)
                if nl > 0:
                    labels[:, 1] = 1 - labels[:, 1]
        
        # Prepare targets tensor (pad to max possible detections)
        max_detections = 100  # Maximum number of detections per image
        targets = np.zeros((max_detections, 6), dtype=np.float32)
        if nl > 0:
            nl = min(nl, max_detections)  # Limit to max_detections
            targets[:nl, 1:] = labels[:nl]
        
        # Convert HWC to CHW and BGR to RGB
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image, dtype=np.float32)
        
        return image, targets, shapes
    
    def _load_image(self, index):
        """Load and resize a single image"""
        image = cv2.imread(self.filenames[index])
        if image is None:
            raise ValueError(f"Could not load image: {self.filenames[index]}")
        
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            if self.augment:
                # Random interpolation for augmentation
                interp = random.choice([cv2.INTER_AREA, cv2.INTER_CUBIC, 
                                      cv2.INTER_LINEAR, cv2.INTER_NEAREST])
            image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=interp)
        
        return image, (h, w)
    
    def _load_mosaic(self, index):
        """Load 4 images and create mosaic"""
        labels4 = []
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)
        
        # Center point
        xc = int(random.uniform(self.input_size * 0.5, self.input_size * 1.5))
        yc = int(random.uniform(self.input_size * 0.5, self.input_size * 1.5))
        
        indices = [index] + random.choices(range(self.n), k=3)
        random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            image, _ = self._load_image(idx)
            h, w, c = image.shape
            
            # Place image in mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.input_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.input_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.input_size * 2), min(self.input_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Process labels
            labels = self.labels[idx].copy() if len(self.labels[idx]) > 0 else np.zeros((0, 5))
            if labels.size > 0:
                labels[:, 1:] = self._wh2xy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)
        
        # Concatenate and clip labels
        labels4 = np.concatenate(labels4, 0)
        if len(labels4) > 0:
            labels4[:, 1:] = np.clip(labels4[:, 1:], 0, 2 * self.input_size)
        
        # Apply perspective transformation
        image4, labels4 = self._random_perspective(image4, labels4, border=(-self.input_size//2, -self.input_size//2))
        
        return image4, labels4
    
    def _resize_image(self, image, input_size, augment):
        """Resize and pad image"""
        shape = image.shape[:2]  # height, width
        
        # Scale ratio
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:
            r = min(r, 1.0)
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = input_size - new_unpad[0], input_size - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            if augment:
                interp = random.choice([cv2.INTER_AREA, cv2.INTER_CUBIC, 
                                      cv2.INTER_LINEAR, cv2.INTER_NEAREST])
            image = cv2.resize(image, new_unpad, interpolation=interp)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return image, (r, r), (dw, dh)
    
    def _augment_hsv(self, image):
        """HSV color space augmentation"""
        if random.random() < 0.5:
            h, s, v = self.params['hsv_h'], self.params['hsv_s'], self.params['hsv_v']
            r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
            
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype('uint8')
            lut_sat = np.clip(x * r[1], 0, 255).astype('uint8')
            lut_val = np.clip(x * r[2], 0, 255).astype('uint8')
            
            hue = cv2.LUT(hue, lut_hue)
            sat = cv2.LUT(sat, lut_sat)
            val = cv2.LUT(val, lut_val)
            
            image = cv2.cvtColor(cv2.merge((hue, sat, val)), cv2.COLOR_HSV2BGR)
        
        return image
    
    def _random_perspective(self, image, targets, border=(0, 0)):
        """Apply random perspective transformation"""
        height, width = image.shape[:2]
        
        # Create transformation matrices
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2
        C[1, 2] = -image.shape[0] / 2
        
        P = np.eye(3)  # perspective
        R = np.eye(3)  # rotation
        S = np.eye(3)  # shear
        T = np.eye(3)  # translation
        
        # Rotation and Scale
        angle = random.uniform(-self.params['degrees'], self.params['degrees'])
        scale = random.uniform(1 - self.params['scale'], 1 + self.params['scale'])
        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
        
        # Shear
        S[0, 1] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)
        
        # Translation
        T[0, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * width
        T[1, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * height
        
        # Combined transformation
        M = T @ S @ R @ P @ C
        
        if (border[0] != 0) or (border[1] != 0) or not np.array_equal(M, np.eye(3)):
            if border[0] != 0 or border[1] != 0:
                height += border[0] * 2
                width += border[1] * 2
            
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
        
        # Transform label coordinates
        n = len(targets)
        if n > 0:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = xy[:, :2].reshape(n, 8)
            
            # Create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            
            # Clip boxes
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
            
            # Filter valid boxes
            i = self._box_candidates(targets[:, 1:5].T * scale, new.T)
            targets = targets[i]
            targets[:, 1:5] = new[i]
        
        return image, targets
    
    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1):
        """Filter box candidates"""
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)
    
    @staticmethod
    def _wh2xy(x, w=640, h=640, padw=0, padh=0):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # x1
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # y1
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # x2
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # y2
        return y
    
    @staticmethod
    def _xy2wh(x, w=640, h=640):
        """Convert [x1, y1, x2, y2] to [x, y, w, h] normalized"""
        x = x.clip(0, [w-1e-3, h-1e-3, w-1e-3, h-1e-3])
        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
        y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
        y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
        return y
    
    def _load_labels(self, filenames):
        """Load labels from cache or create cache"""
        if not filenames:
            return {}
            
        cache_path = f'{os.path.dirname(filenames[0])}.cache'
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded cached labels from {cache_path}")
                return cache
            except:
                print("Cache file corrupted, regenerating...")
        
        # Create cache
        cache = {}
        print(f"Caching labels for {len(filenames)} images...")
        
        for filename in filenames:
            try:
                # Verify image
                with open(filename, 'rb') as f:
                    img = Image.open(f)
                    img.verify()
                shape = img.size  # (width, height)
                
                assert shape[0] > 9 and shape[1] > 9, f'Image size {shape} too small'
                assert img.format.lower() in FORMATS, f'Invalid format {img.format}'
                
                # Load labels
                label_file = filename.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                labels = np.zeros((0, 5), dtype=np.float32)
                
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        lines = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if lines:
                            labels = np.array(lines, dtype=np.float32)
                            
                    # Validate labels
                    if len(labels):
                        assert labels.shape[1] == 5, f'Labels require 5 columns, got {labels.shape[1]}'
                        assert (labels >= 0).all(), 'Negative label values found'
                        assert (labels[:, 1:] <= 1).all(), 'Non-normalized coordinates found'
                        
                        # Remove duplicates
                        labels = np.unique(labels, axis=0)
                
                cache[filename] = [labels, shape]
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # Save cache
        if cache:
            try:
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache, f)
                print(f"Cached {len(cache)} labels to {cache_path}")
            except Exception as e:
                print(f"Failed to save cache: {e}")
        
        return cache


# Example usage
def create_yolo_dataset(image_dir: str, 
                       input_size: int = 320, 
                       batch_size: int = 32,
                       augment: bool = True):
    """Helper function to create YOLOv8 dataset"""
    # Find all image files
    image_files = []
    for ext in FORMATS:
        pattern = os.path.join(image_dir, f"**/*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=True))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Create dataset
    dataset_loader = YOLOv8Dataset(
        filenames=image_files,
        input_size=input_size,
        augment=augment,
        batch_size=batch_size
    )
    
    return dataset_loader.create_dataset()


if __name__ == "__main__":
    # Example usage
    train_dir = "/path/to/train/images"  # Replace with your path
    val_dir = "/path/to/val/images"      # Replace with your path
    
    try:
        # Create training dataset
        train_dataset = create_yolo_dataset(
            image_dir=train_dir,
            input_size=320,
            batch_size=32,
            augment=True
        )
        
        # Create validation dataset
        val_dataset = create_yolo_dataset(
            image_dir=val_dir,
            input_size=320,
            batch_size=32,
            augment=False
        )
        
        print("Datasets created successfully!")
        
        # Test loading a batch
        for images, targets, shapes in train_dataset.take(1):
            print(f"Images shape: {images.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Shapes shape: {shapes.shape}")
            break
            
    except Exception as e:
        print(f"Error creating dataset: {e}")