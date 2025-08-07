import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Optional, Union

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util_tf import make_anchors, preprocess_image


def autopad(k, p=None, d=1):
    """Calculate padding for 'same' convolution"""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(keras.layers.Layer):
    """Basic Conv2d + BatchNorm + ReLU block"""
    
    def __init__(self, out_ch, k=1, s=1, p=None, d=1, g=1, act=True, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.kernel_size = k if isinstance(k, int) else k[0]
        self.stride = s if isinstance(s, int) else s[0]
        self.dilation_rate = d
        self.groups = g
        self.act = act
        self.padding = autopad(k, p, d)
        
    def build(self, input_shape):
        in_ch = input_shape[-1]
        
        # Determine padding mode
        if self.stride == 1:
            padding = 'same'
        else:
            padding = 'valid'
            
        # Conv2D layer
        self.conv_layer = layers.Conv2D(
            filters=self.out_ch,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=padding,
            dilation_rate=self.dilation_rate,
            groups=self.groups if self.groups > 1 else 1,
            use_bias=False,
            name=f'conv_{self.kernel_size}x{self.kernel_size}'
        )
        
        # BatchNorm
        self.batch_norm = layers.BatchNormalization(
            momentum=0.97,  # TF uses 1-momentum compared to PyTorch
            epsilon=0.001,
            name='bn'
        )
        
        # Activation
        if self.act:
            self.activation = layers.ReLU(name='relu')
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        # Manual padding for stride > 1
        if self.stride > 1:
            # Calculate padding
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            
            # Apply padding
            x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], 
                      mode='CONSTANT', constant_values=0)
        
        x = self.conv_layer(x)
        x = self.batch_norm(x, training=training)
        
        if self.act:
            x = self.activation(x)
            
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'out_ch': self.out_ch,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'act': self.act,
            'padding': self.padding
        })
        return config


class Residual(keras.layers.Layer):
    """Residual block with two convolutions"""
    
    def __init__(self, ch, add=True, **kwargs):
        super().__init__(**kwargs)
        self.use_residual = add
        self.ch = ch
        
    def build(self, input_shape):
        self.conv1 = Conv(self.ch, k=3, s=1)
        self.conv2 = Conv(self.ch, k=3, s=1)
        super().build(input_shape)
        
    def call(self, x, training=None):
        residual = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        
        if self.use_residual:
            x = layers.Add()([x, residual])
        return x


class CSP(keras.layers.Layer):
    """Cross Stage Partial Network block"""
    
    def __init__(self, out_ch, n=1, add=True, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.n = n
        self.add = add
        
    def build(self, input_shape):
        # Hidden channels (half of output)
        hidden_ch = self.out_ch // 2
        
        self.input_conv = Conv(hidden_ch, k=1, s=1)
        self.bottleneck_conv = Conv(hidden_ch, k=1, s=1)
        
        # Calculate concatenated channels
        concat_ch = (2 + self.n) * hidden_ch
        self.fusion_conv = Conv(self.out_ch, k=1, s=1)
        
        self.residual_blocks = [
            Residual(hidden_ch, self.add) for _ in range(self.n)
        ]
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        # Split pathway
        x1 = self.input_conv(x, training=training)
        x2 = self.bottleneck_conv(x, training=training)
        
        # Process through residual blocks and collect features
        features = [x1, x2]
        for block in self.residual_blocks:
            x2 = block(x2, training=training)
            features.append(x2)
            
        # Concatenate all features
        x = tf.concat(features, axis=-1)
        
        # Final convolution
        x = self.fusion_conv(x, training=training)
        return x


class SPP(keras.layers.Layer):
    """Spatial Pyramid Pooling block"""
    
    def __init__(self, out_ch, k=5, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.kernel_sizes = [k, k, k]  # Multiple kernel sizes for pooling
        
    def build(self, input_shape):
        in_ch = input_shape[-1]
        
        self.input_conv = Conv(in_ch // 2, k=1, s=1)
        self.fusion_conv = Conv(self.out_ch, k=1, s=1)
        
        # Create multiple MaxPooling layers
        self.pools = []
        for ks in self.kernel_sizes:
            pool = layers.MaxPooling2D(
                pool_size=ks,
                strides=1,
                padding='same'
            )
            self.pools.append(pool)
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        x = self.input_conv(x, training=training)
        
        # Apply pyramid pooling
        features = [x]
        for i, pool in enumerate(self.pools):
            if i == 0:
                pooled = pool(x)
            else:
                pooled = pool(features[-1])
            features.append(pooled)
        
        # Concatenate all pooled features
        x = tf.concat(features, axis=-1)
        x = self.fusion_conv(x, training=training)
        return x


class DarkNet(keras.layers.Layer):
    """Backbone network"""
    
    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        
    def build(self, input_shape):
        # Build each stage
        self.stage_p1 = keras.Sequential([
            Conv(self.width[1], k=3, s=2)
        ], name='stage_p1')
        
        self.stage_p2 = keras.Sequential([
            Conv(self.width[2], k=3, s=2),
            CSP(self.width[2], n=self.depth[0])
        ], name='stage_p2')
        
        self.stage_p3 = keras.Sequential([
            Conv(self.width[3], k=3, s=2),
            CSP(self.width[3], n=self.depth[1])
        ], name='stage_p3')
        
        self.stage_p4 = keras.Sequential([
            Conv(self.width[4], k=3, s=2),
            CSP(self.width[4], n=self.depth[2])
        ], name='stage_p4')
        
        self.stage_p5 = keras.Sequential([
            Conv(self.width[5], k=3, s=2),
            CSP(self.width[5], n=self.depth[0]),
            SPP(self.width[5], k=5)
        ], name='stage_p5')
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        p1_out = self.stage_p1(x, training=training)
        p2_out = self.stage_p2(p1_out, training=training)
        p3_out = self.stage_p3(p2_out, training=training)
        p4_out = self.stage_p4(p3_out, training=training)
        p5_out = self.stage_p5(p4_out, training=training)
        
        return p3_out, p4_out, p5_out


class DarkFPN(keras.layers.Layer):
    """Feature Pyramid Network"""
    
    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        
    def build(self, input_shape):
        # Upsampling layer
        self.upsample = layers.UpSampling2D(size=2, interpolation='nearest')
        
        # Top-down pathway - fusion blocks
        self.td_conv1 = Conv(self.width[4], k=1, s=1)  # Reduce channels before CSP
        self.td_block1 = CSP(self.width[4], n=self.depth[0], add=False)
        
        self.td_conv2 = Conv(self.width[3], k=1, s=1)  # Reduce channels before CSP
        self.td_block2 = CSP(self.width[3], n=self.depth[0], add=False)
        
        # Bottom-up pathway
        self.bu_conv1 = Conv(self.width[3], k=3, s=2)
        self.bu_block1 = CSP(self.width[4], n=self.depth[0], add=False)
        
        self.bu_conv2 = Conv(self.width[4], k=3, s=2)
        self.bu_block2 = CSP(self.width[5], n=self.depth[0], add=False)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        p3_in, p4_in, p5_in = inputs
        
        # Top-down pathway
        # P5 -> P4
        p5_upsampled = self.upsample(p5_in)
        fpn_out1 = tf.concat([p5_upsampled, p4_in], axis=-1)
        fpn_out1 = self.td_block1(fpn_out1, training=training)
        
        # P4 -> P3
        fpn_out1_upsampled = self.upsample(fpn_out1)
        fpn_out2 = tf.concat([fpn_out1_upsampled, p3_in], axis=-1)
        fpn_out2 = self.td_block2(fpn_out2, training=training)
        
        # Bottom-up pathway
        # P3 -> P4
        fpn_out2_downsampled = self.bu_conv1(fpn_out2, training=training)
        fpn_out3 = tf.concat([fpn_out2_downsampled, fpn_out1], axis=-1)
        fpn_out3 = self.bu_block1(fpn_out3, training=training)
        
        # P4 -> P5
        fpn_out3_downsampled = self.bu_conv2(fpn_out3, training=training)
        fpn_out4 = tf.concat([fpn_out3_downsampled, p5_in], axis=-1)
        fpn_out4 = self.bu_block2(fpn_out4, training=training)
        
        return fpn_out2, fpn_out3, fpn_out4


class DFL(keras.layers.Layer):
    """Distribution Focal Loss module"""
    
    def __init__(self, ch=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = ch
        
    def build(self, input_shape):
        # Initialize weights for integral
        self.register_buffer = tf.range(0, self.channels, dtype=tf.float32)
        super().build(input_shape)
        
    def call(self, x, training=None):
        b, h, w, c = x.shape
        
        # Reshape to separate box coordinates and channels
        x = tf.reshape(x, [b, h, w, 4, self.channels])
        
        # Apply softmax along the channel dimension
        x = tf.nn.softmax(x, axis=-1)
        
        # Weighted sum (integral)
        weights = tf.reshape(self.register_buffer, [1, 1, 1, 1, self.channels])
        x = tf.reduce_sum(x * weights, axis=-1)
        
        # Reshape back
        x = tf.reshape(x, [b, h, w, 4])
        
        return x


class Head(keras.layers.Layer):
    """Detection head"""
    
    def __init__(self, nc=80, filters=(), training_mode=True, **kwargs):
        super().__init__(**kwargs)
        self.dfl_channels = 16
        self.num_classes = nc
        self.num_layers = len(filters)
        self.filters = filters
        self.num_outputs = nc + self.dfl_channels * 4
        self.training_mode = training_mode
        
    def build(self, input_shape):
        c1 = max(self.filters[0], self.num_classes)
        c2 = max(self.filters[0] // 4, self.dfl_channels * 4)
        
        self.dfl = DFL(self.dfl_channels)
        
        # Classification heads
        self.cls_heads = []
        for i, filter_size in enumerate(self.filters):
            cls_head = keras.Sequential([
                Conv(c1, k=3, s=1),
                Conv(c1, k=3, s=1),
                layers.Conv2D(self.num_classes, kernel_size=1, use_bias=True)
            ], name=f'cls_head_{i}')
            self.cls_heads.append(cls_head)
        
        # Box regression heads
        self.box_heads = []
        for i, filter_size in enumerate(self.filters):
            box_head = keras.Sequential([
                Conv(c2, k=3, s=1),
                Conv(c2, k=3, s=1),
                layers.Conv2D(4 * self.dfl_channels, kernel_size=1, use_bias=True)
            ], name=f'box_head_{i}')
            self.box_heads.append(box_head)
        
        super().build(input_shape)
            
    def call(self, x, training=None):
        if not isinstance(x, (list, tuple)):
            x = [x]
            
        outputs = []
        for i, (box_head, cls_head) in enumerate(zip(self.box_heads, self.cls_heads)):
            box_output = box_head(x[i], training=training)
            cls_output = cls_head(x[i], training=training)
            output = tf.concat([box_output, cls_output], axis=-1)
            outputs.append(output)
            
        if self.training_mode or training:
            return outputs
            
        # Inference mode processing
        processed_outputs = []
        for output in outputs:
            b, h, w, c = output.shape
            # Reshape to [batch, num_anchors, num_outputs]
            output = tf.reshape(output, [b, h * w, self.num_outputs])
            processed_outputs.append(output)
            
        # Concatenate all outputs
        x_cat = tf.concat(processed_outputs, axis=1)
        
        # Split into box and class predictions
        box_preds = x_cat[..., :self.dfl_channels * 4]
        cls_preds = x_cat[..., self.dfl_channels * 4:]
        
        # Apply sigmoid to class predictions
        cls_preds = tf.nn.sigmoid(cls_preds)
        
        # Combine predictions
        output = tf.concat([box_preds, cls_preds], axis=-1)
        
        return output


class YOLOv8(keras.Model):
    """Complete YOLOv8 model in TensorFlow"""
    
    def __init__(self, width, depth, num_classes, input_size=320, training_mode=True, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.input_size = input_size
        self.training_mode = training_mode
        
        # Build model components
        self.backbone = DarkNet(width, depth)
        self.neck = DarkFPN(width, depth)
        self.detection_head = Head(
            num_classes, 
            (width[3], width[4], width[5]),
            training_mode=training_mode
        )
        
        # Initialize stride
        self.stride = tf.constant([8., 16., 32.])
        
    def call(self, x, training=None):
        # Backbone
        backbone_features = self.backbone(x, training=training)
        
        # Neck (FPN)
        neck_features = self.neck(backbone_features, training=training)
        
        # Detection head
        outputs = self.detection_head(list(neck_features), training=training)
        
        return outputs
    
    def build_graph(self, input_shape):
        """Build the model graph for summary"""
        x = keras.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


# Model factory functions
def yolo_v8_tiny(num_classes: int = 1, input_size: int = 320, training_mode: bool = True):
    """YOLOv8 Tiny variant"""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 128]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)


def yolo_v8_n(num_classes: int = 80, input_size: int = 640, training_mode: bool = True):
    """YOLOv8 Nano variant"""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)


def yolo_v8_s(num_classes: int = 80, input_size: int = 640, training_mode: bool = True):
    """YOLOv8 Small variant"""
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)


def yolo_v8_m(num_classes: int = 80, input_size: int = 640, training_mode: bool = True):
    """YOLOv8 Medium variant"""
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)


def yolo_v8_l(num_classes: int = 80, input_size: int = 640, training_mode: bool = True):
    """YOLOv8 Large variant"""
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)


def yolo_v8_x(num_classes: int = 80, input_size: int = 640, training_mode: bool = True):
    """YOLOv8 Extra Large variant"""
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLOv8(width, depth, num_classes, input_size, training_mode)





# Example usage and testing
if __name__ == "__main__":
    # Import utilities for testing
    from utils.util_tf import make_anchors, preprocess_image
    
    # Test model creation
    print("Creating YOLOv8 Tiny model...")
    model = yolo_v8_tiny(num_classes=1, input_size=320, training_mode=True)
    
    # Create dummy input
    dummy_input = tf.random.normal((1, 320, 320, 3))
    
    # Forward pass
    print("Testing forward pass...")
    outputs = model(dummy_input, training=True)
    
    print(f"\nNumber of output layers: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # Calculate parameters
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test different input sizes
    print("\nTesting different input sizes:")
    for size in [256, 320, 416, 512]:
        test_input = tf.random.normal((1, size, size, 3))
        test_outputs = model(test_input, training=True)
        print(f"Input size {size}x{size}:")
        for i, out in enumerate(test_outputs):
            print(f"  Output {i}: {out.shape}")
    
    print("\nModel created successfully!")