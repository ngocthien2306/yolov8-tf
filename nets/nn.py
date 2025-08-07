import math

import torch

from utils.util import make_anchors
from torch import Tensor
from typing import List, Tuple, Union


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv

class Conv(torch.nn.Module):
    """Basic Conv2d + BatchNorm + ReLU block"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.batch_norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv_layer(x)))

    def fuse_forward(self, x):
        return self.activation(self.conv_layer(x))


class Residual(torch.nn.Module):
    """Residual block with two convolutions"""
    def __init__(self, ch, add=True):
        super().__init__()
        self.use_residual = add
        self.res_block = torch.nn.Sequential(
            Conv(ch, ch, 3),  # first_conv_3x3
            Conv(ch, ch, 3)   # second_conv_3x3
        )

    def forward(self, x):
        return self.res_block(x) + x if self.use_residual else self.res_block(x)


class CSP(torch.nn.Module):
    """Cross Stage Partial Network block"""
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.input_conv = Conv(in_ch, out_ch // 2)
        self.bottleneck_conv = Conv(in_ch, out_ch // 2)
        self.fusion_conv = Conv((2 + n) * out_ch // 2, out_ch)
        self.residual_blocks = torch.nn.ModuleList(
            Residual(out_ch // 2, add) for _ in range(n)
        )

    def forward(self, x):
        features = [self.input_conv(x), self.bottleneck_conv(x)]
        features.extend(block(features[-1]) for block in self.residual_blocks)
        return self.fusion_conv(torch.cat(features, dim=1))


class SPP(torch.nn.Module):
    """Spatial Pyramid Pooling block"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.input_conv = Conv(in_ch, in_ch // 2)
        self.fusion_conv = Conv(in_ch * 2, out_ch)
        self.maxpool = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.input_conv(x)
        pool1 = self.maxpool(x)
        pool2 = self.maxpool(pool1)
        pool3 = self.maxpool(pool2)
        return self.fusion_conv(torch.cat([x, pool1, pool2, pool3], 1))


class DarkNet(torch.nn.Module):
    """Backbone network"""
    def __init__(self, width, depth):
        super().__init__()
        # P1: Initial stage
        stage_p1 = [Conv(width[0], width[1], 3, 2)]

        # P2: First feature stage
        stage_p2 = [
            Conv(width[1], width[2], 3, 2),
            CSP(width[2], width[2], depth[0])
        ]

        # P3: Second feature stage
        stage_p3 = [
            Conv(width[2], width[3], 3, 2),
            CSP(width[3], width[3], depth[1])
        ]

        # P4: Third feature stage
        stage_p4 = [
            Conv(width[3], width[4], 3, 2),
            CSP(width[4], width[4], depth[2])
        ]

        # P5: Final feature stage
        stage_p5 = [
            Conv(width[4], width[5], 3, 2),
            CSP(width[5], width[5], depth[0]),
            SPP(width[5], width[5])
        ]

        self.stage_p1 = torch.nn.Sequential(*stage_p1)
        self.stage_p2 = torch.nn.Sequential(*stage_p2)
        self.stage_p3 = torch.nn.Sequential(*stage_p3)
        self.stage_p4 = torch.nn.Sequential(*stage_p4)
        self.stage_p5 = torch.nn.Sequential(*stage_p5)

    def forward(self, x):
        p1_out = self.stage_p1(x)
        p2_out = self.stage_p2(p1_out)
        p3_out = self.stage_p3(p2_out)
        p4_out = self.stage_p4(p3_out)
        p5_out = self.stage_p5(p4_out)
        return p3_out, p4_out, p5_out


class DarkFPN(torch.nn.Module):
    """Feature Pyramid Network"""
    def __init__(self, width, depth):
        super().__init__()
        self.upsample = torch.nn.Upsample(None, 2)
        
        # Top-down pathway
        self.td_block1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.td_block2 = CSP(width[3] + width[4], width[3], depth[0], False)
        
        # Bottom-up pathway
        self.bu_conv1 = Conv(width[3], width[3], 3, 2)
        self.bu_block1 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.bu_conv2 = Conv(width[4], width[4], 3, 2)
        self.bu_block2 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3_in, p4_in, p5_in = x
        
        # Top-down pathway
        fpn_out1 = self.td_block1(torch.cat([self.upsample(p5_in), p4_in], 1))
        fpn_out2 = self.td_block2(torch.cat([self.upsample(fpn_out1), p3_in], 1))
        
        # Bottom-up pathway
        fpn_out3 = self.bu_block1(torch.cat([self.bu_conv1(fpn_out2), fpn_out1], 1))
        fpn_out4 = self.bu_block2(torch.cat([self.bu_conv2(fpn_out3), p5_in], 1))
        
        return (fpn_out2, fpn_out3, fpn_out4)


class DFL(torch.nn.Module):
    """Distribution Focal Loss module"""
    def __init__(self, ch=16):
        super().__init__()
        self.channels = ch
        self.integral_conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.integral_conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.channels, a).transpose(2, 1)
        return self.integral_conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    """Detection head"""
    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.dfl_channels = 16
        self.num_classes = nc
        self.num_layers = len(filters)
        self.num_outputs = nc + self.dfl_channels * 4
        self.stride = torch.zeros(self.num_layers)

        c1 = max(filters[0], self.num_classes)
        c2 = max((filters[0] // 4, self.dfl_channels * 4))

        self.dfl = DFL(self.dfl_channels)
        
        # Classification heads
        self.cls_heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                Conv(x, c1, 3),  # cls_conv1
                Conv(c1, c1, 3),  # cls_conv2
                torch.nn.Conv2d(c1, self.num_classes, 1)  # cls_pred
            ) for x in filters
        )
        
        # Box regression heads
        self.box_heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                Conv(x, c2, 3),  # box_conv1
                Conv(c2, c2, 3),  # box_conv2
                torch.nn.Conv2d(c2, 4 * self.dfl_channels, 1)  # box_pred
            ) for x in filters
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
            
        # Process through detection heads
        for i, (box_head, cls_head) in enumerate(zip(self.box_heads, self.cls_heads)):
            x[i] = torch.cat((box_head(x[i]), cls_head(x[i])), 1)
            
        if self.training:
            return x

        # Generate anchors for inference
        anchors, strides = make_anchors(x, self.stride, 0.5)
        self.anchors = anchors.transpose(0, 1)
        self.strides = strides.transpose(0, 1)
        
        # Process predictions
        tensor_list = []
        for item in x:
            if isinstance(item, torch.Tensor):
                reshaped = item.view(x[0].shape[0], self.num_outputs, -1)
                tensor_list.append(reshaped)
            elif isinstance(item, (int, float)):
                tensor_item = torch.full((x[0].shape[0], self.num_outputs, 1), item, 
                                    dtype=x[0].dtype, 
                                    device=x[0].device)
                tensor_list.append(tensor_item)
            else:
                print(f"Unexpected type encountered: {type(item)}")
                continue

        if tensor_list:
            x_cat = torch.cat(tensor_list, dim=2)
        else:
            raise ValueError("No valid tensors found to concatenate")
            
        # Split predictions
        box_preds, cls_preds = x_cat.split((self.dfl_channels * 4, self.num_classes), 1)
        
        # Process box predictions
        box_start, box_end = torch.split(self.dfl(box_preds), 2, 1)
        box_start = self.anchors.unsqueeze(0) - box_start
        box_end = self.anchors.unsqueeze(0) + box_end
        boxes = torch.cat(((box_start + box_end) / 2, box_end - box_start), 1)
        
        # Final output
        return torch.cat((boxes * self.strides, cls_preds.sigmoid()), 1)

    def initialize_biases(self):
        for box_head, cls_head, stride in zip(self.box_heads, self.cls_heads, self.stride):
            box_head[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_head[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / stride) ** 2)


class YOLO(torch.nn.Module):
    """Complete YOLO model"""
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.backbone = DarkNet(width, depth)
        self.neck = DarkFPN(width, depth)

        # Initialize detection head
        dummy_input = torch.zeros(1, 3, 256, 256)
        self.detection_head = Head(num_classes, (width[3], width[4], width[5]))
        self.detection_head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(dummy_input)])
        self.stride = self.detection_head.stride
        self.detection_head.initialize_biases()

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        return self.detection_head(list(neck_features))

    def fuse(self):
        """Fuse Conv+BN layers for inference"""
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'batch_norm'):
                m.conv_layer = fuse_conv(m.conv_layer, m.batch_norm)
                m.forward = m.fuse_forward
                delattr(m, 'batch_norm')
        return self

def yolo_v8_tiny(num_classes: int = 1):
    # depth = [1, 2, 2]
    # width = [3, 8, 8, 16, 16, 18]
    # return YOLO(width, depth, num_classes)
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 128]
    return YOLO(width, depth, num_classes)
    
def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)
