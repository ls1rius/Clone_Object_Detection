import torchvision
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torchvision.models import ResNet as ResNet_official
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['ResNet', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
           'seresnet152', 'seresnext50_32x4d', 'seresnext101_32x8d',
           'wide_seresnet50_2', 'wide_seresnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(ResNet_official):
    def __init__(self,
        block,
        layers,
        num_classes = 3,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None):
        super(ResNet, self).__init__(block=block, 
                                     layers=layers, 
                                     num_classes=num_classes,
                                     zero_init_residual=zero_init_residual, 
                                     groups=groups, 
                                     width_per_group=width_per_group, 
                                     replace_stride_with_dilation=replace_stride_with_dilation, 
                                     norm_layer=norm_layer) 
        
        self.fc = torch.nn.Linear(self.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x    

def _resnet(
    arch,
    block,
    layers,
    pretrained,
    progress,
    **kwargs
):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # 去掉fc的参数
        state_dict = {k: v for k, v in state_dict.items() if (k in state_dict and 'fc' not in k)}
        model.load_state_dict(state_dict, strict=False)
    return model

def seresnet18(pretrained = False, progress = True, **kwargs):
    return _resnet('resnet18', SEBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def seresnet34(pretrained = False, progress = True, **kwargs):
    return _resnet('resnet34', SEBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
# get pretrained data
def seresnet50(pretrained = False, progress = True, **kwargs):
    model = _resnet('resnet50', SEBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model

def seresnet101(pretrained = False, progress = True, **kwargs):
    return _resnet('resnet101', SEBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def seresnet152(pretrained = False, progress = True, **kwargs):
    return _resnet('resnet152', SEBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def seresnext50_32x4d(pretrained = False, progress = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', SEBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def seresnext101_32x8d(pretrained = False, progress = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', SEBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_seresnet50_2(pretrained = False, progress = True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', SEBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def wide_seresnet101_2(pretrained = False, progress = True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', SEBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
