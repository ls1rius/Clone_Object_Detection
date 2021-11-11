import torch
import torch.nn as nn
from collections import OrderedDict

from .deform_conv import DeformConv2d

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_dcn=True,
                 num_deformable_groups=1,
                 dcn_offset_lr_mult=0.1,
                 use_regular_conv_on_stride=False):
        super(BasicBlock, self).__init__()
        conv1_stride = 1
        conv2_stride = stride
        self.with_dcn = with_dcn
        if use_regular_conv_on_stride and stride > 1:
            self.with_dcn = False
            
        self.conv1 = conv3x3(inplanes, planes, conv1_stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        
        if self.with_dcn:
            self.conv2_offset = nn.Conv2d(
                planes,
                num_deformable_groups * 18, # 3x3 x 2
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2_offset.lr_mult = dcn_offset_lr_mult
            self.conv2_offset.zero_init = True
            self.conv2 = DeformConv2d(planes, planes, (3, 3), stride=conv2_stride,
                padding=dilation, dilation=dilation,
                groups=num_deformable_groups)
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
            
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.with_dcn:
            offset = self.conv2_offset(out)
            # add bias to the offset to solve the bug of dilation rates within dcn.
            dilation = self.conv2.dilation[0]
            bias_w = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).to(x.device) * (dilation - 1)
            bias_h = bias_w.permute(1, 0)
            bias_w.requires_grad = False
            bias_h.requires_grad = False
            offset += torch.cat([bias_h.reshape(-1), bias_w.reshape(-1)]).view(1, -1, 1, 1)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_dcn=False,
                 num_deformable_groups=1,
                 dcn_offset_lr_mult=0.1,
                 use_regular_conv_on_stride=False):
        super(Bottleneck, self).__init__()
        conv1_stride = 1
        conv2_stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)

        self.with_dcn = with_dcn
        if use_regular_conv_on_stride and stride > 1:
            self.with_dcn = False
            
        if self.with_dcn:
            self.conv2_offset = nn.Conv2d(
                planes,
                num_deformable_groups * 18,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2_offset.lr_mult = dcn_offset_lr_mult
            self.conv2_offset.zero_init = True
            self.conv2 = DeformConv2d(planes, planes, (3, 3), stride=conv2_stride,
                padding=dilation, dilation=dilation,
                groups=num_deformable_groups)
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if self.with_dcn:
                offset = self.conv2_offset(out)
                # add bias to the offset to solve the bug of dilation rates within dcn.
                dilation = self.conv2.dilation[0]
                bias_w = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).to(x.device) * (dilation - 1)
                bias_h = bias_w.permute(1, 0)
                bias_w.requires_grad = False
                bias_h.requires_grad = False
                offset += torch.cat([bias_h.reshape(-1), bias_w.reshape(-1)]).view(1, -1, 1, 1)
                out = self.conv2(out, offset)
            else:
                out = self.conv2(out)
                
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out = out + residual
            return out


        out = _inner_forward(x)
        out = self.relu_in(out)

        return out

class DCNResNet(nn.Module):
    """ResNet backbone.
    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """
    def __init__(self,
                 block,
                 layers,
                 num_classes=3,
                 deep_base=True,
                 norm_layer=None,
                 replace_stride_with_dilation=None):
        super(DCNResNet, self).__init__()

        self.inplanes = 64
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [1, 1, 1]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(self.inplanes)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))
            ]))
        else:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False))]
            ))
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_res_layer(block, 64, layers[0], 
                                          with_dcn=False)
        self.layer2 = self.make_res_layer(block, 128, layers[1], stride=2, 
                                          dilation=replace_stride_with_dilation[0],
                                          with_dcn=False)
        self.layer3 = self.make_res_layer(block, 256, layers[2], stride=2,
                                          dilation=replace_stride_with_dilation[1],
                                          with_dcn=True)
        self.layer4 = self.make_res_layer(block, 512, layers[3], stride=2,
                                          dilation=replace_stride_with_dilation[2],
                                          with_dcn=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def make_res_layer(self,
                       block,
                       planes,
                       blocks,
                       stride=1,
                       dilation=1,
                       with_dcn=False,
                       dcn_offset_lr_mult=0.1,
                       use_regular_conv_on_stride=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation,
                downsample,
                with_dcn=with_dcn,
                dcn_offset_lr_mult=dcn_offset_lr_mult,
                use_regular_conv_on_stride=use_regular_conv_on_stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, 1, dilation, with_dcn=with_dcn, 
                      dcn_offset_lr_mult=dcn_offset_lr_mult, use_regular_conv_on_stride=use_regular_conv_on_stride))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def dcnresnet18(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DCNResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, deep_base=False, **kwargs)
    return model

def dcnresnet34(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DCNResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, deep_base=False, **kwargs)
    return model
    
def dcnresnet50(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DCNResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, deep_base=True, **kwargs)
    return model

def dcnresnet101(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DCNResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, deep_base=True, **kwargs)
    return model

def dcnresnet152(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DCNResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, deep_base=True, **kwargs)
    return model

