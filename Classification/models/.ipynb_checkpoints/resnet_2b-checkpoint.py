import torchvision
import torch
import torch.nn as nn
from .resnet import resnet18

class Fuse_Block(nn.Module):
    def __init__(self, in_planes):
        super(Fuse_Block, self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(in_planes//16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes//16, in_planes//16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(in_planes//16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes//16, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_planes, in_planes//16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += res
        x = self.conv(x)
        return x
        
        
class resnet18_2b(nn.Module):
    def __init__(self, num_classes=3):
        super(resnet18_2b, self).__init__() 
        self.model1 = resnet18(pretrained=True)
        self.model1 = nn.Sequential(*list(self.model1.children())[:-2])
        self.model2 = resnet18(pretrained=True)
        self.model2 = nn.Sequential(*list(self.model2.children())[:-2])
        
        self.fuse = Fuse_Block(512*2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512*2//16, out_features=3, bias=True)
        
    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fuse(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        
