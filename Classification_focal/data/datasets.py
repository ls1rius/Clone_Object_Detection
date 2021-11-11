import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
import cv2
import string
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data.sampler import BatchSampler, Sampler

CLONE_CLASSES = ['holoclone', 'meroclone', 'paraclone']

        
def get_transforms(train=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_list = []
    if train:
        transforms_list += [
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        ]
    transforms_list += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    return torchvision.transforms.Compose(transforms_list)
    
class CloneDataSet(object):
    def __init__(self, img_path, anno_path, transforms):
        self.img_path = img_path
        self.csv_data = pd.read_csv(anno_path)
        self.data = self.load_data()
        self.transforms = transforms
        
    def load_data(self):
        objs = []
        for idx in range(len(self.csv_data)):
            x1 = self.csv_data.iloc[idx]['x1']
            y1 = self.csv_data.iloc[idx]['y1']
            x2 = self.csv_data.iloc[idx]['x2']
            y2 = self.csv_data.iloc[idx]['y2']
            label = CLONE_CLASSES.index(self.csv_data.iloc[idx]['label'].strip(string.digits))
            image_id = int(self.csv_data.iloc[idx]['image_id'])
            filename = self.csv_data.iloc[idx]['filename_gfp']
            if x2 >= x1 and y2 >= y1:
                objs.append([x1, y1, x2, y2, label, image_id, filename])
        return objs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # h,w,c
        img = cv2.imread(os.path.join(self.img_path, self.data[idx][-1]))
        x1, y1, x2, y2 = self.data[idx][:4]
        img = img[y1: y2, x1: x2]
        resized_img = cv2.resize(
            img,
            (224, 224), # (w, h)
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        img = self.transforms(Image.fromarray(resized_img))
        target = self.data[idx][-3]
        image_id = self.data[idx][-2]
        return img, target, image_id
        

        
class CloneDataSet_Test(object):
    def __init__(self, img_path, anno_path, transforms):
        self.img_path = img_path
        self.csv_data = pd.read_csv(anno_path)
        self.data = self.load_data()
        self.transforms = transforms
        
    def load_data(self):
        objs = []
        for idx in range(len(self.csv_data)):
            x1 = self.csv_data.iloc[idx]['x1']
            y1 = self.csv_data.iloc[idx]['y1']
            x2 = self.csv_data.iloc[idx]['x2']
            y2 = self.csv_data.iloc[idx]['y2']
            image_id = int(self.csv_data.iloc[idx]['image_id'])
            filename = self.csv_data.iloc[idx]['filename_gfp']
            if x2 >= x1 and y2 >= y1:
                objs.append([x1, y1, x2, y2, image_id, filename])
        return objs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # h,w,c
        img = cv2.imread(os.path.join(self.img_path, self.data[idx][-1]))
        image_id = self.data[idx][-2]
        x1, y1, x2, y2 = self.data[idx][:4]
        img = img[y1: y2, x1: x2]
        resized_img = cv2.resize(
            img,
            (224, 224), # (w, h)
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        img = self.transforms(Image.fromarray(resized_img))
        return img, image_id
